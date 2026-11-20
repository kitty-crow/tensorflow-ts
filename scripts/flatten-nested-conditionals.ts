import { readFileSync, writeFileSync } from 'node:fs';
import ts from 'typescript';

/**
 * Conservative AST-guided flattening.
 *
 * Only rewrites patterns with equivalent short-circuit evaluation and
 * control-flow behaviour. No changes are made to complex branching.
 */
type Edit = { start: number; end: number; replacement: string; kind: string };

const configuration = ts.readConfigFile('tsconfig.code-guards.json', ts.sys.readFile);
if (configuration.error) {
  throw new Error(ts.flattenDiagnosticMessageText(configuration.error.messageText, '\n'));
}
const parsed = ts.parseJsonConfigFileContent(configuration.config, ts.sys, '.');
if (parsed.errors.length) {
  throw new Error(parsed.errors.map(d => ts.flattenDiagnosticMessageText(d.messageText, '\n')).join('\n'));
}

function hasComments(value: string): boolean {
  return value.includes('//') || value.includes('/*');
}

function returnOrThrow(statement: ts.Statement): boolean {
  if (ts.isReturnStatement(statement) || ts.isThrowStatement(statement)) return true;
  if (!ts.isBlock(statement)) return false;
  const last = statement.statements.at(-1);
  return last !== undefined && (ts.isReturnStatement(last) || ts.isThrowStatement(last));
}

function mergeConjunction(node: ts.IfStatement, source: ts.SourceFile): Edit | null {
  if (node.elseStatement !== undefined || !ts.isBlock(node.thenStatement)) return null;
  const outer = node.thenStatement;
  if (outer.statements.length !== 1) return null;
  const child = outer.statements[0];
  if (child === undefined || !ts.isIfStatement(child) || child.elseStatement !== undefined) return null;

  const before = source.text.slice(outer.getStart(source) + 1, child.getStart(source));
  const after = source.text.slice(child.end, outer.end - 1);
  const conditionGap = source.text.slice(node.getStart(source), node.expression.getStart(source));
  const childGap = source.text.slice(child.getStart(source), child.expression.getStart(source));
  if (before.trim() || after.trim() || hasComments(conditionGap) || hasComments(childGap)) return null;
  const left = node.expression.getText(source);
  const right = child.expression.getText(source);
  return {
    start: node.getStart(source),
    end: node.end,
    replacement: 'if ((' + left + ') && (' + right + ')) ' + child.thenStatement.getText(source),
    kind: 'conjoined guards'
  };
}

function flattenTerminalElse(node: ts.IfStatement, source: ts.SourceFile): Edit | null {
  const branch = node.elseStatement;
  if (branch === undefined || !ts.isIfStatement(branch) || !returnOrThrow(node.thenStatement)) return null;
  const parent = node.parent;
  if (!ts.isBlock(parent) && !ts.isSourceFile(parent) && !ts.isCaseBlock(parent)) return null;
  const gap = source.text.slice(node.thenStatement.end, branch.getStart(source));
  if (!/^\s*else\s*$/.test(gap)) return null;
  return {
    start: node.getStart(source),
    end: node.end,
    replacement: 'if (' + node.expression.getText(source) + ') ' +
      node.thenStatement.getText(source) + '\n' + branch.getText(source),
    kind: 'terminal else-if'
  };
}

type Choice = { literal: string; body: ts.Statement; ifNode: ts.IfStatement };

function equality(node: ts.IfStatement, source: ts.SourceFile): { identifier: string; literal: string } | null {
  const expression = node.expression;
  if (!ts.isBinaryExpression(expression) || expression.operatorToken.kind !== ts.SyntaxKind.EqualsEqualsEqualsToken) return null;
  const left = expression.left;
  const right = expression.right;
  if (!ts.isIdentifier(left)) return null;
  if (!ts.isStringLiteral(right) && !ts.isNumericLiteral(right) &&
      right.kind !== ts.SyntaxKind.TrueKeyword && right.kind !== ts.SyntaxKind.FalseKeyword) return null;
  return { identifier: left.getText(source), literal: right.getText(source) };
}

function unlabelledBreak(node: ts.Node): boolean {
  let found = false;
  const visit = (entry: ts.Node): void => {
    if (ts.isBreakStatement(entry) && entry.label === undefined) found = true;
    ts.forEachChild(entry, visit);
  };
  visit(node);
  return found;
}

function switchDispatch(node: ts.IfStatement, source: ts.SourceFile): Edit | null {
  const first = equality(node, source);
  if (first === null) return null;
  const choices: Choice[] = [];
  let next: ts.IfStatement = node;
  while (true) {
    const current = equality(next, source);
    if (current === null || current.identifier !== first.identifier ||
        unlabelledBreak(next.thenStatement)) return null;
    choices.push({ literal: current.literal, body: next.thenStatement, ifNode: next });
    const alternative = next.elseStatement;
    if (alternative === undefined || !ts.isIfStatement(alternative)) break;
    const between = source.text.slice(next.thenStatement.end, alternative.getStart(source));
    if (!/^\s*else\s*$/.test(between)) return null;
    next = alternative;
  }
  if (choices.length < 2) return null;
  const otherwise = next.elseStatement;
  if (otherwise !== undefined && unlabelledBreak(otherwise)) return null;
  const caseStrings = choices.map(choice =>
    '  case ' + choice.literal + ': {\n' +
    '    ' + choice.body.getText(source) + '\n    break;\n  }');
  if (otherwise !== undefined) caseStrings.push('  default: {\n    ' + otherwise.getText(source) + '\n  }');
  return {
    start: node.getStart(source),
    end: node.end,
    replacement: 'switch (' + first.identifier + ') {\n' + caseStrings.join('\n') + '\n}',
    kind: 'literal-dispatch switch'
  };
}

function findEdits(source: ts.SourceFile): Edit[] {
  const options: Edit[] = [];
  const walk = (node: ts.Node): void => {
    if (ts.isIfStatement(node)) {
      const replacement = mergeConjunction(node, source) ??
        switchDispatch(node, source) ?? flattenTerminalElse(node, source);
      if (replacement !== null) options.push(replacement);
    }
    ts.forEachChild(node, walk);
  };
  walk(source);
  const accepted: Edit[] = [];
  let previousEnd = -1;
  for (const choice of options.sort((a, b) => a.start - b.start || b.end - a.end)) {
    if (choice.start < previousEnd) continue;
    accepted.push(choice);
    previousEnd = choice.end;
  }
  return accepted;
}

function rewrite(filename: string, original: string): { text: string; changes: number; kinds: string[] } {
  let text = original;
  const kinds: string[] = [];
  for (let pass = 0; pass < 16; pass += 1) {
    const source = ts.createSourceFile(filename, text, ts.ScriptTarget.Latest, true);
    const edits = findEdits(source);
    if (!edits.length) break;
    for (const edit of edits.slice().reverse()) {
      text = text.slice(0, edit.start) + edit.replacement + text.slice(edit.end);
      kinds.push(edit.kind);
    }
  }
  return {text, changes: kinds.length, kinds};
}

const apply = process.argv.includes('--apply');
const files = parsed.fileNames.filter(filename =>
  /\.(?:ts|tsx)$/.test(filename) &&
  !/[/\\](?:node_modules|vendor|third_party|third-party|dist|build)[/\\]/.test(filename));
let changedFiles = 0;
let changes = 0;
const summary = new Map<string, number>();
for (const filename of files) {
  const before = readFileSync(filename, 'utf8');
  const result = rewrite(filename, before);
  if (!result.changes) continue;
  const check = ts.transpileModule(result.text, {
    fileName: filename,
    compilerOptions: {target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext},
    reportDiagnostics: true
  });
  const errors = (check.diagnostics ?? []).filter(d => d.category === ts.DiagnosticCategory.Error);
  if (errors.length) throw new Error('Transformed file has syntax errors: ' + filename);
  changedFiles += 1;
  changes += result.changes;
  for (const kind of result.kinds) summary.set(kind, (summary.get(kind) ?? 0) + 1);
  if (apply) writeFileSync(filename, result.text);
}
console.log(JSON.stringify({mode: apply ? 'apply' : 'dry-run', scannedFiles: files.length,
  changedFiles, changes, kinds: Object.fromEntries(summary)}, null, 2));
