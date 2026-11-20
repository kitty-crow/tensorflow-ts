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

function indentation(source: ts.SourceFile, position: number): string {
  const location = source.getLineAndCharacterOfPosition(position);
  const lineStart = source.getPositionOfLineAndCharacter(location.line, 0);
  const leading = source.text.slice(lineStart, position);
  return /^[ \t]*$/.test(leading) ? leading : '';
}

function returnOrThrow(statement: ts.Statement): boolean {
  if (ts.isReturnStatement(statement) || ts.isThrowStatement(statement)) return true;
  if (!ts.isBlock(statement)) return false;
  const last = statement.statements.at(-1);
  return last !== undefined && (ts.isReturnStatement(last) || ts.isThrowStatement(last));
}

function nestedConditional(statement: ts.Statement): boolean {
  let nested = false;
  const visit = (node: ts.Node): void => {
    if (ts.isIfStatement(node)) nested = true;
    if (ts.isFunctionLike(node)) return;
    ts.forEachChild(node, visit);
  };
  ts.forEachChild(statement, visit);
  return nested;
}

/**
 * A terminating fallback may be lifted to an early guard. Keeping the then
 * block as a standalone block preserves let/const scope and abrupt completion.
 */
function liftTerminalFallback(node: ts.IfStatement, source: ts.SourceFile): Edit | null {
  const fallback = node.elseStatement;
  if (fallback === undefined || !returnOrThrow(fallback) ||
      !nestedConditional(node.thenStatement) || !ts.isBlock(node.parent)) return null;
  const between = source.text.slice(node.thenStatement.end, fallback.getStart(source));
  if (!/^\s*else\s*$/.test(between)) return null;
  const pad = indentation(source, node.getStart(source));
  return {
    start: node.getStart(source),
    end: node.end,
    replacement: 'if (!(' + node.expression.getText(source) + ')) ' +
      fallback.getText(source) + '\n' + pad + node.thenStatement.getText(source),
    kind: 'terminal fallback guard'
  };
}

/** Flatten a final guarded body when skipping it is equivalent to exiting the scope. */
function liftFinalGuard(node: ts.IfStatement, source: ts.SourceFile): Edit | null {
  if (node.elseStatement !== undefined || !nestedConditional(node.thenStatement)) return null;
  const block = node.parent;
  if (!ts.isBlock(block) || block.statements.at(-1) !== node) return null;
  const owner = block.parent;
  let exit: string | null = null;
  if ((ts.isForStatement(owner) || ts.isForOfStatement(owner) ||
      ts.isForInStatement(owner) || ts.isWhileStatement(owner) ||
      ts.isDoStatement(owner)) && (owner.statement === block)) exit = 'continue;';
  if (ts.isFunctionLike(owner) && owner.body === block) {
    const returnType = owner.type === undefined ? '' : owner.type.getText(source);
    let returnsValue = false;
    const inspect = (entry: ts.Node): void => {
      if (ts.isFunctionLike(entry) && entry !== owner) return;
      if (ts.isReturnStatement(entry) && entry.expression !== undefined) returnsValue = true;
      ts.forEachChild(entry, inspect);
    };
    inspect(block);
    if ((returnType === '' && !returnsValue) ||
        returnType === 'void' || returnType === 'Promise<void>') exit = 'return;';
  }
  if (exit === null) return null;
  const pad = indentation(source, node.getStart(source));
  return {
    start: node.getStart(source),
    end: node.end,
    replacement: 'if (!(' + node.expression.getText(source) + ')) ' + exit +
      '\n' + pad + node.thenStatement.getText(source),
    kind: 'final-scope guard'
  };
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
      node.thenStatement.getText(source) + '\n' +
      indentation(source, node.getStart(source)) + branch.getText(source),
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
  const pad = indentation(source, node.getStart(source));
  const caseBody = (body: ts.Statement): string => {
    const text = body.getText(source);
    const raw = ts.isBlock(body) ? text.slice(1, -1) : text;
    const lines = raw.replace(/^\s*\n/, '').replace(/\n\s*$/, '').split('\n');
    const nonempty = lines.filter(line => line.trim().length > 0);
    const margin = Math.min(...nonempty.map(line => (line.match(/^[ \t]*/) ?? [''])[0].length));
    return lines.map(line => pad + '    ' + line.slice(margin)).join('\n');
  };
  const caseStrings = choices.map(choice =>
    pad + '  case ' + choice.literal + ': {\n' +
    caseBody(choice.body) +
    (returnOrThrow(choice.body) ? '' : '\n' + pad + '    break;') +
    '\n' + pad + '  }');
  if (otherwise !== undefined) {
    caseStrings.push(pad + '  default: {\n' + caseBody(otherwise) + '\n' + pad + '  }');
  }
  return {
    start: node.getStart(source),
    end: node.end,
    replacement: 'switch (' + first.identifier + ') {\n' +
      caseStrings.join('\n') + '\n' + pad + '}',
    kind: 'literal-dispatch switch'
  };
}

function findEdits(source: ts.SourceFile): Edit[] {
  const options: Edit[] = [];
  const walk = (node: ts.Node): void => {
    if (ts.isIfStatement(node)) {
      const replacement = mergeConjunction(node, source) ??
        switchDispatch(node, source) ??
        liftTerminalFallback(node, source) ??
        liftFinalGuard(node, source) ??
        flattenTerminalElse(node, source);
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
