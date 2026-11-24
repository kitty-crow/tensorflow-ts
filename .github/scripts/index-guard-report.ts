/** Index a full Code Guards report for bounded, connector-only reads. */
import {mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {join, dirname} from 'node:path';
import {createHash} from 'node:crypto';

const [input, worktree, source, detector, runId] = process.argv.slice(2);
if (!input || !worktree || !source || !detector || !runId ||
    !/^[a-f0-9]{40}$/.test(source) || !/^[a-f0-9]{40}$/.test(detector)) {
  throw new Error('Expected report, worktree, source SHA, detector SHA and run ID.');
}
const raw = readFileSync(input, 'utf8');
const findings = JSON.parse(raw);
if (!Array.isArray(findings)) throw new Error('Report is not an array.');
const root = 'reports/' + source;
const hash = createHash('sha256').update(raw).digest('hex');
const groups = new Map<string, typeof findings>();
const counts = new Map<string, number>();
const fileCounts = new Map<string, number>();
const messageCounts = new Map<string, Map<string, number>>();

function increment(map: Map<string, number>, key: string) {
  map.set(key, (map.get(key) ?? 0) + 1);
}
function write(relative: string, value: string) {
  const path = join(worktree, relative);
  mkdirSync(dirname(path), {recursive: true});
  writeFileSync(path, value);
}
function ranked(map: Map<string, number>, limit: number) {
  return [...map].sort((a, b) => b[1] - a[1]).slice(0, limit)
      .map(([key, count]) => ({key, count}));
}
for (const finding of findings) {
  const {code, location, message} = finding;
  if (typeof code !== 'string' || !/^CG\d+$/.test(code) ||
      typeof location?.file !== 'string' || typeof message !== 'string') {
    throw new Error('Invalid finding shape.');
  }
  const first = location.file.split('/')[0] ?? '';
  const pkg = /^[a-zA-Z0-9_-]+$/.test(first) && first !== location.file ? first : 'root';
  const key = code + '/' + pkg;
  let group = groups.get(key);
  if (!group) {
    group = [];
    groups.set(key, group);
  }
  group.push(finding);
  increment(counts, code);
  increment(fileCounts, code + ':' + location.file);
  let messages = messageCounts.get(code);
  if (!messages) {
    messages = new Map<string, number>();
    messageCounts.set(code, messages);
  }
  increment(messages, message);
}
const parts: {path: string; count: number; code: string; package: string}[] = [];
for (const [key, group] of [...groups].sort(([a], [b]) => a.localeCompare(b))) {
  const [code, pkg] = key.split('/');
  if (!code || !pkg) throw new Error('Invalid group.');
  let chunk: string[] = [];
  let bytes = 2;
  let sequence = 0;
  function flush() {
    if (!chunk.length) return;
    const path = root + '/' + key + '/' + String(sequence++).padStart(4, '0') + '.json';
    write(path, '[' + chunk.join(',') + ']\n');
    parts.push({path, count: chunk.length, code, package: pkg});
    chunk = [];
    bytes = 2;
  }
  for (const finding of group) {
    const item = JSON.stringify(finding);
    const size = Buffer.byteLength(item, 'utf8') + 1;
    if (size > 350000) throw new Error('One finding exceeds the chunk limit.');
    if (bytes + size > 350000 || chunk.length >= 200) flush();
    chunk.push(item);
    bytes += size;
  }
  flush();
}
if (parts.reduce((sum, part) => sum + part.count, 0) !== findings.length) {
  throw new Error('Report indexing lost findings.');
}
const full = root + '/findings.json';
const index = root + '/index.json';
write(full, raw);
write(index, JSON.stringify({
  source, detector, runId, total: findings.length, sha256: hash, full,
  scoring: 'native FP32 CPU; reuse matching WebGPU cache entries; isolated CPU misses',
  rules: ranked(counts, counts.size),
  topFiles: ranked(fileCounts, 80),
  topMessages: Object.fromEntries([...messageCounts].map(([code, map]) => [code, ranked(map, 15)])),
  parts
}, null, 2) + '\n');
write('reports/latest.json', JSON.stringify({
  source, detector, runId, total: findings.length, index, full, sha256: hash
}, null, 2) + '\n');
console.log('Published report index:', index, 'parts:', parts.length, 'findings:', findings.length);
