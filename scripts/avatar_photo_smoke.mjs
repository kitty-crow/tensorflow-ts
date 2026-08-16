import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';

const ref = '5940117af9afc86a0994d959f50603aeeeac1f2d';
const base = `https://raw.githubusercontent.com/kitty-crow/nsfwjs/${ref}/models/mobilenet_v2/`;
const worker = fileURLToPath(new URL('./avatar_photo_worker.mjs', import.meta.url));
const dir = await mkdtemp(join(tmpdir(), 'tfjs-avatar-smoke-'));
const node = process.env.MAE_NODE_BINARY?.trim() || 'node';

const download = async name => {
  const response = await fetch(`${base}${name}`);
  if (!response.ok) throw new Error(`Failed to fetch ${name}: HTTP ${response.status}`);
  await writeFile(join(dir, name), new Uint8Array(await response.arrayBuffer()));
};

try {
  await Promise.all([download('model.json'), download('group1-shard1of1')]);
  const child = spawn(node, [worker, dir], {
    stdio: ['pipe', 'pipe', 'inherit'],
  });
  const lines = createInterface({ input: child.stdout, crlfDelay: Infinity });
  const it = lines[Symbol.asyncIterator]();
  const ready = await it.next();
  if (ready.done || JSON.parse(ready.value).ready !== true) throw new Error('Avatar worker did not become ready');

  const rgb = Buffer.alloc(300 * 300 * 3).toString('base64');
  child.stdin.write(`${JSON.stringify({ id: 'smoke', width: 300, height: 300, rgb })}\n`);
  child.stdin.end();

  const resultLine = await it.next();
  if (resultLine.done) throw new Error('Avatar worker returned no result');
  const result = JSON.parse(resultLine.value);
  for (const key of ['person', 'photographic', 'score']) {
    if (typeof result[key] !== 'number' || !Number.isFinite(result[key]) || result[key] < 0 || result[key] > 1) {
      throw new Error(`Invalid avatar worker ${key}: ${result[key]}`);
    }
  }
  const exit = await new Promise((resolve, reject) => {
    child.once('error', reject);
    child.once('exit', code => resolve(code));
  });
  if (exit !== 0) throw new Error(`Avatar worker exited ${exit}`);
  console.log('Photographic-person worker smoke passed.');
} finally {
  await rm(dir, { recursive: true, force: true });
}
