import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';

const ref = '5940117af9afc86a0994d959f50603aeeeac1f2d';
const base = `https://raw.githubusercontent.com/kitty-crow/nsfwjs/${ref}/models/mobilenet_v2/`;
const worker = fileURLToPath(new URL('./avatar_photo_worker.mjs', import.meta.url));
const dir = await mkdtemp(join(tmpdir(), 'tfjs-avatar-smoke-'));
const node = process.env.MAE_NODE_BINARY?.trim() || 'node';
const startupTimeoutMs = 600_000;
const inferenceTimeoutMs = 60_000;

const download = async name => {
  const response = await fetch(`${base}${name}`);
  if (!response.ok) throw new Error(`Failed to fetch ${name}: HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength < 1) throw new Error(`Downloaded empty ${name}`);
  await writeFile(join(dir, name), bytes);
};

const withTimeout = async (promise, ms, label) => {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms} ms`)), ms);
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
};

let child;
let lines;
try {
  console.error('[Avatar smoke] downloading pinned photographic model…');
  await Promise.all([download('model.json'), download('group1-shard1of1')]);

  console.error(`[Avatar smoke] starting worker with ${node}…`);
  child = spawn(node, [worker, dir], {
    stdio: ['pipe', 'pipe', 'inherit'],
  });
  const spawnError = new Promise((_, reject) => child.once('error', reject));
  const exitPromise = once(child, 'exit');
  lines = createInterface({ input: child.stdout, crlfDelay: Infinity });
  const it = lines[Symbol.asyncIterator]();

  const nextJson = async (label, timeoutMs) => {
    const next = await withTimeout(Promise.race([it.next(), spawnError]), timeoutMs, label);
    if (next.done) throw new Error(`${label}: worker stdout closed`);
    try {
      return JSON.parse(next.value);
    } catch (error) {
      throw new Error(`${label}: worker returned invalid JSON: ${next.value}`, { cause: error });
    }
  };

  const writeRequest = async (payload, label) => {
    await withTimeout(new Promise((resolve, reject) => {
      child.stdin.write(`${JSON.stringify(payload)}\n`, error => error ? reject(error) : resolve());
    }), 30_000, `${label} write`);
  };

  const ready = await nextJson('Avatar worker startup', startupTimeoutMs);
  if (ready?.error) throw new Error(`Avatar worker startup failed: ${ready.error}`);
  if (ready?.ready !== true) throw new Error(`Avatar worker did not become ready: ${JSON.stringify(ready)}`);

  const send = async (id, width, height, fill) => {
    const rgb = Buffer.alloc(width * height * 3, fill).toString('base64');
    await writeRequest({ id, width, height, rgb }, `Avatar worker inference ${id}`);
    const result = await nextJson(`Avatar worker inference ${id}`, inferenceTimeoutMs);
    if (typeof result?.error === 'string') throw new Error(`Avatar worker inference ${id} failed: ${result.error}`);
    if (result?.id !== id) throw new Error(`Avatar worker inference id mismatch: expected ${id}, got ${result?.id}`);
    for (const key of ['person', 'photographic', 'score']) {
      if (typeof result?.[key] !== 'number' || !Number.isFinite(result[key]) || result[key] < 0 || result[key] > 1) {
        throw new Error(`Invalid avatar worker ${key}: ${result?.[key]}`);
      }
    }
    const expected = Math.min(result.person, result.photographic);
    if (Math.abs(result.score - expected) > 1e-7) {
      throw new Error(`Avatar worker score mismatch: got ${result.score}, expected ${expected}`);
    }
    console.error(`[Avatar smoke] ${id} ok: person=${result.person.toFixed(4)} photographic=${result.photographic.toFixed(4)} score=${result.score.toFixed(4)}`);
  };

  const expectError = async () => {
    const id = 'invalid-rgb';
    await writeRequest({ id, width: 2, height: 2, rgb: '' }, 'Avatar worker invalid request');
    const result = await nextJson('Avatar worker invalid request', inferenceTimeoutMs);
    if (result?.id !== id || typeof result?.error !== 'string' || !result.error.includes('Invalid RGB byte length')) {
      throw new Error(`Avatar worker invalid-request contract failed: ${JSON.stringify(result)}`);
    }
    console.error('[Avatar smoke] invalid request rejected without killing worker.');
  };

  await send('black-256', 256, 256, 0);
  await expectError();
  await send('white-300', 300, 300, 255);

  child.stdin.end();
  const [code, signal] = await withTimeout(exitPromise, 30_000, 'Avatar worker shutdown');
  if (code !== 0) throw new Error(`Avatar worker exited ${code ?? `by signal ${signal ?? 'unknown'}`}`);
  child = undefined;
  console.log('Photographic-person worker smoke passed.');
} finally {
  lines?.close();
  if (child !== undefined && !child.killed) child.kill('SIGTERM');
  await rm(dir, { recursive: true, force: true });
}
