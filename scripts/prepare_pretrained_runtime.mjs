import { spawnSync, execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { chmod, cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import { access } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import process from 'node:process';

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, '..');
const runtime = resolve(repo, '.cache', 'pretrained-runtime');
const tfDst = resolve(runtime, 'node_modules', '@tensorflow');
const stamp = resolve(runtime, 'stamp');
const bazeliskVersion = '1.29.0';
const packages = ['tfjs-core', 'tfjs-backend-cpu', 'tfjs-converter', 'tfjs-layers'];
const targets = packages.map(name => `//${name}:${name}_pkg`);
const runtimeDependencies = {
  long: '4.0.0',
  'node-fetch': '2.6.1',
  seedrandom: '3.0.5',
};

const assets = {
  'linux-x64': ['bazelisk-linux-amd64', '5a408715e932c0250d28bd84555f12edbf70117de42f9181691c736eacc4a992'],
  'linux-arm64': ['bazelisk-linux-arm64', 'e20e8b0f4f240091b7a55bf17b9398bd4f40ee70ae0208dff95dd4c445fb4010'],
  'darwin-x64': ['bazelisk-darwin-amd64', '16c3d7aa15323a9fb69f56c7ec5733ed18bedb786680d0ba13bb12a3c8083007'],
  'darwin-arm64': ['bazelisk-darwin-arm64', 'cee851f726789227d5561004e9904a52be45c3efb56f8b38b6993d6adbaa0409'],
  'win32-x64': ['bazelisk-windows-amd64.exe', '092a8738d5b41aae7a85c42cc961b1034e3389aba43ffc20c0fabda7b43e095b'],
  'win32-arm64': ['bazelisk-windows-arm64.exe', '8bc42bd5d7857f18a21440b906469bb6c7cf91a7c72364d4b1e5ec56a76fe94f'],
};

const exists = async path => await access(path, fsConstants.F_OK).then(() => true).catch(() => false);
const gitHead = () => execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repo, encoding: 'utf8' }).trim();
const stage = message => console.error(`[TensorFlow runtime] ${message}`);
const childStdio = ['ignore', 2, 2];

const sha256 = async path => createHash('sha256').update(await readFile(path)).digest('hex');

const bazelisk = async () => {
  const key = `${process.platform}-${process.arch}`;
  const asset = assets[key];
  if (asset === undefined) throw new Error(`Unsupported Bazelisk platform ${key}`);
  const [name, expected] = asset;
  const path = resolve(repo, '.cache', `bazelisk-${bazeliskVersion}-${name}`);
  await mkdir(dirname(path), { recursive: true });

  let valid = await exists(path) && await sha256(path) === expected;
  if (!valid) {
    stage(`downloading Bazelisk ${bazeliskVersion} for ${key}…`);
    const url = `https://github.com/bazelbuild/bazelisk/releases/download/v${bazeliskVersion}/${name}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Bazelisk download failed: HTTP ${response.status}`);
    await writeFile(path, new Uint8Array(await response.arrayBuffer()), { mode: 0o755 });
    valid = await sha256(path) === expected;
  }
  if (!valid) throw new Error('Bazelisk checksum mismatch');
  if (process.platform !== 'win32') await chmod(path, 0o755);
  return path;
};

const runtimeDependencyPaths = () => Object.keys(runtimeDependencies)
  .map(name => resolve(runtime, 'node_modules', name, 'package.json'));

const staged = async head => {
  if (!await exists(stamp)) return false;
  if ((await readFile(stamp, 'utf8')).trim() !== head) return false;
  const paths = [
    ...packages.map(name => resolve(tfDst, name, 'package.json')),
    ...runtimeDependencyPaths(),
  ];
  return (await Promise.all(paths.map(exists))).every(Boolean);
};

const installRuntimeDependencies = async () => {
  const pkg = `${JSON.stringify({ private: true, dependencies: runtimeDependencies }, null, 2)}\n`;
  await writeFile(resolve(runtime, 'package.json'), pkg);
  const isBun = typeof process.versions.bun === 'string';
  const command = isBun ? process.execPath : 'npm';
  const args = isBun
    ? ['install', '--ignore-scripts']
    : ['install', '--ignore-scripts', '--omit=dev', '--no-audit', '--no-fund', '--package-lock=false'];
  stage('installing prepared runtime dependencies…');
  const result = spawnSync(command, args, { cwd: runtime, stdio: childStdio });
  if (result.error !== undefined) throw result.error;
  if (result.status !== 0) throw new Error(`Runtime dependency install failed with exit ${result.status ?? 'unknown'}`);
};

export const prepareRuntime = async () => {
  const head = gitHead();
  if (await staged(head)) {
    stage('using cached prepared runtime.');
    return runtime;
  }

  stage('building TensorFlow.js core, CPU backend, converter and layers…');
  const bin = await bazelisk();
  const result = spawnSync(bin, ['build', ...targets], { cwd: repo, stdio: childStdio });
  if (result.error !== undefined) throw result.error;
  if (result.status !== 0) throw new Error(`Bazel runtime build failed with exit ${result.status ?? 'unknown'}`);

  stage('staging built TensorFlow.js packages…');
  await rm(resolve(runtime, 'node_modules'), { recursive: true, force: true });
  await mkdir(tfDst, { recursive: true });
  for (const name of packages) {
    const src = resolve(repo, 'dist', 'bin', name, `${name}_pkg`);
    if (!await exists(resolve(src, 'package.json'))) throw new Error(`Built TensorFlow package is missing: ${name}`);
    await cp(src, resolve(tfDst, name), { recursive: true });
  }
  await installRuntimeDependencies();
  await writeFile(stamp, `${head}\n`);
  stage('prepared runtime is ready.');
  return runtime;
};

if (resolve(process.argv[1] ?? '') === fileURLToPath(import.meta.url)) {
  const path = await prepareRuntime();
  console.log(`Pretrained TensorFlow runtime ready: ${path}`);
}
