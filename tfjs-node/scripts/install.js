/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const fs = require('node:fs');
let path = require('node:path');
const cp = require('node:child_process');
const os = require('node:os');
const {
  depsPath,
  depsLibPath,
  depsLibTensorFlowPath,
  LIBTENSORFLOW_VERSION,
  PLATFORM_MAPPING,
  ARCH_MAPPING,
  PLATFORM_EXTENSION,
  ALL_SUPPORTED_COMBINATION,
  modulePath,
  customTFLibUri,
  customAddon
} = require('./deps-constants.js');
const resources = require('./resources');
const {addonName} = require('./get-addon-name.js');

const CDN_STORAGE = process.env.TFJS_NODE_CDN_STORAGE ||
    process.env.npm_config_TFJS_NODE_CDN_STORAGE || process.env.CDN_STORAGE;
const BASE_HOST = CDN_STORAGE || 'https://storage.googleapis.com/';
const BASE_URI = process.env.TFJS_NODE_BASE_URI ||
    process.env.npm_config_TFJS_NODE_BASE_URI ||
    `${BASE_HOST}tensorflow/libtensorflow/libtensorflow-`;

const platform = os.platform();
if (platform === 'win32') {
  path = path.win32;
}
let libType = process.argv[2] === undefined ? 'cpu' : process.argv[2];
let system = `${libType}-${PLATFORM_MAPPING[platform]}-` +
    `${ARCH_MAPPING[os.arch()]}`;
const forceDownload = process.argv[3] === undefined ? undefined : process.argv[3];

let packageJsonFile;

function setPackageJsonFile() {
  packageJsonFile = JSON.parse(
      fs.readFileSync(`${__dirname}/../package.json`, 'utf8'));
}

function updateAddonName() {
  if (customAddon !== undefined) {
    Object.assign(packageJsonFile.binary, customAddon);
  } else {
    packageJsonFile.binary.package_name = addonName;
  }
  fs.writeFileSync(
      `${__dirname}/../package.json`, JSON.stringify(packageJsonFile, null, 2));
}

function revertAddonName(orig) {
  packageJsonFile.binary = orig;
  fs.writeFileSync(
      `${__dirname}/../package.json`,
      `${JSON.stringify(packageJsonFile, null, 2)}\n`);
}

function getPlatformLibtensorflowUri() {
  if (platform === 'darwin') {
    if (os.arch() === 'arm64') {
      return `${BASE_HOST}tf-builds/libtensorflow_r2_7_darwin_arm64_cpu.tar.gz`;
    }
    system = `cpu-${PLATFORM_MAPPING[platform]}-${ARCH_MAPPING[os.arch()]}`;
  }

  if (customTFLibUri !== undefined) {
    return customTFLibUri;
  }

  if (platform === 'linux') {
    if (os.arch() === 'arm') {
      return `${BASE_HOST}tf-builds/libtensorflow_r2_5_linux_arm7l.tar.gz`;
    }
    if (os.arch() === 'arm64') {
      return `${BASE_HOST}tf-builds/libtensorflow_r2_7_linux_arm64.tar.gz`;
    }
  }

  if (!ALL_SUPPORTED_COMBINATION.includes(system)) {
    throw new Error(`Unsupported system: ${libType}-${platform}-${os.arch()}`);
  }

  return `${BASE_URI}${system}-${LIBTENSORFLOW_VERSION}.${PLATFORM_EXTENSION}`;
}

async function ensureDir(dirPath) {
  await fs.promises.mkdir(dirPath, {recursive: true});
}

async function cleanDeps() {
  await fs.promises.rm(depsPath, {recursive: true, force: true});
  await ensureDir(depsPath);
}

async function downloadLibtensorflow() {
  await ensureDir(depsPath);
  console.warn('* Downloading libtensorflow');
  console.log(getPlatformLibtensorflowUri());
  await resources.downloadAndUnpackResource(
      getPlatformLibtensorflowUri(), depsPath);

  if (platform !== 'win32' || fs.existsSync(depsLibTensorFlowPath)) {
    return;
  }

  const libtensorflowDll = path.join(depsPath, 'tensorflow.dll');
  if (!fs.existsSync(libtensorflowDll)) {
    throw new Error('Could not find libtensorflow.dll');
  }

  await ensureDir(depsLibPath);
  await fs.promises.rename(libtensorflowDll, depsLibTensorFlowPath);
}

function exec(command, args, env = process.env) {
  return new Promise((resolve, reject) => {
    const child = cp.spawn(command, args, {env, shell: false, stdio: 'inherit'});
    child.once('error', reject);
    child.once('exit', code => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(
          `${command} ${args.join(' ')} exited with code ${code}`));
    });
  });
}

function getNapiBuildVersion() {
  const runtimeVersion = Number(process.versions.napi);
  const versions = packageJsonFile.binary.napi_versions
      .filter(version => version <= runtimeVersion)
      .sort((left, right) => left - right);
  const version = versions.at(-1);
  if (version === undefined) {
    throw new Error(`No supported N-API version for runtime ${runtimeVersion}`);
  }
  return version;
}

async function buildFromSource() {
  const nodeGyp = require.resolve('node-gyp/bin/node-gyp.js');
  const napiBuildVersion = getNapiBuildVersion();
  const binary = packageJsonFile.binary;
  const outputPath = binary.module_path.replace(
      '{napi_build_version}', String(napiBuildVersion));

  console.error('* Building TensorFlow Node.js bindings from source');
  await exec(process.execPath, [
    nodeGyp,
    'rebuild',
    `--module_name=${binary.module_name}`,
    `--module_path=${outputPath}`,
    `--napi_version=${process.versions.napi}`,
    `--napi_build_version=${napiBuildVersion}`,
    '--node_abi_napi=napi'
  ]);
}

async function installPrebuiltAddon() {
  const nodePreGyp = require.resolve('@mapbox/node-pre-gyp/bin/node-pre-gyp');
  const origBinary = structuredClone(packageJsonFile.binary);
  updateAddonName();
  console.error('* Installing TensorFlow Node.js bindings');

  try {
    await exec(process.execPath, [
      nodePreGyp,
      'install',
      customTFLibUri !== undefined && customAddon === undefined ?
          '--build-from-source' : '--fallback-to-build'
    ]);
  } finally {
    revertAddonName(origBinary);
  }
}

async function build() {
  setPackageJsonFile();
  const sourceBuild = packageJsonFile.version === '0.0.0' ||
      process.env.TFJS_NODE_BUILD_FROM_SOURCE === '1' ||
      (customTFLibUri !== undefined && customAddon === undefined);

  if (sourceBuild) {
    await buildFromSource();
  } else {
    await installPrebuiltAddon();
  }

  if (platform === 'win32') {
    await exec(process.execPath, [
      'scripts/deps-stage.js',
      'symlink',
      modulePath
    ]);
  }
}

async function run() {
  if (forceDownload !== 'download' && fs.existsSync(depsLibTensorFlowPath)) {
    await build();
    return;
  }

  await cleanDeps();
  await downloadLibtensorflow();
  await build();
}

run().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
