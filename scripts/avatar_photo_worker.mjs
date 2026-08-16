import { createRequire } from 'node:module';
import { createInterface } from 'node:readline';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { prepareRuntime } from './prepare_pretrained_runtime.mjs';

const cocoUrl = 'https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/model.json';
const photoModelDir = process.argv[2];
if (!photoModelDir) throw new Error('Usage: node scripts/avatar_photo_worker.mjs <photo-model-dir>');
const stage = message => console.error(`[Avatar model] ${message}`);
const errorText = error => error instanceof Error ? error.message : String(error);

const runtime = await prepareRuntime();
const require = createRequire(resolve(runtime, 'package.json'));
const tfPath = resolve(runtime, 'node_modules', '@tensorflow', 'tfjs-core', 'dist', 'tf-core.node.js');
const cpuPath = resolve(runtime, 'node_modules', '@tensorflow', 'tfjs-backend-cpu', 'dist', 'tf-backend-cpu.node.js');
const converterPath = resolve(runtime, 'node_modules', '@tensorflow', 'tfjs-converter', 'dist', 'tf-converter.node.js');
const layersPath = resolve(runtime, 'node_modules', '@tensorflow', 'tfjs-layers', 'dist', 'tf-layers.node.js');
stage('loading prepared TensorFlow.js Node bundles…');
const tf = require(tfPath);
require(cpuPath);
const { loadGraphModel } = require(converterPath);
const { loadLayersModel } = require(layersPath);

stage('initialising TensorFlow.js CPU backend…');
if (!await tf.setBackend('cpu')) throw new Error('TensorFlow.js CPU backend did not initialise');
await tf.ready();

const clamp = value => Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
const shape = tensor => Array.isArray(tensor?.shape) ? tensor.shape.join('x') : 'missing';

const layersHandler = async dir => {
  const root = JSON.parse(await readFile(resolve(dir, 'model.json'), 'utf8'));
  const specs = [];
  const shards = [];
  for (const group of root.weightsManifest ?? []) {
    specs.push(...group.weights);
    for (const path of group.paths ?? []) shards.push(await readFile(resolve(dir, path)));
  }
  if (specs.length === 0 || shards.length === 0) throw new Error('Photographic model has no weights');
  const length = shards.reduce((sum, shard) => sum + shard.byteLength, 0);
  if (length < 1) throw new Error('Photographic model weight data is empty');
  const data = new Uint8Array(length);
  let offset = 0;
  for (const shard of shards) {
    data.set(shard, offset);
    offset += shard.byteLength;
  }
  return {
    load: async () => ({
      modelTopology: root.modelTopology,
      format: root.format,
      generatedBy: root.generatedBy,
      convertedBy: root.convertedBy,
      weightSpecs: specs,
      weightData: data.buffer,
      userDefinedMetadata: root.userDefinedMetadata,
    }),
  };
};

stage('loading COCO SSDLite MobileNet V2 person detector…');
const coco = await loadGraphModel(cocoUrl);
stage('loading pinned MobileNet V2 photographic-vs-drawing model…');
const photo = await loadLayersModel(await layersHandler(photoModelDir));

const cocoScoreTensor = outputs => {
  const scores = outputs.find(tensor =>
    tensor?.shape?.length === 3 && tensor.shape[0] === 1 && tensor.shape[2] === 90);
  const boxes = outputs.find(tensor =>
    tensor?.shape?.length === 4 && tensor.shape[0] === 1 && tensor.shape[2] === 1 && tensor.shape[3] === 4);
  if (scores === undefined || boxes === undefined) {
    throw new Error(`Unexpected COCO outputs: ${outputs.map(shape).join(', ') || 'none'}`);
  }
  if (scores.shape[1] !== boxes.shape[1]) {
    throw new Error(`COCO detector count mismatch: scores=${shape(scores)} boxes=${shape(boxes)}`);
  }
  return scores;
};

const personScore = async image => {
  const batch = tf.expandDims(image, 0);
  let outputs = [];
  try {
    const raw = await coco.executeAsync(batch);
    outputs = Array.isArray(raw) ? raw : [raw];
    const scores = cocoScoreTensor(outputs);
    const values = await scores.data();
    const boxes = scores.shape[1];
    const classes = scores.shape[2];
    let person = 0;
    for (let box = 0; box < boxes; box += 1) {
      const value = Number(values[box * classes]);
      if (!Number.isFinite(value)) throw new Error(`COCO person score is not finite at detector ${box}`);
      person = Math.max(person, value);
    }
    return clamp(person);
  } finally {
    tf.dispose(outputs);
    batch.dispose();
  }
};

const photographicScore = async image => {
  const logits = tf.tidy(() => {
    const normalized = tf.div(tf.cast(image, 'float32'), 255);
    const resized = tf.image.resizeBilinear(normalized, [224, 224], true);
    const batched = tf.reshape(resized, [1, 224, 224, 3]);
    return photo.predict(batched);
  });
  try {
    const tensor = Array.isArray(logits) ? logits[0] : logits;
    if (tensor === undefined) throw new Error('Photographic model returned no tensor');
    if (tensor.shape.length !== 2 || tensor.shape[0] !== 1 || tensor.shape[1] !== 5) {
      throw new Error(`Unexpected photographic model output shape ${shape(tensor)}`);
    }
    const values = await tensor.data();
    if (values.length !== 5) throw new Error(`Unexpected photographic model output length ${values.length}`);
    let sum = 0;
    for (let i = 0; i < values.length; i += 1) {
      const value = Number(values[i]);
      if (!Number.isFinite(value) || value < -0.001 || value > 1.001) {
        throw new Error(`Invalid photographic class probability at index ${i}: ${value}`);
      }
      sum += value;
    }
    if (Math.abs(sum - 1) > 0.02) throw new Error(`Photographic class probabilities sum to ${sum}`);
    return clamp(Number(values[2]) + Number(values[3]) + Number(values[4]));
  } finally {
    tf.dispose(logits);
  }
};

const scoreImage = async image => {
  const [person, photographic] = await Promise.all([
    personScore(image),
    photographicScore(image),
  ]);
  return {
    person,
    photographic,
    score: Math.min(person, photographic),
  };
};

const classify = async request => {
  const width = Number(request.width);
  const height = Number(request.height);
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width > 4096 || height > 4096) {
    throw new Error('Invalid RGB dimensions');
  }
  const bytes = Buffer.from(String(request.rgb ?? ''), 'base64');
  const expected = width * height * 3;
  if (bytes.byteLength !== expected) {
    throw new Error(`Invalid RGB byte length ${bytes.byteLength}; expected ${expected}`);
  }

  const image = tf.tensor3d(
    new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength),
    [height, width, 3],
    'int32',
  );
  try {
    return {
      id: String(request.id ?? ''),
      ...await scoreImage(image),
    };
  } finally {
    image.dispose();
  }
};

stage('running end-to-end inference self-test…');
const selfImage = tf.zeros([256, 256, 3], 'int32');
try {
  const self = await scoreImage(selfImage);
  for (const [name, value] of Object.entries(self)) {
    if (!Number.isFinite(value) || value < 0 || value > 1) {
      throw new Error(`Self-test returned invalid ${name}: ${value}`);
    }
  }
} finally {
  selfImage.dispose();
}
stage('ready.');
process.stdout.write(`${JSON.stringify({ ready: true })}\n`);

const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of lines) {
  if (!line.trim()) continue;
  let id = '';
  try {
    const request = JSON.parse(line);
    id = String(request.id ?? '');
    process.stdout.write(`${JSON.stringify(await classify(request))}\n`);
  } catch (error) {
    const message = errorText(error);
    stage(`request ${id || '<unknown>'} failed: ${message}`);
    process.stdout.write(`${JSON.stringify({ id, error: message })}\n`);
  }
}

photo.dispose();
coco.dispose();
