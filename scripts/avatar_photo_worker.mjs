import { createRequire } from 'node:module';
import { createInterface } from 'node:readline';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { prepareRuntime } from './prepare_pretrained_runtime.mjs';

const cocoUrl = 'https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/model.json';
const photoModelDir = process.argv[2];
if (!photoModelDir) throw new Error('Usage: node scripts/avatar_photo_worker.mjs <photo-model-dir>');

const runtime = await prepareRuntime();
const require = createRequire(resolve(runtime, 'package.json'));
const tfPath = resolve(runtime, 'node_modules', '@tensorflow', 'tfjs-core', 'dist', 'tf-core.node.js');
const cpuPath = resolve(runtime, 'node_modules', '@tensorflow', 'tfjs-backend-cpu', 'dist', 'tf-backend-cpu.node.js');
const converterPath = resolve(runtime, 'node_modules', '@tensorflow', 'tfjs-converter', 'dist', 'tf-converter.node.js');
const layersPath = resolve(runtime, 'node_modules', '@tensorflow', 'tfjs-layers', 'dist', 'tf-layers.node.js');
const tf = require(tfPath);
require(cpuPath);
const { loadGraphModel } = require(converterPath);
const { loadLayersModel } = require(layersPath);

if (!await tf.setBackend('cpu')) throw new Error('TensorFlow.js CPU backend did not initialise');
await tf.ready();

const clamp = value => Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));

const layersHandler = async dir => {
  const root = JSON.parse(await readFile(resolve(dir, 'model.json'), 'utf8'));
  const specs = [];
  const shards = [];
  for (const group of root.weightsManifest ?? []) {
    specs.push(...group.weights);
    for (const path of group.paths ?? []) shards.push(await readFile(resolve(dir, path)));
  }
  const length = shards.reduce((sum, shard) => sum + shard.byteLength, 0);
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

const coco = await loadGraphModel(cocoUrl);
const photo = await loadLayersModel(await layersHandler(photoModelDir));

const warmCoco = tf.zeros([1, 300, 300, 3], 'int32');
const warmOut = await coco.executeAsync(warmCoco);
tf.dispose(warmOut);
warmCoco.dispose();
const warmPhoto = tf.zeros([1, 224, 224, 3]);
const warmPhotoOut = photo.predict(warmPhoto);
tf.dispose(warmPhotoOut);
warmPhoto.dispose();

const classify = async request => {
  const width = Number(request.width);
  const height = Number(request.height);
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
    throw new Error('Invalid RGB dimensions');
  }
  const bytes = Buffer.from(String(request.rgb ?? ''), 'base64');
  if (bytes.byteLength !== width * height * 3) {
    throw new Error(`Invalid RGB byte length ${bytes.byteLength}; expected ${width * height * 3}`);
  }

  const image = tf.tensor3d(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength), [height, width, 3], 'int32');
  let person = 0;
  let photographic = 0;
  try {
    const batch = image.expandDims(0);
    try {
      const raw = await coco.executeAsync(batch);
      const outputs = Array.isArray(raw) ? raw : [raw];
      const scores = outputs[0];
      if (scores === undefined || scores.shape.length !== 3 || scores.shape[0] !== 1 || scores.shape[2] !== 90) {
        throw new Error(`Unexpected COCO score tensor shape ${scores?.shape.join('x') ?? 'missing'}`);
      }
      const values = await scores.data();
      const boxes = scores.shape[1] ?? 0;
      const classes = scores.shape[2] ?? 0;
      for (let box = 0; box < boxes; box += 1) {
        person = Math.max(person, Number(values[box * classes] ?? 0));
      }
      tf.dispose(outputs);
    } finally {
      batch.dispose();
    }

    const logits = tf.tidy(() => {
      const resized = tf.image.resizeBilinear(image.toFloat().div(255), [224, 224], true);
      return photo.predict(resized.reshape([1, 224, 224, 3]));
    });
    try {
      const tensor = Array.isArray(logits) ? logits[0] : logits;
      if (tensor === undefined) throw new Error('Photographic model returned no tensor');
      const values = await tensor.data();
      if (values.length !== 5) throw new Error(`Unexpected photographic model output length ${values.length}`);
      photographic = clamp(Number(values[2] ?? 0) + Number(values[3] ?? 0) + Number(values[4] ?? 0));
    } finally {
      tf.dispose(logits);
    }
  } finally {
    image.dispose();
  }

  person = clamp(person);
  return {
    id: String(request.id ?? ''),
    person,
    photographic,
    score: Math.min(person, photographic),
  };
};

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
    process.stdout.write(`${JSON.stringify({ id, error: error instanceof Error ? error.message : String(error) })}\n`);
  }
}

photo.dispose();
coco.dispose();
