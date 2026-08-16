import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { loadLayersModel } from '@tensorflow/tfjs-layers';

const modelUrl = process.env.TFJS_LAYERS_SMOKE_MODEL ??
  'https://raw.githubusercontent.com/kitty-crow/nsfwjs/5940117af9afc86a0994d959f50603aeeeac1f2d/models/mobilenet_v2/model.json';

if (!await tf.setBackend('cpu')) throw new Error('TensorFlow.js CPU backend did not initialise');
await tf.ready();

const before = tf.memory().numTensors;
const model = await loadLayersModel(modelUrl);
const input = tf.zeros([1, 224, 224, 3]);

try {
  const raw = model.predict(input);
  const out = Array.isArray(raw) ? raw : [raw];
  if (out.length !== 1) throw new Error(`Expected one NSFWJS output tensor, got ${out.length}`);
  const tensor = out[0];
  if (tensor === undefined) throw new Error('NSFWJS model returned no tensor');
  const data = await tensor.data();
  if (data.length !== 5) throw new Error(`Expected five NSFWJS classes, got ${data.length}`);
  tensor.dispose();
  console.log(`Pretrained LayersModel smoke passed on ${tf.getBackend()} with ${data.length} classes.`);
} finally {
  input.dispose();
  model.dispose();
}

const after = tf.memory().numTensors;
if (after > before) throw new Error(`Tensor leak after LayersModel smoke: ${before} -> ${after}`);
