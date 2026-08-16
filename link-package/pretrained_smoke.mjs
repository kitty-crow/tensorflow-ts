import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { loadGraphModel } from '@tensorflow/tfjs-converter';

const modelUrl = process.env.TFJS_SMOKE_MODEL ??
  'https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/model.json';

if (!await tf.setBackend('cpu')) throw new Error('TensorFlow.js CPU backend did not initialise');
await tf.ready();

const before = tf.memory().numTensors;
const model = await loadGraphModel(modelUrl);
const input = tf.zeros([1, 300, 300, 3], 'int32');

try {
  const raw = await model.executeAsync(input);
  const out = Array.isArray(raw) ? raw : [raw];
  if (out.length < 2) throw new Error(`Expected at least two COCO-SSD outputs, got ${out.length}`);

  for (const tensor of out) {
    const data = await tensor.data();
    if (data.length === 0) throw new Error('Pretrained model produced an empty tensor');
    tensor.dispose();
  }

  console.log(`Pretrained GraphModel smoke passed on ${tf.getBackend()} with ${out.length} output tensor(s).`);
} finally {
  input.dispose();
  model.dispose();
}

const after = tf.memory().numTensors;
if (after > before) throw new Error(`Tensor leak after smoke: ${before} -> ${after}`);
