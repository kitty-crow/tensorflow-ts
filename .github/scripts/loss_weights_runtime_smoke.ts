import assert from 'node:assert/strict';

import * as tfc from '../../tfjs-core/src/index.ts';
import '../../tfjs-backend-cpu/src/index.ts';
import * as tfl from '../../tfjs-layers/src/index.ts';
import type {TrainingConfig} from '../../tfjs-layers/src/keras_format/training_config.ts';

function close(actual: number, expected: number, label: string): void {
  assert.ok(
      Math.abs(actual - expected) <= 1e-5,
      `${label}: expected ${expected}, received ${actual}`);
}

function makeTwoOutputModel(): tfl.LayersModel {
  const input = tfl.input({shape: [1], name: 'input'});
  const firstDense = tfl.layers.dense({
    units: 3,
    useBias: false,
    kernelInitializer: 'zeros',
    name: 'firstDense'
  }).apply(input) as tfl.SymbolicTensor;
  const secondDense = tfl.layers.dense({
    units: 2,
    useBias: false,
    kernelInitializer: 'zeros',
    name: 'secondDense'
  }).apply(input) as tfl.SymbolicTensor;
  const first = tfl.layers.reshape({
    targetShape: [3, 1],
    name: 'first'
  }).apply(firstDense) as tfl.SymbolicTensor;
  const second = tfl.layers.reshape({
    targetShape: [2, 1],
    name: 'second'
  }).apply(secondDense) as tfl.SymbolicTensor;
  return tfl.model({inputs: input, outputs: [first, second]});
}

function scalarValues(value: tfc.Scalar|tfc.Scalar[]): number[] {
  const values = Array.isArray(value) ? value : [value];
  return values.map(tensor => tensor.dataSync()[0]!);
}

function assertWeightedValues(values: readonly number[], label: string): void {
  assert.equal(values.length, 3, `${label}: expected total plus two output losses`);
  close(values[0]!, 14, `${label} total`);
  close(values[1]!, 1, `${label} first raw loss`);
  close(values[2]!, 4, `${label} second raw loss`);
}

async function exercise(
    compileArgs: {loss_weights?: number[]|Record<string, number>,
                  lossWeights?: number[]|Record<string, number>},
    label: string): Promise<void> {
  const model = makeTwoOutputModel();
  const xs = tfc.zeros([2, 1]);
  const firstTarget = tfc.ones([2, 3, 1]);
  const secondTarget = tfc.fill([2, 2, 1], 2);
  try {
    model.compile({
      optimizer: 'sgd',
      loss: ['meanSquaredError', 'meanSquaredError'],
      ...compileArgs
    });

    const evaluated = model.evaluate(xs, [firstTarget, secondTarget]);
    assertWeightedValues(scalarValues(evaluated), `${label} evaluate`);
    tfc.dispose(evaluated);

    const trained = await model.trainOnBatch(xs, [firstTarget, secondTarget]);
    assert.ok(Array.isArray(trained), `${label}: multi-output trainOnBatch must be an array`);
    assertWeightedValues(trained as number[], `${label} trainOnBatch`);
  } finally {
    model.dispose();
    tfc.dispose([xs, firstTarget, secondTarget]);
  }
}

async function main(): Promise<void> {
  await tfc.setBackend('cpu');
  await tfc.ready();
  assert.equal(tfc.getBackend(), 'cpu');

  await exercise({loss_weights: [2, 3]}, 'Keras spelling');
  await exercise({lossWeights: [2, 3]}, 'TensorFlow.js alias');
  await exercise(
      {loss_weights: {second: 3, first: 2}},
      'dictionary output-name ordering');

  const singleInput = tfl.input({shape: [1], name: 'singleInput'});
  const singleOutput = tfl.layers.dense({
    units: 1,
    useBias: false,
    kernelInitializer: 'zeros',
    name: 'singleOutput'
  }).apply(singleInput) as tfl.SymbolicTensor;
  const single = tfl.model({inputs: singleInput, outputs: singleOutput});
  const sx = tfc.zeros([2, 1]);
  const sy = tfc.ones([2, 1]);
  try {
    single.compile({
      optimizer: 'sgd',
      loss: 'meanSquaredError',
      loss_weights: 2
    });
    const evaluated = scalarValues(single.evaluate(sx, sy));
    assert.equal(evaluated.length, 1);
    close(evaluated[0]!, 2, 'single-output scalar loss_weights');
  } finally {
    single.dispose();
    tfc.dispose([sx, sy]);
  }

  const original = makeTwoOutputModel();
  const restored = makeTwoOutputModel();
  try {
    original.compile({
      optimizer: 'sgd',
      loss: ['meanSquaredError', 'meanSquaredError'],
      loss_weights: {first: 2, second: 3}
    });
    const trainingConfig = (original as unknown as {
      getTrainingConfig(): TrainingConfig
    }).getTrainingConfig();
    assert.deepEqual(trainingConfig.loss_weights, {first: 2, second: 3});

    restored.loadTrainingConfig(trainingConfig);
    assert.deepEqual(restored.loss_weights, {first: 2, second: 3});
    assert.deepEqual(restored.lossWeights, {first: 2, second: 3});
  } finally {
    original.dispose();
    restored.dispose();
  }

  const duplicate = makeTwoOutputModel();
  try {
    assert.throws(() => duplicate.compile({
      optimizer: 'sgd',
      loss: ['meanSquaredError', 'meanSquaredError'],
      loss_weights: [2, 3],
      lossWeights: [2, 3]
    }), /only one.*loss_weights.*lossWeights/i);
  } finally {
    duplicate.dispose();
  }

  console.log(JSON.stringify({
    kerasLossWeights: true,
    exactSnakeCaseApi: true,
    camelCaseAlias: true,
    heterogeneousOutputShapes: true,
    scalarSingleOutput: true,
    dictionaryOutputNames: true,
    trainingConfigRoundTrip: true,
    backend: tfc.getBackend()
  }, null, 2));
}

void main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
