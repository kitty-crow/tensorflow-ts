/**
 * @license
 * Copyright 2026
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import * as tfc from '@tensorflow/tfjs-core';

import * as tfl from '../index';
import {TrainingConfig} from '../keras_format/training_config';
import {describeMathCPU} from '../utils/test_utils';

import {standardizeLossWeights} from './training';

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

async function expectWeightedLoss(
    compileArgs: {loss_weights?: number[]|{[key: string]: number},
                  lossWeights?: number[]|{[key: string]: number}},
    expected = 14): Promise<void> {
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

    const trained = await model.trainOnBatch(xs, [firstTarget, secondTarget]);
    expect(Array.isArray(trained)).toBe(true);
    const trainValues = trained as number[];
    expect(trainValues.length).toBe(3);
    expect(trainValues[0]).toBeCloseTo(expected, 5);
    // Keras reports per-output losses before applying loss_weights.
    expect(trainValues[1]).toBeCloseTo(1, 5);
    expect(trainValues[2]).toBeCloseTo(4, 5);

    const evaluated = model.evaluate(xs, [firstTarget, secondTarget]);
    expect(Array.isArray(evaluated)).toBe(true);
    const tensors = evaluated as tfc.Scalar[];
    const values = await Promise.all(tensors.map(async value => {
      return (await value.data())[0];
    }));
    expect(values.length).toBe(3);
    expect(values[0]).toBeCloseTo(expected, 5);
    expect(values[1]).toBeCloseTo(1, 5);
    expect(values[2]).toBeCloseTo(4, 5);
    tfc.dispose(tensors);
  } finally {
    model.dispose();
    tfc.dispose([xs, firstTarget, secondTarget]);
  }
}

describeMathCPU('Keras loss_weights compatibility', () => {
  it('normalizes scalar, list and dictionary forms in output order', () => {
    expect(standardizeLossWeights(undefined, ['a', 'b'])).toEqual([1, 1]);
    expect(standardizeLossWeights(2, ['a'])).toEqual([2]);
    expect(standardizeLossWeights([2, 3], ['a', 'b'])).toEqual([2, 3]);
    expect(standardizeLossWeights({b: 3, a: 2}, ['a', 'b']))
        .toEqual([2, 3]);
  });

  it('rejects Keras-incompatible loss_weights structures', () => {
    expect(() => standardizeLossWeights(2, ['a', 'b']))
        .toThrowError(/loss_weights.*single-output/);
    expect(() => standardizeLossWeights([2], ['a', 'b']))
        .toThrowError(/loss_weights.*2.*1/);
    expect(() => standardizeLossWeights({a: 2}, ['a', 'b']))
        .toThrowError(/loss_weights.*missing.*b/);
    expect(() => standardizeLossWeights({a: 2, b: 3, c: 4}, ['a', 'b']))
        .toThrowError(/loss_weights.*unknown.*c/);
  });

  it('uses exact Keras loss_weights spelling and reduces heterogeneous losses',
     async () => {
       await expectWeightedLoss({loss_weights: [2, 3]});
     });

  it('retains the TensorFlow.js lossWeights alias', async () => {
    await expectWeightedLoss({lossWeights: [2, 3]});
  });

  it('maps dictionary loss_weights by output name, not insertion order',
     async () => {
       await expectWeightedLoss({loss_weights: {second: 3, first: 2}});
     });

  it('rejects specifying both loss_weights and lossWeights', () => {
    const model = makeTwoOutputModel();
    try {
      expect(() => model.compile({
        optimizer: 'sgd',
        loss: ['meanSquaredError', 'meanSquaredError'],
        loss_weights: [2, 3],
        lossWeights: [2, 3]
      })).toThrowError(/only one.*loss_weights.*lossWeights/i);
    } finally {
      model.dispose();
    }
  });

  it('serializes and reloads Keras loss_weights in training_config', () => {
    const model = makeTwoOutputModel();
    const reloaded = makeTwoOutputModel();
    try {
      model.compile({
        optimizer: 'sgd',
        loss: ['meanSquaredError', 'meanSquaredError'],
        loss_weights: {first: 2, second: 3}
      });
      const config = (model as unknown as {
        getTrainingConfig(): TrainingConfig
      }).getTrainingConfig();
      expect(config.loss_weights).toEqual({first: 2, second: 3});

      reloaded.loadTrainingConfig(config);
      expect(reloaded.loss_weights).toEqual({first: 2, second: 3});
      expect(reloaded.lossWeights).toEqual({first: 2, second: 3});
    } finally {
      model.dispose();
      reloaded.dispose();
    }
  });
});
