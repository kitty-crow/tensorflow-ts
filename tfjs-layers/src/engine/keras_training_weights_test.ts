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
import {describeMathCPU} from '../utils/test_utils';

function makeTwoOutputStructuralModel(): tfl.LayersModel {
  const input = tfl.input({shape: [1], name: 'input'});
  const firstDense = tfl.layers.dense({
    units: 2,
    useBias: false,
    kernelInitializer: 'zeros',
    name: 'firstDense'
  }).apply(input) as tfl.SymbolicTensor;
  const secondDense = tfl.layers.dense({
    units: 6,
    useBias: false,
    kernelInitializer: 'zeros',
    name: 'secondDense'
  }).apply(input) as tfl.SymbolicTensor;
  const first = tfl.layers.reshape({
    targetShape: [2, 1],
    name: 'first'
  }).apply(firstDense) as tfl.SymbolicTensor;
  const second = tfl.layers.reshape({
    targetShape: [2, 3, 1],
    name: 'second'
  }).apply(secondDense) as tfl.SymbolicTensor;
  return tfl.model({inputs: input, outputs: [first, second]});
}

function compileTwoOutputModel(model: tfl.LayersModel): void {
  model.compile({
    optimizer: tfc.train.sgd(0.01),
    loss: ['meanSquaredError', 'meanSquaredError'],
    loss_weights: {first: 2, second: 3}
  });
}

function makeSingleOutputClassifier(): tfl.LayersModel {
  const input = tfl.input({shape: [1], name: 'input'});
  const output = tfl.layers.dense({
    units: 2,
    activation: 'softmax',
    useBias: false,
    kernelInitializer: 'zeros',
    name: 'classes'
  }).apply(input) as tfl.SymbolicTensor;
  return tfl.model({inputs: input, outputs: output});
}

async function expectStructuralWeightedBatch(
    useSnakeCase: boolean, returnDict: boolean): Promise<void> {
  const model = makeTwoOutputStructuralModel();
  const xs = tfc.zeros([2, 1]);
  const firstTarget = tfc.ones([2, 2, 1]);
  const secondTarget = tfc.fill([2, 2, 3, 1], 2);
  const firstWeight = tfc.tensor2d([[1, 3], [1, 3]]);
  const secondWeight = tfc.tensor3d([
    [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
    [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
  ]);

  try {
    compileTwoOutputModel(model);
    const result = useSnakeCase ?
        await model.train_on_batch(
            xs, [firstTarget, secondTarget],
            {second: secondWeight, first: firstWeight}, null, returnDict) :
        await model.trainOnBatch(
            xs, [firstTarget, secondTarget],
            {second: secondWeight, first: firstWeight}, null, returnDict);

    // first raw weighted loss = mean(1 * [1,3]) = 2
    // second raw weighted loss = mean(4 * 0.5) = 2
    // total = 2*2 + 3*2 = 10
    if (returnDict) {
      const values = result as tfl.BatchMetricMap;
      expect(values.loss).toBeCloseTo(10, 5);
      expect(values.first_loss).toBeCloseTo(2, 5);
      expect(values.second_loss).toBeCloseTo(2, 5);
    } else {
      const values = result as number[];
      expect(values.length).toBe(3);
      expect(values[0]).toBeCloseTo(10, 5);
      expect(values[1]).toBeCloseTo(2, 5);
      expect(values[2]).toBeCloseTo(2, 5);
    }

    // Training cleanup must never dispose user-provided weights.
    expect(firstWeight.isDisposed).toBe(false);
    expect(secondWeight.isDisposed).toBe(false);
  } finally {
    model.dispose();
    tfc.dispose([
      xs, firstTarget, secondTarget, firstWeight, secondWeight
    ]);
  }
}

describeMathCPU('Keras sample_weight and class_weight compatibility', () => {
  it('applies nested structural sample weights before loss_weights',
     async () => {
       await expectStructuralWeightedBatch(false, false);
     });

  it('supports canonical train_on_batch and return_dict', async () => {
    await expectStructuralWeightedBatch(true, true);
  });

  it('supports test_on_batch without a gradient update', async () => {
    const model = makeTwoOutputStructuralModel();
    const xs = tfc.zeros([2, 1]);
    const firstTarget = tfc.ones([2, 2, 1]);
    const secondTarget = tfc.fill([2, 2, 3, 1], 2);
    const firstWeight = tfc.tensor2d([[1, 3], [1, 3]]);
    const secondWeight = tfc.fill([2, 2, 3], 0.5);
    try {
      compileTwoOutputModel(model);
      const result = await model.test_on_batch(
          xs, [firstTarget, secondTarget],
          [firstWeight, secondWeight], true) as tfl.BatchMetricMap;
      expect(result.loss).toBeCloseTo(10, 5);
      expect(result.first_loss).toBeCloseTo(2, 5);
      expect(result.second_loss).toBeCloseTo(2, 5);
      expect(firstWeight.isDisposed).toBe(false);
      expect(secondWeight.isDisposed).toBe(false);
    } finally {
      model.dispose();
      tfc.dispose([
        xs, firstTarget, secondTarget, firstWeight, secondWeight
      ]);
    }
  });

  it('replicates one sample-wise weight across multiple outputs', async () => {
    const model = makeTwoOutputStructuralModel();
    const xs = tfc.zeros([2, 1]);
    const firstTarget = tfc.ones([2, 2, 1]);
    const secondTarget = tfc.ones([2, 2, 3, 1]);
    const sampleWeight = tfc.tensor1d([1, 3]);
    try {
      compileTwoOutputModel(model);
      const result = await model.testOnBatch(
          xs, [firstTarget, secondTarget], sampleWeight) as number[];
      // Both raw MSE losses are 1. Shared mean sample weight is 2.
      // total = 2*2 + 2*3 = 10.
      expect(result[0]).toBeCloseTo(10, 5);
      expect(result[1]).toBeCloseTo(2, 5);
      expect(result[2]).toBeCloseTo(2, 5);
    } finally {
      model.dispose();
      tfc.dispose([xs, firstTarget, secondTarget, sampleWeight]);
    }
  });

  it('uses Keras class_weight default 1.0 for omitted classes', async () => {
    const model = makeSingleOutputClassifier();
    const xs = tfc.zeros([2, 1]);
    const ys = tfc.tensor2d([[1, 0], [0, 1]]);
    try {
      model.compile({
        optimizer: tfc.train.sgd(0.01),
        loss: 'categoricalCrossentropy'
      });
      const result = await model.testOnBatch(
          xs, ys,
          // class_weight belongs to train_on_batch, so exercise it there.
          null) as number;
      expect(result).toBeCloseTo(Math.log(2), 5);

      const trained = await model.train_on_batch(
          xs, ys, null, {1: 3}, false) as number;
      // Class 0 is omitted and therefore receives 1.0. Class 1 receives 3.0.
      expect(trained).toBeCloseTo(2 * Math.log(2), 5);
    } finally {
      model.dispose();
      tfc.dispose([xs, ys]);
    }
  });

  it('rejects sample_weight and class_weight together', async () => {
    const model = makeSingleOutputClassifier();
    const xs = tfc.zeros([2, 1]);
    const ys = tfc.tensor2d([[1, 0], [0, 1]]);
    const weight = tfc.ones([2]);
    let caught: Error;
    try {
      model.compile({optimizer: 'sgd', loss: 'categoricalCrossentropy'});
      try {
        await model.train_on_batch(xs, ys, weight, {0: 1, 1: 2});
      } catch (error) {
        caught = error;
      }
      expect(caught.message).toMatch(/sample_weight.*class_weight.*same time/);
    } finally {
      model.dispose();
      tfc.dispose([xs, ys, weight]);
    }
  });

  it('rejects class_weight for multi-output models like Keras 3', async () => {
    const model = makeTwoOutputStructuralModel();
    const xs = tfc.zeros([2, 1]);
    const firstTarget = tfc.ones([2, 2, 1]);
    const secondTarget = tfc.ones([2, 2, 3, 1]);
    let caught: Error;
    try {
      compileTwoOutputModel(model);
      try {
        await model.trainOnBatch(
            xs, [firstTarget, secondTarget], null, {0: 1, 1: 2});
      } catch (error) {
        caught = error;
      }
      expect(caught.message).toMatch(/class_weight.*single output/);
    } finally {
      model.dispose();
      tfc.dispose([xs, firstTarget, secondTarget]);
    }
  });

  it('fit accepts Keras sample_weight spelling and validation weights',
     async () => {
       const model = makeTwoOutputStructuralModel();
       const xs = tfc.zeros([4, 1]);
       const firstTarget = tfc.ones([4, 2, 1]);
       const secondTarget = tfc.fill([4, 2, 3, 1], 2);
       const firstWeight = tfc.ones([4, 2]);
       const secondWeight = tfc.ones([4, 2, 3]);
       const valX = tfc.zeros([2, 1]);
       const valFirst = tfc.ones([2, 2, 1]);
       const valSecond = tfc.fill([2, 2, 3, 1], 2);
       const valFirstWeight = tfc.ones([2, 2]);
       const valSecondWeight = tfc.ones([2, 2, 3]);
       try {
         compileTwoOutputModel(model);
         const history = await model.fit(
             xs, [firstTarget, secondTarget], {
               epochs: 1,
               batchSize: 2,
               verbose: 0,
               shuffle: false,
               sample_weight: [firstWeight, secondWeight],
               validationData: [
                 valX,
                 [valFirst, valSecond],
                 [valFirstWeight, valSecondWeight]
               ]
             });
         expect(history.history.loss.length).toBe(1);
         expect(history.history.val_loss.length).toBe(1);
         expect(firstWeight.isDisposed).toBe(false);
         expect(secondWeight.isDisposed).toBe(false);
         expect(valFirstWeight.isDisposed).toBe(false);
         expect(valSecondWeight.isDisposed).toBe(false);
       } finally {
         model.dispose();
         tfc.dispose([
           xs, firstTarget, secondTarget, firstWeight, secondWeight,
           valX, valFirst, valSecond, valFirstWeight, valSecondWeight
         ]);
       }
     });

  it('fit slices structural sample weights with validationSplit', async () => {
    const model = makeTwoOutputStructuralModel();
    const xs = tfc.zeros([4, 1]);
    const firstTarget = tfc.ones([4, 2, 1]);
    const secondTarget = tfc.ones([4, 2, 3, 1]);
    const firstWeight = tfc.ones([4, 2]);
    const secondWeight = tfc.ones([4, 2, 3]);
    try {
      compileTwoOutputModel(model);
      const history = await model.fit(
          xs, [firstTarget, secondTarget], {
            epochs: 1,
            batchSize: 2,
            verbose: 0,
            shuffle: false,
            validationSplit: 0.5,
            sampleWeight: [firstWeight, secondWeight]
          });
      expect(history.history.loss.length).toBe(1);
      expect(history.history.val_loss.length).toBe(1);
      expect(firstWeight.isDisposed).toBe(false);
      expect(secondWeight.isDisposed).toBe(false);
    } finally {
      model.dispose();
      tfc.dispose([xs, firstTarget, secondTarget, firstWeight, secondWeight]);
    }
  });

  it('keeps predictOnBatch and adds the Keras predict_on_batch alias', () => {
    const model = makeSingleOutputClassifier();
    const xs = tfc.zeros([2, 1]);
    try {
      const camel = model.predictOnBatch(xs) as tfc.Tensor;
      const snake = model.predict_on_batch(xs) as tfc.Tensor;
      expect(camel.shape).toEqual(snake.shape);
      tfc.dispose([camel, snake]);
    } finally {
      model.dispose();
      xs.dispose();
    }
  });
});
