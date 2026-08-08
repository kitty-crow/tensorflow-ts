/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {memory, tensor1d, tensor2d} from '@tensorflow/tfjs-core';

import {describeMathCPU, expectTensorsClose} from '../utils/test_utils';

import {
  ClassWeight,
  ClassWeightMap,
  standardizeClassWeights,
  standardizeSampleWeights,
  standardizeWeights
} from './training_utils';

describeMathCPU('standardizeWeights', () => {
  it('classWeights with 1D class-index target', async () => {
    const y = tensor1d([0, 1, 2, 1, 0]);
    const classWeight: ClassWeight = {0: 10, 1: 1, 2: 0.1};
    const numTensors0 = memory().numTensors;
    const classSampleWeight = await standardizeWeights(y, null, classWeight);
    expect(memory().numTensors).toEqual(numTensors0 + 1);
    expectTensorsClose(classSampleWeight, tensor1d([10, 1, 0.1, 1, 10]));
    expect(y.isDisposed).toEqual(false);
  });

  it('classWeights with 2D class-index target', async () => {
    const y = tensor2d([[3], [2], [0]]);
    const classWeight: ClassWeight = {0: 10, 1: 1, 2: 0.1, 3: 0.01};
    const numTensors0 = memory().numTensors;
    const classSampleWeight = await standardizeWeights(y, null, classWeight);
    expect(memory().numTensors).toEqual(numTensors0 + 1);
    expectTensorsClose(classSampleWeight, tensor1d([0.01, 0.1, 10]));
    expect(y.isDisposed).toEqual(false);
  });

  it('classWeights with 2D one-hot target', async () => {
    const y = tensor2d([[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]]);
    const classWeight: ClassWeight = {0: 10, 1: 1, 2: 0.1, 3: 0.01};
    const numTensors0 = memory().numTensors;
    const classSampleWeight = await standardizeWeights(y, null, classWeight);
    expect(memory().numTensors).toEqual(numTensors0 + 1);
    expectTensorsClose(classSampleWeight, tensor1d([0.01, 0.1, 10]));
    expect(y.isDisposed).toEqual(false);
  });

  it('rounds half-valued sparse classes ties-to-even like NumPy', async () => {
    const y = tensor1d([0.5, 1.5, 2.5, 3.5, -0.5, -1.5]);
    const classWeight: ClassWeight = {[-2]: 20, 0: 10, 2: 2, 4: 4};
    const classSampleWeight = await standardizeWeights(y, null, classWeight);
    // np.round => [0, 2, 2, 4, -0, -2].
    expectTensorsClose(classSampleWeight, tensor1d([10, 2, 2, 4, 10, 20]));
  });

  it('defaults omitted classWeight entries to one like Keras', async () => {
    const y = tensor1d([0, 1, 2, 3, 2, 1, 0]);
    const classWeight: ClassWeight = {0: 10, 1: 2, 2: 0.1};
    const classSampleWeight = await standardizeWeights(y, null, classWeight);
    expectTensorsClose(
        classSampleWeight, tensor1d([10, 2, 0.1, 1, 0.1, 2, 10]));
  });

  it('defaults omitted one-hot classWeight entries to one like Keras',
     async () => {
       const y = tensor2d([[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]]);
       const classWeight: ClassWeight = {0: 10, 3: 0.01};
       const classSampleWeight = await standardizeWeights(y, null, classWeight);
       expectTensorsClose(classSampleWeight, tensor1d([0.01, 1, 10]));
     });

  it('accepts structural sample weights and returns an owned clone',
     async () => {
       const y = tensor2d([[1, 0, 0], [0, 1, 0]]);
       const sampleWeight = tensor2d([[1, 3, 3], [2, 2, 2]]);
       const standardized = await standardizeWeights(y, sampleWeight, null);
       expectTensorsClose(standardized, sampleWeight);
       expect(standardized.id).not.toEqual(sampleWeight.id);
       expect(sampleWeight.isDisposed).toEqual(false);
     });

  it('rejects sample_weight and class_weight together', async () => {
    const y = tensor2d([[1, 0], [0, 1]]);
    const sampleWeight = tensor1d([1, 2]);
    let caught: Error;
    try {
      await standardizeWeights(y, sampleWeight, {0: 1, 1: 2});
    } catch (error) {
      caught = error;
    }
    expect(caught.message).toMatch(/sample_weight.*class_weight.*same time/);
  });
});

describe('standardizeSampleWeights', () => {
  it('replicates one sample-wise Tensor across multiple outputs', () => {
    const weight = tensor1d([1, 2, 3]);
    const output = standardizeSampleWeights(weight, ['first', 'second']);
    expect(output).toEqual([weight, weight]);
  });

  it('orders output-name dictionaries by model output order', () => {
    const first = tensor2d([[1, 2], [3, 4]]);
    const second = tensor2d([[5, 6], [7, 8]]);
    const output = standardizeSampleWeights(
        {second, first}, ['first', 'second']);
    expect(output).toEqual([first, second]);
  });

  it('rejects one structural Tensor shared across multiple outputs', () => {
    const structural = tensor2d([[1, 2], [3, 4]]);
    expect(() => standardizeSampleWeights(structural, ['first', 'second']))
        .toThrowError(/multiple outputs.*one scalar score per sample/);
  });

  it('requires one structured sample_weight Tensor per output', () => {
    const first = tensor1d([1, 2]);
    expect(() => standardizeSampleWeights([first], ['first', 'second']))
        .toThrowError(/one `sample_weight` array per output/);
  });
});

describe('standardizeClassWeights', () => {
  it('One output, ClassWeight singleton', () => {
    const outputNames = ['output1'];
    const classWeight: ClassWeight = {0: 1, 1: 2};
    const output = standardizeClassWeights(classWeight, outputNames);
    expect(output).toEqual([{0: 1, 1: 2}]);
  });

  it('One output, ClassWeight array', () => {
    const outputNames = ['output1'];
    const classWeight: ClassWeight[] = [{0: 1, 1: 2}];
    const output = standardizeClassWeights(classWeight, outputNames);
    expect(output).toEqual([{0: 1, 1: 2}]);
  });

  it('One output, ClassWeight dict', () => {
    const outputNames = ['output1'];
    const classWeight: ClassWeightMap = {'output1': {0: 1, 1: 2}};
    const output = standardizeClassWeights(classWeight, outputNames);
    expect(output).toEqual([{0: 1, 1: 2}]);
  });

  it('Two outputs, ClassWeight array', () => {
    const outputNames = ['output1', 'output2'];
    const classWeight: ClassWeight[] = [{0: 1, 1: 2}, {0: 10, 1: 20}];
    const output = standardizeClassWeights(classWeight, outputNames);
    expect(output).toEqual([{0: 1, 1: 2}, {0: 10, 1: 20}]);
  });

  it('Two outputs, ClassWeight dict', () => {
    const outputNames = ['output1', 'output2'];
    const classWeight:
        ClassWeightMap = {'output2': {0: 10, 1: 20}, 'output1': {0: 1, 1: 2}};
    const output = standardizeClassWeights(classWeight, outputNames);
    expect(output).toEqual([{0: 1, 1: 2}, {0: 10, 1: 20}]);
  });

  it('Two outputs, ClassWeight singleton leads to Error', () => {
    const outputNames = ['output1', 'output2'];
    const classWeight: ClassWeight = {0: 10, 1: 20};
    expect(() => standardizeClassWeights(classWeight, outputNames))
        .toThrowError(/.*has multiple \(2\) outputs.*/);
  });

  it('Three outputs, ClassWeight array missing element', () => {
    const outputNames = ['output1', 'output2', 'output3'];
    const classWeight: ClassWeight[] = [{0: 1, 1: 2}, {0: 10, 1: 20}];
    expect(() => standardizeClassWeights(classWeight, outputNames))
        .toThrowError(
            /.*classWeight is an array of 2 element.* model has 3 outputs/);
  });

  it('Three outputs, ClassWeight dict missing element is okay', () => {
    const outputNames = ['output1', 'output2', 'output3'];
    const classWeight:
        ClassWeightMap = {'output1': {0: 1, 1: 2}, 'output3': {0: 10, 1: 20}};
    const output = standardizeClassWeights(classWeight, outputNames);
    expect(output).toEqual([{0: 1, 1: 2}, null, {0: 10, 1: 20}]);
  });
});