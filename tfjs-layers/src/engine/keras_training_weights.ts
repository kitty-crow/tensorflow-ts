/**
 * @license
 * Copyright 2026
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Keras-compatible sample/class weighting for the TensorFlow.js Layers API.
 *
 * This module intentionally contains no application-specific policy. It fills
 * the public training API gaps between this TypeScript fork and current Keras:
 *
 *   fit(..., sample_weight=..., class_weight=...)
 *   train_on_batch(..., sample_weight=..., class_weight=..., return_dict=...)
 *   test_on_batch(..., sample_weight=..., return_dict=...)
 *
 * Existing TensorFlow.js camelCase methods remain available and route through
 * the same implementation.
 *
 * The compatibility implementation is installed from the public package
 * entrypoint (`../index.ts`). Keeping it isolated avoids changing the model
 * serialization/topology implementation and makes the Keras-specific contract
 * easy to audit against upstream Keras.
 */

import * as tfc from '@tensorflow/tfjs-core';
import {Scalar, Tensor} from '@tensorflow/tfjs-core';

import * as K from '../backend/tfjs_backend';
import {History, standardizeCallbacks} from '../base_callbacks';
import {ValueError} from '../errors';
import {singletonOrArray} from '../utils/generic_utils';

import {execute, FeedDict} from './executor';
import {LayersModel} from './training';
import {
  checkBatchSize,
  disposeNewTensors,
  ModelFitArgs,
  sliceArrays
} from './training_tensors';
import {
  ClassWeight,
  ClassWeightMap,
  computeWeightedLoss,
  SampleWeight,
  standardizeSampleWeights,
  standardizeWeights
} from './training_utils';

export type BatchMetricMap = {[metricName: string]: number};
export type BatchMetricResult = number|number[]|BatchMetricMap;

export type TensorData =
    Tensor|Tensor[]|{[inputName: string]: Tensor};

type FitArgsWithKerasAliases = ModelFitArgs&{
  sample_weight?: SampleWeight;
  class_weight?: ClassWeight;
};

function canonicalWeightConflictError(
    sampleWeight: SampleWeight, _classWeight: unknown): ValueError {
  return new ValueError(
      'Arguments `sample_weight` and `class_weight` cannot be specified at ' +
      'the same time. Received both sample_weight and class_weight.');
}

function resolveAliasedValue<T>(
    args: {[key: string]: unknown}, camelName: string,
    snakeName: string): T {
  const camel = args[camelName] as T;
  const snake = args[snakeName] as T;
  if (camel != null && snake != null) {
    throw new ValueError(
        `Specify only one of \`${snakeName}\` and \`${camelName}\`.`);
  }
  return snake != null ? snake : camel;
}

function normalizeFitArgs(rawArgs: FitArgsWithKerasAliases = {}):
    ModelFitArgs&{sampleWeight?: SampleWeight, classWeight?: unknown} {
  const args = {...rawArgs} as ModelFitArgs&{
    sampleWeight?: SampleWeight,
    classWeight?: unknown,
    sample_weight?: SampleWeight,
    class_weight?: ClassWeight
  };

  args.sampleWeight = resolveAliasedValue<SampleWeight>(
      args as unknown as {[key: string]: unknown},
      'sampleWeight', 'sample_weight');
  args.classWeight = resolveAliasedValue<ClassWeight|ClassWeight[]|ClassWeightMap>(
      args as unknown as {[key: string]: unknown},
      'classWeight', 'class_weight');

  delete args.sample_weight;
  delete args.class_weight;

  if (args.sampleWeight != null && args.classWeight != null) {
    throw canonicalWeightConflictError(args.sampleWeight, args.classWeight);
  }
  return args;
}

function normalizeSingleOutputClassWeight(
    classWeight: ClassWeight|ClassWeight[]|ClassWeightMap,
    outputNames: string[]): ClassWeight {
  if (outputNames.length !== 1) {
    throw new ValueError(
        '`class_weight` is only supported for Models with a single output.');
  }
  if (classWeight == null) {
    return null;
  }

  // Preserve source compatibility with the older TensorFlow.js singleton
  // array/output-map forms while resolving them to Keras' single dictionary.
  if (Array.isArray(classWeight)) {
    if (classWeight.length !== 1) {
      throw new ValueError(
          '`class_weight` is only supported for Models with a single output.');
    }
    return classWeight[0];
  }

  const outputName = outputNames[0];
  const possibleMapValue =
      (classWeight as ClassWeightMap)[outputName] as ClassWeight;
  if (possibleMapValue != null && typeof possibleMapValue === 'object') {
    return possibleMapValue;
  }
  return classWeight as ClassWeight;
}

function sliceWeightArrays(
    weights: Tensor[], start: number, stop: number): Tensor[] {
  if (weights == null) {
    return null;
  }
  return weights.map(weight => {
    if (weight == null) {
      return null;
    }
    return K.sliceAlongFirstAxis(weight, start, stop - start);
  });
}

function disposeWeightArrays(...collections: Tensor[][]): void {
  const disposedIds = new Set<number>();
  for (const collection of collections) {
    if (collection == null) {
      continue;
    }
    for (const tensor of collection) {
      if (tensor == null || disposedIds.has(tensor.id)) {
        continue;
      }
      disposedIds.add(tensor.id);
      if (!tensor.isDisposed) {
        tensor.dispose();
      }
    }
  }
}

function valuesToMetricMap(model: any, values: number[]): BatchMetricMap {
  const names = model.getDedupedMetricsNames() as string[];
  const out: BatchMetricMap = {};
  for (let i = 0; i < values.length; ++i) {
    out[names[i] == null ? `metric_${i}` : names[i]] = values[i];
  }
  return out;
}

async function tensorScalarsToNumbers(tensors: Scalar[]): Promise<number[]> {
  const values: number[] = [];
  for (const tensor of tensors) {
    const data = await tensor.data();
    values.push(data[0]);
  }
  return values;
}

/**
 * Declaration merging adds the Keras spellings and the weighted camelCase
 * overloads to the public LayersModel TypeScript surface.
 */
declare module './training' {
  interface LayersModel {
    // Preserve the historic TFJS return type for the ordinary call. This is
    // important for Sequential, which delegates to LayersModel.trainOnBatch().
    trainOnBatch(x: TensorData, y: TensorData): Promise<number|number[]>;
    trainOnBatch(
        x: TensorData, y: TensorData, sampleWeight?: SampleWeight,
        classWeight?: ClassWeight|ClassWeight[]|ClassWeightMap,
        returnDict?: false): Promise<number|number[]>;
    trainOnBatch(
        x: TensorData, y: TensorData, sampleWeight: SampleWeight,
        classWeight: ClassWeight|ClassWeight[]|ClassWeightMap,
        returnDict: true): Promise<BatchMetricMap>;
    trainOnBatch(
        x: TensorData, y: TensorData, sampleWeight: SampleWeight,
        classWeight: ClassWeight|ClassWeight[]|ClassWeightMap,
        returnDict: boolean): Promise<BatchMetricResult>;

    train_on_batch(
        x: TensorData, y: TensorData, sample_weight?: SampleWeight,
        class_weight?: ClassWeight,
        return_dict?: false): Promise<number|number[]>;
    train_on_batch(
        x: TensorData, y: TensorData, sample_weight: SampleWeight,
        class_weight: ClassWeight, return_dict: true):
        Promise<BatchMetricMap>;
    train_on_batch(
        x: TensorData, y: TensorData, sample_weight: SampleWeight,
        class_weight: ClassWeight, return_dict: boolean):
        Promise<BatchMetricResult>;

    testOnBatch(
        x: TensorData, y: TensorData, sampleWeight?: SampleWeight,
        returnDict?: false): Promise<number|number[]>;
    testOnBatch(
        x: TensorData, y: TensorData, sampleWeight: SampleWeight,
        returnDict: true): Promise<BatchMetricMap>;
    testOnBatch(
        x: TensorData, y: TensorData, sampleWeight: SampleWeight,
        returnDict: boolean): Promise<BatchMetricResult>;

    test_on_batch(
        x: TensorData, y: TensorData, sample_weight?: SampleWeight,
        return_dict?: false): Promise<number|number[]>;
    test_on_batch(
        x: TensorData, y: TensorData, sample_weight: SampleWeight,
        return_dict: true): Promise<BatchMetricMap>;
    test_on_batch(
        x: TensorData, y: TensorData, sample_weight: SampleWeight,
        return_dict: boolean): Promise<BatchMetricResult>;

    predict_on_batch(x: Tensor|Tensor[]): Tensor|Tensor[];
  }
}

const prototype = LayersModel.prototype as any;
const PATCH_FLAG = '__kerasSampleClassWeightCompatibilityInstalled';

if (!prototype[PATCH_FLAG]) {
  prototype[PATCH_FLAG] = true;

  /**
   * Standardize TensorFlow.js/Keras weight structures to one optional Tensor
   * per output. Returned weight tensors are owned by the training call, never
   * by the caller, so normal training cleanup is safe.
   */
  prototype.standardizeUserData = async function(
      x: TensorData, y: TensorData, sampleWeight?: SampleWeight,
      classWeight?: ClassWeight|ClassWeight[]|ClassWeightMap,
      checkBatchAxis = true,
      batchSize?: number): Promise<[Tensor[], Tensor[], Tensor[]]> {
    const [standardXs, standardYs] =
        this.standardizeUserDataXY(x, y, checkBatchAxis, batchSize) as
        [Tensor[], Tensor[]];

    if (sampleWeight != null && classWeight != null) {
      throw canonicalWeightConflictError(sampleWeight, classWeight);
    }

    let standardSampleWeights: Tensor[] =
        this.outputNames.map((_name: string): Tensor => null);

    if (sampleWeight != null) {
      const orderedWeights =
          standardizeSampleWeights(sampleWeight, this.outputNames);
      standardSampleWeights = [];
      for (let i = 0; i < orderedWeights.length; ++i) {
        standardSampleWeights.push(await standardizeWeights(
            standardYs[i], orderedWeights[i], null));
      }
    } else if (classWeight != null) {
      const canonicalClassWeight =
          normalizeSingleOutputClassWeight(classWeight, this.outputNames);
      standardSampleWeights = [await standardizeWeights(
          standardYs[0], null, canonicalClassWeight)];
    }

    return [standardXs, standardYs, standardSampleWeights];
  };

  /**
   * Test/evaluation function matching makeTrainFunction's sample-weight and
   * loss_weights order: unreduced loss -> sample_weight -> reduction ->
   * whole-output loss_weight -> sum. Ordinary metrics remain unweighted, as
   * in Keras unless `weighted_metrics` is explicitly configured.
   */
  prototype.makeTestFunction = function(): void {
    const model = this;
    model.testFunction = (data: Tensor[]) => {
      return tfc.tidy(() => {
        const valOutputs: Scalar[] = [];
        const lossValues: Scalar[] = [];
        const inputs = data.slice(0, model.inputs.length);
        const targets = data.slice(
            model.inputs.length, model.inputs.length + model.outputs.length);
        const sampleWeights = data.slice(
            model.inputs.length + model.outputs.length,
            model.inputs.length + model.outputs.length * 2);

        const feeds = [];
        for (let i = 0; i < model.inputs.length; ++i) {
          feeds.push({key: model.inputs[i], value: inputs[i]});
        }
        const feedDict = new FeedDict(feeds);
        const outputs = execute(model.outputs, feedDict) as Tensor[];

        let totalLoss: Scalar = null;
        for (let i = 0; i < model.lossFunctions.length; ++i) {
          const lossFunction = model.lossFunctions[i];
          let elementLoss = lossFunction(targets[i], outputs[i]);
          if (sampleWeights[i] != null) {
            elementLoss = computeWeightedLoss(
                elementLoss, sampleWeights[i]);
          }
          const meanLoss = tfc.mean(elementLoss) as Scalar;
          lossValues.push(meanLoss);

          const lossWeight = model.lossWeightValues[i];
          const weightedLoss = lossWeight === 1 ? meanLoss :
              tfc.mul(meanLoss, lossWeight) as Scalar;
          totalLoss = totalLoss == null ? weightedLoss :
              tfc.add(totalLoss, weightedLoss) as Scalar;
        }
        if (totalLoss == null) {
          totalLoss = tfc.scalar(0);
        }
        valOutputs.push(totalLoss);

        for (let i = 0; i < model.metricsTensors.length; ++i) {
          let meanMetric: Scalar;
          if (model.outputs.length > 1 && i < model.outputs.length) {
            meanMetric = lossValues[i];
          } else {
            const metric = model.metricsTensors[i][0];
            const outputIndex = model.metricsTensors[i][1];
            meanMetric = tfc.mean(
                metric(targets[outputIndex], outputs[outputIndex])) as Scalar;
          }
          valOutputs.push(meanMetric);
        }
        return valOutputs;
      });
    };
  };

  /**
   * Full tensor-based fit path with Keras sample-weight semantics, including
   * validation_data=(x, y, sample_weight) and validation_split slicing of the
   * corresponding sample weights.
   */
  prototype.fit = async function(
      x: TensorData, y: TensorData,
      rawArgs: FitArgsWithKerasAliases = {}): Promise<History> {
    const args = normalizeFitArgs(rawArgs);
    if (this.isTraining) {
      throw new Error(
          'Cannot start training because another fit() call is ongoing.');
    }

    this.isTraining = true;
    let inputs: Tensor[];
    let targets: Tensor[];
    let sampleWeights: Tensor[];
    let originalInputs: Tensor[];
    let originalTargets: Tensor[];
    let originalSampleWeights: Tensor[];
    let inputValX: TensorData;
    let inputValY: TensorData;
    let valX: Tensor[];
    let valY: Tensor[];
    let valSampleWeights: Tensor[];

    try {
      const batchSize = args.batchSize == null ? 32 : args.batchSize;
      checkBatchSize(batchSize);

      const standardizedOuts = await this.standardizeUserData(
          x, y, args.sampleWeight, args.classWeight, false, batchSize) as
          [Tensor[], Tensor[], Tensor[]];
      inputs = standardizedOuts[0];
      targets = standardizedOuts[1];
      sampleWeights = standardizedOuts[2];

      let doValidation = false;
      let valIns: Tensor[] = [];

      if (args.validationData != null && args.validationData.length > 0) {
        doValidation = true;
        if (args.validationData.length !== 2 &&
            args.validationData.length !== 3) {
          throw new ValueError(
              'When passing validation data, it must contain 2 ' +
              '(valX, valY) or 3 (valX, valY, valSampleWeight) items.');
        }

        inputValX = args.validationData[0] as TensorData;
        inputValY = args.validationData[1] as TensorData;
        const inputValSampleWeight = args.validationData.length === 3 ?
            args.validationData[2] as SampleWeight : null;

        const valStandardized = await this.standardizeUserData(
            inputValX, inputValY, inputValSampleWeight, null, true,
            batchSize) as [Tensor[], Tensor[], Tensor[]];
        valX = valStandardized[0];
        valY = valStandardized[1];
        valSampleWeights = valStandardized[2];
        valIns = valX.concat(valY).concat(valSampleWeights);
      } else if (
          args.validationSplit != null && args.validationSplit > 0 &&
          args.validationSplit < 1) {
        doValidation = true;
        const splitAt = Math.floor(
            inputs[0].shape[0] * (1 - args.validationSplit));
        const originalBatchSize = inputs[0].shape[0];

        originalInputs = inputs;
        originalTargets = targets;
        originalSampleWeights = sampleWeights;

        valX = sliceArrays(
            originalInputs, splitAt, originalBatchSize) as Tensor[];
        inputs = sliceArrays(originalInputs, 0, splitAt) as Tensor[];
        valY = sliceArrays(
            originalTargets, splitAt, originalBatchSize) as Tensor[];
        targets = sliceArrays(originalTargets, 0, splitAt) as Tensor[];
        valSampleWeights = sliceWeightArrays(
            originalSampleWeights, splitAt, originalBatchSize);
        sampleWeights = sliceWeightArrays(originalSampleWeights, 0, splitAt);
        valIns = valX.concat(valY).concat(valSampleWeights);
      } else if (args.validationSteps != null) {
        doValidation = true;
      }

      const ins = inputs.concat(targets).concat(sampleWeights);
      this.checkTrainableWeightsConsistency();

      const trainFunction = this.makeTrainFunction();
      const outLabels = this.getDedupedMetricsNames();

      let valFunction: (data: Tensor[]) => Scalar[] = null;
      let callbackMetrics: string[];
      if (doValidation && valIns.length > 0) {
        this.makeTestFunction();
        valFunction = this.testFunction;
        callbackMetrics =
            outLabels.slice().concat(outLabels.map((name: string) =>
              'val_' + name));
      } else {
        callbackMetrics = outLabels.slice();
      }

      const callbacks = standardizeCallbacks(args.callbacks, args.yieldEvery);
      return await this.fitLoop(
          trainFunction, ins, outLabels, batchSize, args.epochs,
          args.verbose, callbacks, valFunction, valIns, args.shuffle,
          callbackMetrics, args.initialEpoch, null, null);
    } finally {
      this.isTraining = false;
      disposeNewTensors(inputs, x);
      disposeNewTensors(targets, y);
      disposeNewTensors(originalInputs, x);
      disposeNewTensors(originalTargets, y);
      disposeNewTensors(valX, inputValX);
      disposeNewTensors(valY, inputValY);
      disposeWeightArrays(
          sampleWeights, originalSampleWeights, valSampleWeights);
    }
  };

  prototype.trainOnBatch = async function(
      x: TensorData, y: TensorData, sampleWeight?: SampleWeight,
      classWeight?: ClassWeight|ClassWeight[]|ClassWeightMap,
      returnDict = false): Promise<BatchMetricResult> {
    if (sampleWeight != null && classWeight != null) {
      throw canonicalWeightConflictError(sampleWeight, classWeight);
    }

    let standardized: [Tensor[], Tensor[], Tensor[]];
    let lossTensors: Scalar[] = [];
    try {
      standardized = await this.standardizeUserData(
          x, y, sampleWeight, classWeight) as
          [Tensor[], Tensor[], Tensor[]];
      const trainFunction = this.makeTrainFunction();
      lossTensors = trainFunction(
          standardized[0].concat(standardized[1]).concat(standardized[2]));
      const values = await tensorScalarsToNumbers(lossTensors);
      return returnDict ? valuesToMetricMap(this, values) :
          singletonOrArray(values);
    } finally {
      tfc.dispose(lossTensors);
      if (standardized != null) {
        disposeNewTensors(standardized[0], x);
        disposeNewTensors(standardized[1], y);
        disposeWeightArrays(standardized[2]);
      }
    }
  };

  prototype.train_on_batch = async function(
      x: TensorData, y: TensorData, sample_weight?: SampleWeight,
      class_weight?: ClassWeight,
      return_dict = false): Promise<BatchMetricResult> {
    return this.trainOnBatch(
        x, y, sample_weight, class_weight, return_dict);
  };

  prototype.testOnBatch = async function(
      x: TensorData, y: TensorData, sampleWeight?: SampleWeight,
      returnDict = false): Promise<BatchMetricResult> {
    let standardized: [Tensor[], Tensor[], Tensor[]];
    let resultTensors: Scalar[] = [];
    try {
      standardized = await this.standardizeUserData(
          x, y, sampleWeight, null) as [Tensor[], Tensor[], Tensor[]];
      this.makeTestFunction();
      resultTensors = this.testFunction(
          standardized[0].concat(standardized[1]).concat(standardized[2]));
      const values = await tensorScalarsToNumbers(resultTensors);
      return returnDict ? valuesToMetricMap(this, values) :
          singletonOrArray(values);
    } finally {
      tfc.dispose(resultTensors);
      if (standardized != null) {
        disposeNewTensors(standardized[0], x);
        disposeNewTensors(standardized[1], y);
        disposeWeightArrays(standardized[2]);
      }
    }
  };

  prototype.test_on_batch = async function(
      x: TensorData, y: TensorData, sample_weight?: SampleWeight,
      return_dict = false): Promise<BatchMetricResult> {
    return this.testOnBatch(x, y, sample_weight, return_dict);
  };

  prototype.predict_on_batch = function(
      x: Tensor|Tensor[]): Tensor|Tensor[] {
    return this.predictOnBatch(x);
  };
}
