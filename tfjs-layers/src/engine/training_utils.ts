/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {
  argMax,
  clone,
  dispose,
  mul,
  reshape,
  Tensor,
  tensor1d,
  tidy
} from '@tensorflow/tfjs-core';

/**
 * For multi-class classification problems, this object is designed to store a
 * mapping from class index to the "weight" of the class, where higher weighted
 * classes have larger impact on loss, accuracy, and other metrics.
 *
 * This is useful for cases in which you want the model to "pay more attention"
 * to examples from an under-represented class, e.g., in unbalanced datasets.
 */
export type ClassWeight = {
  [classIndex: number]: number
};

/**
 * Legacy TensorFlow.js class weighting for a model with multiple outputs.
 *
 * Keras 3 itself only accepts `class_weight` for single-output models. The
 * type is retained because older TensorFlow.js callers may import it directly;
 * the public fit/batch compatibility layer enforces the Keras 3 restriction.
 */
export type ClassWeightMap = {
  [outputName: string]: ClassWeight
};

/** A dictionary of sample-weight tensors keyed by model output name. */
export type SampleWeightMap = {
  [outputName: string]: Tensor
};

/**
 * Keras-compatible sample-weight structure.
 *
 * A single-output model accepts one Tensor. A multi-output model accepts one
 * Tensor per output as either an Array or an output-name dictionary. A single
 * rank-1 (or `[batch, 1]`) Tensor may also be shared by all outputs.
 */
export type SampleWeight = Tensor|Tensor[]|SampleWeightMap;

function standardizeSampleOrClassWeights(
    xWeight: ClassWeight|ClassWeight[]|ClassWeightMap, outputNames: string[],
    weightType: 'sampleWeight'|'classWeight'): ClassWeight[] {
  const numOutputs = outputNames.length;
  if (xWeight == null || (Array.isArray(xWeight) && xWeight.length === 0)) {
    return outputNames.map(_name => null);
  }
  if (numOutputs === 1) {
    if (Array.isArray(xWeight) && xWeight.length === 1) {
      return xWeight;
    } else if (typeof xWeight === 'object' && outputNames[0] in xWeight &&
               typeof (xWeight as ClassWeightMap)[outputNames[0]] ===
                   'object') {
      return [(xWeight as ClassWeightMap)[outputNames[0]]];
    } else {
      return [xWeight as ClassWeight];
    }
  }
  if (Array.isArray(xWeight)) {
    if (xWeight.length !== numOutputs) {
      throw new Error(
          `Provided ${weightType} is an array of ${xWeight.length} ` +
          `element(s), but the model has ${numOutputs} outputs. ` +
          `Make sure a set of weights is provided for each model output.`);
    }
    return xWeight;
  } else if (
      typeof xWeight === 'object' && Object.keys(xWeight).length > 0 &&
      typeof (xWeight as ClassWeightMap)[Object.keys(xWeight)[0]] ===
          'object') {
    const output: ClassWeight[] = [];
    outputNames.forEach(outputName => {
      if (outputName in xWeight) {
        output.push((xWeight as ClassWeightMap)[outputName]);
      } else {
        output.push(null);
      }
    });
    return output;
  } else {
    throw new Error(
        `The model has multiple (${numOutputs}) outputs, ` +
        `so ${weightType} must be either an array with ` +
        `${numOutputs} elements or an object with ${outputNames} keys. ` +
        `Provided ${weightType} not understood: ${JSON.stringify(xWeight)}`);
  }
}

/**
 * Standardize legacy TensorFlow.js class weighting objects.
 *
 * Public Keras-compatible training APIs restrict `class_weight` to a
 * single-output model, matching current Keras. This helper remains exported
 * for backwards compatibility with existing TensorFlow.js code.
 */
export function standardizeClassWeights(
    classWeight: ClassWeight|ClassWeight[]|ClassWeightMap,
    outputNames: string[]): ClassWeight[] {
  return standardizeSampleOrClassWeights(
      classWeight, outputNames, 'classWeight');
}

/**
 * Normalize a Keras sample-weight structure to model output order.
 *
 * For a multi-output model, a singleton rank-1 Tensor (or `[batch, 1]`
 * Tensor) is sample-wise and therefore shared across all outputs. Structured
 * weights must contain exactly one Tensor per output.
 */
export function standardizeSampleWeights(
    sampleWeight: SampleWeight, outputNames: string[]): Tensor[] {
  const numOutputs = outputNames.length;
  if (sampleWeight == null) {
    return outputNames.map(_name => null);
  }

  if (numOutputs === 1) {
    if (sampleWeight instanceof Tensor) {
      return [sampleWeight];
    }
    if (Array.isArray(sampleWeight)) {
      if (sampleWeight.length !== 1) {
        throw new Error(
            'You should provide one `sample_weight` array per output in `y`. ' +
            `The model has 1 output but received ${sampleWeight.length} ` +
            'sample-weight arrays.');
      }
      return [sampleWeight[0]];
    }

    const keys = Object.keys(sampleWeight);
    const expectedName = outputNames[0];
    const unknown = keys.filter(key => key !== expectedName);
    if (unknown.length > 0 || !(expectedName in sampleWeight)) {
      throw new Error(
          'You should provide one `sample_weight` array per output in `y`. ' +
          `Expected output key "${expectedName}", received: ${keys}.`);
    }
    return [(sampleWeight as SampleWeightMap)[expectedName]];
  }

  if (sampleWeight instanceof Tensor) {
    const isSamplewise = sampleWeight.rank === 1 ||
        (sampleWeight.rank === 2 && sampleWeight.shape[1] === 1);
    if (!isSamplewise) {
      throw new Error(
          'For a model with multiple outputs, when providing a single ' +
          '`sample_weight` array, it should only have one scalar score per ' +
          'sample (i.e. shape `(num_samples,)`). If you want to use ' +
          'non-scalar sample weights, pass a `sample_weight` argument with ' +
          'one array per model output.');
    }
    return outputNames.map(_name => sampleWeight);
  }

  if (Array.isArray(sampleWeight)) {
    if (sampleWeight.length !== numOutputs) {
      throw new Error(
          'You should provide one `sample_weight` array per output in `y`. ' +
          `The model has ${numOutputs} outputs but received ` +
          `${sampleWeight.length} sample-weight arrays.`);
    }
    return sampleWeight.slice();
  }

  const keys = Object.keys(sampleWeight);
  const missing = outputNames.filter(name => !(name in sampleWeight));
  const unknown = keys.filter(name => outputNames.indexOf(name) === -1);
  if (missing.length > 0 || unknown.length > 0) {
    throw new Error(
        'You should provide one `sample_weight` array per output in `y`. ' +
        `Expected keys: ${outputNames}. Missing: ${missing}. Unknown: ` +
        `${unknown}.`);
  }
  return outputNames.map(name => (sampleWeight as SampleWeightMap)[name]);
}

/**
 * Standardize by-sample and/or by-class weights for one model output.
 *
 * Keras treats `sample_weight` and `class_weight` as mutually exclusive.
 * Missing entries in `class_weight` receive the canonical default weight 1.0.
 *
 * @param y The target tensor that the weight applies to.
 * @param sampleWeight Explicit sample/structural weights. The first dimension
 *     must match the batch dimension of `y`; remaining dimensions are kept so
 *     they can broadcast against the unreduced loss.
 * @param classWeight Mapping from class index to scalar weight. This is only
 *     valid when there is exactly one class label per batch sample.
 * @param sampleWeightMode Retained for TensorFlow.js API compatibility. Modern
 *     Keras no longer requires a temporal mode; structural dimensions are
 *     represented directly by the sample-weight tensor.
 * @return A training-owned weight tensor, or null if no weighting was
 *     requested.
 */
export async function standardizeWeights(
    y: Tensor, sampleWeight?: Tensor, classWeight?: ClassWeight,
    sampleWeightMode?: 'temporal'): Promise<Tensor> {
  if (sampleWeight != null && classWeight != null) {
    throw new Error(
        'Arguments `sample_weight` and `class_weight` cannot be specified at ' +
        'the same time.');
  }

  if (sampleWeight != null) {
    if (sampleWeight.rank < 1) {
      throw new Error('`sample_weight` must have rank 1 or greater.');
    }
    if (sampleWeight.shape[0] !== y.shape[0]) {
      throw new Error(
          '`sample_weight` must contain one first-axis entry per target ' +
          `sample. Received target batch ${y.shape[0]} and sample_weight ` +
          `batch ${sampleWeight.shape[0]}.`);
    }
    // Clone so training cleanup never disposes a caller-owned Tensor. Keeping
    // all structural dimensions is essential for losses such as
    // [batch, groups] and [batch, groups, satellites].
    return clone(sampleWeight);
  }

  if (classWeight != null) {
    // Keras class_weight_to_sample_weights() produces exactly one scalar
    // weight per batch item. Derive one class index per sample and reject
    // structural targets for which no single class per sample exists.
    const yClasses = tidy(() => {
      if (y.rank === 1) {
        return clone(y);
      }

      const lastAxis = y.rank - 1;
      const lastDim = y.shape[lastAxis];
      if (lastDim == null || lastDim < 1) {
        throw new Error(
            'Encountered an unexpected empty last dimension while handling ' +
            '`class_weight`.');
      }

      if (lastDim !== 1) {
        return argMax(y, lastAxis);
      }
      return reshape(y, y.shape.slice(0, -1));
    });

    try {
      if (yClasses.rank !== 1) {
        throw new Error(
            '`class_weight` is only supported when each sample has one class ' +
            'label. Use `sample_weight` for structural or temporal targets.');
      }

      const yClassIndices = Array.from(await yClasses.data());
      const classSampleWeight: number[] = [];
      yClassIndices.forEach(rawClassIndex => {
        // Keras rounds class-index targets before conversion to int32.
        const classIndex = Math.round(Number(rawClassIndex));
        const configuredWeight = classWeight[classIndex];
        classSampleWeight.push(
            configuredWeight == null ? 1.0 : configuredWeight);
      });
      return tensor1d(classSampleWeight, 'float32');
    } finally {
      dispose(yClasses);
    }
  }

  // `sampleWeightMode` is intentionally not used. It remains in the signature
  // solely for source compatibility with older TensorFlow.js callers.
  void sampleWeightMode;
  return null;
}

/**
 * Apply sample/structural weights to the unreduced loss tensor.
 *
 * Keras sample-wise weights are conceptually expanded across trailing loss
 * dimensions. TensorFlow broadcasting is then used for temporal/structural
 * weight tensors that already retain those dimensions.
 */
export function computeWeightedLoss(losses: Tensor, sampleWeights: Tensor) {
  return tidy(() => {
    let alignedWeights = sampleWeights;
    if (sampleWeights.rank < losses.rank) {
      alignedWeights = reshape(
          sampleWeights,
          sampleWeights.shape.concat(
              new Array(losses.rank - sampleWeights.rank).fill(1)));
    }
    return mul(losses, alignedWeights);
  });
}
