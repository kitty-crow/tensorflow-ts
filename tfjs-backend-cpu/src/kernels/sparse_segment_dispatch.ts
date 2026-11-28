/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {SparseSegmentSumInputs, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {sparseSegmentReductionImpl} from './SparseSegmentReduction_impl';

/** Validate and dispatch sparse sum and mean with their historical errors. */
export function sparseSegmentReduction(
    args: {inputs: SparseSegmentSumInputs, backend: MathBackendCPU},
    isMean: boolean): TensorInfo {
  const {inputs, backend} = args;
  const {data, indices, segmentIds} = inputs;
  const errorIndent = isMean ? '          ' : '         ';
  if (data.shape.length < 1) {
    throw new Error(
        `Data should be at least 1 dimensional but received scalar`);
  }
  if (indices.shape.length !== 1) {
    throw new Error(`Indices should be a vector but received shape
${errorIndent}${indices.shape}`);
  }
  if (segmentIds.shape.length !== 1) {
    throw new Error(`Segment ids should be a vector but received shape
${errorIndent}${segmentIds.shape}`);
  }
  if (util.requiredNumericEntry(indices.shape, 0) !==
      util.requiredNumericEntry(segmentIds.shape, 0)) {
    throw new Error(`segmentIds and indices should have same size.`);
  }

  const $data = backend.requireValues(data.dataId) as TypedArray;
  const $indices = backend.requireValues(indices.dataId) as TypedArray;
  const $segmentIds = backend.requireValues(segmentIds.dataId) as TypedArray;

  const [outputData, outputDataShape] = sparseSegmentReductionImpl(
      $data, data.shape, data.dtype, $indices, $segmentIds, isMean);
  return backend.makeTensorInfo(outputDataShape, data.dtype, outputData);
}
