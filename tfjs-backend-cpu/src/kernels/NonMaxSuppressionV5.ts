/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {kernel_impls, KernelConfig, KernelAttributes, NonMaxSuppressionV5, NonMaxSuppressionV5Attrs, NonMaxSuppressionV5Inputs, TensorInfo} from '@tensorflow/tfjs-core';

const nonMaxSuppressionV5Impl = kernel_impls.nonMaxSuppressionV5Impl;
import {MathBackendCPU} from '../backend_cpu';
import {makeSelectedIndices, readNonMaxSuppressionInputs} from './non_max_suppression_util';

export function nonMaxSuppressionV5(args: {
  inputs: NonMaxSuppressionV5Inputs,
  backend: MathBackendCPU,
  attrs: KernelAttributes<NonMaxSuppressionV5Attrs>
}): [TensorInfo, TensorInfo] {
  const {inputs, backend, attrs} = args;
  const {boxes, scores} = inputs;
  const {maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma} = attrs;

  const [boxesVals, scoresVals] = readNonMaxSuppressionInputs(
      boxes, scores, backend, 'NonMaxSuppressionWithScore');

  const {selectedIndices, selectedScores} = nonMaxSuppressionV5Impl(
      boxesVals, scoresVals, maxOutputSize, iouThreshold,
      scoreThreshold, softNmsSigma);

  return [
    makeSelectedIndices(backend, selectedIndices),
    backend.makeTensorInfo(
        [selectedScores.length], 'float32', new Float32Array(selectedScores))
  ];
}

export const nonMaxSuppressionV5Config: KernelConfig = {
  kernelName: NonMaxSuppressionV5,
  backendName: 'cpu',
  kernelFunc: nonMaxSuppressionV5
};
