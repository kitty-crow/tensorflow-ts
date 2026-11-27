/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {backend_util, KernelConfig, KernelAttributes, SearchSorted, SearchSortedAttrs, SearchSortedInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {SearchSortedProgram} from '../search_sorted_gpu';

export function searchSorted(args: {
  inputs: SearchSortedInputs,
  backend: MathBackendWebGL,
  attrs: KernelAttributes<SearchSortedAttrs>
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {sortedSequence, values} = inputs;
  const {side} = attrs;

  const program = new SearchSortedProgram(
      backend_util.requiredDimension(sortedSequence.shape, 0), backend_util.requiredDimension(sortedSequence.shape, 1), backend_util.requiredDimension(values.shape, 1), side);
  const customValues = [[backend_util.requiredDimension(sortedSequence.shape, 1)]];
  return backend.runWebGLProgram(
      program, [sortedSequence, values], 'int32', customValues);
}

export const searchSortedConfig: KernelConfig = {
  kernelName: SearchSorted,
  backendName: 'webgl',
  kernelFunc: searchSorted,
};
