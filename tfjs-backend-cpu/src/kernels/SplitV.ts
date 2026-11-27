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

import {backend_util, SplitVAttrs, SplitVInputs} from '@tensorflow/tfjs-core';
import {KernelConfig, KernelAttributes, SplitV, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {slice} from './Slice';

export function splitV(
    args: {inputs: SplitVInputs, backend: MathBackendCPU, attrs: KernelAttributes<SplitVAttrs>}):
    TensorInfo[] {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {numOrSizeSplits, axis} = attrs;

  const $axis = backend_util.requiredDimension(util.parseAxisParam(axis, x.shape), 0);
  const splitSizes = backend_util.prepareSplitSize(x, numOrSizeSplits, $axis);

  const begin = new Array<number>(x.shape.length).fill(0);
  const size = x.shape.slice();
  let offset = 0;
  return splitSizes.map(s => {
    begin[$axis] = offset;
    const sliceSize = [...size];
    sliceSize[$axis] = s;
    const sliceT =
        slice({inputs: {x}, backend, attrs: {begin, size: sliceSize}});
    offset += s;
    return sliceT;
  });
}

export const splitVConfig: KernelConfig = {
  kernelName: SplitV,
  backendName: 'cpu',
  kernelFunc: splitV
};
