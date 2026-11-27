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

import {EluGrad, EluGradInputs, KernelConfig, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex, requiredCpuEntry} from '../cpu_util';

export function eluGrad(args: {inputs: EluGradInputs, backend: MathBackendCPU}):
    TensorInfo {
  const {inputs, backend} = args;
  const {dy, y} = inputs;

  assertNotComplex([dy, y], 'eluGrad');

  const resultValues = new Float32Array(util.sizeFromShape(y.shape));
  const values = backend.requireValues(y.dataId) as TypedArray;
  const dyValues = backend.requireValues(dy.dataId) as TypedArray;
  for (let i = 0; i < values.length; ++i) {
    const v = requiredCpuEntry(values, i);
    const gradient = requiredCpuEntry(dyValues, i);
    resultValues[i] = v >= 0 ? gradient : gradient * (v + 1);
  }

  return backend.makeTensorInfo(y.shape, 'float32', resultValues);
}

export const eluGradConfig: KernelConfig = {
  kernelName: EluGrad,
  backendName: 'cpu',
  kernelFunc: eluGrad
};
