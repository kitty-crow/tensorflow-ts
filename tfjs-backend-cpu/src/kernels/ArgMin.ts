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

import {ArgMin, ArgMinAttrs, ArgMinInputs, KernelConfig, KernelAttributes, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {argMinMaxImpl} from './arg_min_max_impl';

export function argMin(
    args: {inputs: ArgMinInputs, backend: MathBackendCPU, attrs: KernelAttributes<ArgMinAttrs>}):
    TensorInfo {
  return argMinMaxImpl(args.inputs.x, args.attrs.axis, args.backend, false);
}

export const argMinConfig: KernelConfig = {
  kernelName: ArgMin,
  backendName: 'cpu',
  kernelFunc: argMin
};
