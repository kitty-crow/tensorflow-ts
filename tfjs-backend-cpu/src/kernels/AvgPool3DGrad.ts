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

import {AvgPool3DGrad, AvgPool3DGradAttrs, AvgPool3DGradInputs, KernelConfig, KernelAttributes, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {pool3DGrad} from './pool3d_grad_impl';

export function avgPool3DGrad(args: {
  inputs: AvgPool3DGradInputs,
  backend: MathBackendCPU,
  attrs: KernelAttributes<AvgPool3DGradAttrs>
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  return pool3DGrad(inputs.input, inputs.dy, backend, attrs, 'avg');
}

export const avgPool3DGradConfig: KernelConfig = {
  kernelName: AvgPool3DGrad,
  backendName: 'cpu',
  kernelFunc: avgPool3DGrad
};
