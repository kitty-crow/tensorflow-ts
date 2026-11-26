/**
 * @license
 * Copyright 2023 Google LLC.
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

import {MaxPool3D, MaxPool3DAttrs, MaxPool3DInputs, KernelAttributes, KernelConfig, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {dispatchPool3D} from './pool_dispatch';

export function maxPool3d(args: {
  inputs: MaxPool3DInputs,
  backend: WebGPUBackend,
  attrs: KernelAttributes<MaxPool3DAttrs>
}): TensorInfo {
  return dispatchPool3D(args, 'max');
}

export const maxPool3DConfig: KernelConfig = {
  kernelName: MaxPool3D,
  backendName: 'webgpu',
  kernelFunc: maxPool3d
};
