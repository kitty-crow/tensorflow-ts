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

import {Any, AnyAttrs, AnyInputs, KernelConfig, KernelAttributes, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {axisReduction} from './axis_reduction';

export function any(
    args: {inputs: AnyInputs, backend: MathBackendWebGL, attrs: KernelAttributes<AnyAttrs>}):
    TensorInfo {
  return axisReduction(args, 'any');
}

export const anyConfig: KernelConfig = {
  kernelName: Any,
  backendName: 'webgl',
  kernelFunc: any
};
