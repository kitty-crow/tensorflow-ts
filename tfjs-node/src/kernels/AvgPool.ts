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

import {AvgPool, AvgPoolAttrs, AvgPoolInputs, backend_util, KernelAttributes, KernelConfig} from '@tensorflow/tfjs';

import {NodeJSKernelBackend} from '../nodejs_kernel_backend';
import {pool2DAttrs} from './native_dispatch';

export const avgPoolConfig: KernelConfig = {
  kernelName: AvgPool,
  backendName: 'tensorflow',
  kernelFunc: (args: {
    inputs: AvgPoolInputs, backend: NodeJSKernelBackend,
    attrs: KernelAttributes<AvgPoolAttrs>
  }) => {
    const {x} = args.inputs;
    const backend = args.backend;
    const {filterSize, strides, pad, dimRoundingMode} =
        args.attrs;

    const convInfo = backend_util.computePool2DInfo(
        x.shape as [number, number, number, number], filterSize, strides,
        1 /* dilations */, pad, dimRoundingMode);

    const opAttrs = pool2DAttrs(backend, x.dtype, convInfo, 'padding');
    return backend.executeSingleOutput(AvgPool, opAttrs, [x]);
  }
};
