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

import {AvgPoolGrad, AvgPoolGradAttrs, AvgPoolGradInputs, backend_util, KernelConfig, KernelAttributes, tensor1d} from '@tensorflow/tfjs';

import {NodeJSKernelBackend} from '../nodejs_kernel_backend';
import {pool2DAttrs} from './native_dispatch';

export const avgPoolGradConfig: KernelConfig = {
  kernelName: AvgPoolGrad,
  backendName: 'tensorflow',
  kernelFunc: (args: {
    inputs: AvgPoolGradInputs, backend: NodeJSKernelBackend,
    attrs: KernelAttributes<AvgPoolGradAttrs>
  }) => {
    const {dy, input} = args.inputs;
    const backend = args.backend;
    const {filterSize, strides, pad} =
        args.attrs;

    const convInfo = backend_util.computePool2DInfo(
        input.shape as [number, number, number, number], filterSize, strides,
        1 /* dilations */, pad);

    const opAttrs = pool2DAttrs(backend, input.dtype, convInfo, 'padding type');
    const origInputShape = tensor1d(input.shape, 'int32');
    const ret =
        backend.executeSingleOutput(AvgPoolGrad, opAttrs, [origInputShape, dy]);
    origInputShape.dispose();
    return ret;
  }
};
