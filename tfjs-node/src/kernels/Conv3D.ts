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

import {backend_util, Conv3D, Conv3DAttrs, Conv3DInputs, KernelAttributes, KernelConfig} from '@tensorflow/tfjs';

import {NodeJSKernelBackend} from '../nodejs_kernel_backend';
import {conv3DAttrs} from './native_dispatch';

export const conv3DConfig: KernelConfig = {
  kernelName: Conv3D,
  backendName: 'tensorflow',
  kernelFunc: (args: {
    inputs: Conv3DInputs, backend: NodeJSKernelBackend,
    attrs: KernelAttributes<Conv3DAttrs>
  }) => {
    const {x, filter} = args.inputs;
    const backend = args.backend;
    const {strides, pad, dilations} = args.attrs;

    const convInfo = backend_util.computeConv3DInfo(
        x.shape as [number, number, number, number, number],
        filter.shape as [number, number, number, number, number], strides,
        dilations, pad);

    const opAttrs = conv3DAttrs(backend, x.dtype, convInfo);
    return backend.executeSingleOutput(Conv3D, opAttrs, [x, filter]);
  }
};
