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

import {KernelAttributes, KernelConfig, LRN, LRNAttrs, LRNInputs} from '@tensorflow/tfjs';

import {NodeJSKernelBackend} from '../nodejs_kernel_backend';
import {localResponseAttrs} from './native_dispatch';

// tslint:disable-next-line: variable-name
export const LRNConfig: KernelConfig = {
  kernelName: LRN,
  backendName: 'tensorflow',
  kernelFunc: (args: {
    inputs: LRNInputs, backend: NodeJSKernelBackend,
    attrs: KernelAttributes<LRNAttrs>
  }) => {
    const {x} = args.inputs;
    const backend = args.backend;
    const {depthRadius, bias, alpha, beta} = args.attrs;

    const opAttrs = localResponseAttrs(backend, x.dtype, {depthRadius, bias, alpha, beta});
    return backend.executeSingleOutput(LRN, opAttrs, [x]);
  }
};
