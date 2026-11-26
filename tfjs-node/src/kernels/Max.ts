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

import {KernelAttributes, KernelConfig, Max, MaxAttrs, MaxInputs} from '@tensorflow/tfjs';

import {NodeJSKernelBackend} from '../nodejs_kernel_backend';
import {executeNumericReduction} from './native_dispatch';

export const maxConfig: KernelConfig = {
  kernelName: Max,
  backendName: 'tensorflow',
  kernelFunc: (args: {
    inputs: MaxInputs, backend: NodeJSKernelBackend,
    attrs: KernelAttributes<MaxAttrs>
  }) => {
    const {x} = args.inputs;
    const backend = args.backend;
    const {reductionIndices, keepDims} = args.attrs;

    return executeNumericReduction(backend, Max, x, reductionIndices, keepDims);
  }
};
