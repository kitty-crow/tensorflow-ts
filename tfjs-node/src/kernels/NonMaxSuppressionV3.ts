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

import {KernelAttributes, KernelConfig, NonMaxSuppressionV3, NonMaxSuppressionV3Attrs, NonMaxSuppressionV3Inputs} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';
import {withNmsInputs} from './native_dispatch';

export const nonMaxSuppressionV3Config: KernelConfig = {
  kernelName: NonMaxSuppressionV3,
  backendName: 'tensorflow',
  kernelFunc: (args: {
    inputs: NonMaxSuppressionV3Inputs, backend: NodeJSKernelBackend,
    attrs: KernelAttributes<NonMaxSuppressionV3Attrs>
  }) => {
    const {boxes, scores} = args.inputs;
    const {backend} = args;
    const opAttrs = [createTensorsTypeOpAttr('T', boxes.dtype)];
    return withNmsInputs(boxes, scores, args.attrs, inputs =>
        backend.executeSingleOutput(NonMaxSuppressionV3, opAttrs, inputs));
  }
};
