/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {KernelAttributes, KernelConfig, NonMaxSuppressionV4, NonMaxSuppressionV4Attrs, NonMaxSuppressionV4Inputs} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';
import {withNmsInputs} from './native_dispatch';

export const nonMaxSuppressionV4Config: KernelConfig = {
  kernelName: NonMaxSuppressionV4,
  backendName: 'tensorflow',
  kernelFunc: (args: {
    inputs: NonMaxSuppressionV4Inputs, backend: NodeJSKernelBackend,
    attrs: KernelAttributes<NonMaxSuppressionV4Attrs>
  }) => {
    const {boxes, scores} = args.inputs;
    const {backend} = args;
    const opAttrs = [
      createTensorsTypeOpAttr('T', boxes.dtype),
      createTensorsTypeOpAttr('T_threshold', 'float32'), {
        name: 'pad_to_max_output_size',
        type: backend.binding.TF_ATTR_BOOL,
        value: args.attrs.padToMaxOutputSize
      }
    ];
    const [selectedIndices, validOutputs] = withNmsInputs(
        boxes, scores, args.attrs, inputs =>
            backend.executeMultipleOutputs(NonMaxSuppressionV4, opAttrs, inputs, 2));
    return [selectedIndices, validOutputs];
  }
};
