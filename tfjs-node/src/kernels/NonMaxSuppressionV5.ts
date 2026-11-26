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

import {KernelAttributes, KernelConfig, NonMaxSuppressionV5, NonMaxSuppressionV5Attrs, NonMaxSuppressionV5Inputs} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';
import {withNmsInputs} from './native_dispatch';

export const nonMaxSuppressionV5Config: KernelConfig = {
  kernelName: NonMaxSuppressionV5,
  backendName: 'tensorflow',
  kernelFunc: (args: {
    inputs: NonMaxSuppressionV5Inputs, backend: NodeJSKernelBackend,
    attrs: KernelAttributes<NonMaxSuppressionV5Attrs>
  }) => {
    const {boxes, scores} = args.inputs;
    const {backend} = args;
    const opAttrs = [createTensorsTypeOpAttr('T', boxes.dtype)];
    const [selectedIndices, selectedScores, validOutputs] = withNmsInputs(
        boxes, scores, args.attrs, inputs =>
            backend.executeMultipleOutputs(NonMaxSuppressionV5, opAttrs, inputs, 3),
        [args.attrs.softNmsSigma]);
    validOutputs.dispose();
    return [selectedIndices, selectedScores];
  }
};
