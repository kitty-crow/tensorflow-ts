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

import {backend_util, DepthToSpace, DepthToSpaceAttrs, DepthToSpaceInputs, KernelConfig, KernelAttributes, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {DepthToSpaceProgram} from '../depth_to_space_gpu';

export function depthToSpace(args: {
  inputs: DepthToSpaceInputs,
  backend: MathBackendWebGL,
  attrs: KernelAttributes<DepthToSpaceAttrs>
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {blockSize, dataFormat} = attrs;

  const outputShape =
      backend_util.depthToSpaceOutputShape(x.shape, blockSize, dataFormat);

  const program = new DepthToSpaceProgram(outputShape, blockSize, dataFormat);
  return backend.runWebGLProgram(program, [x], x.dtype);
}

export const depthToSpaceConfig: KernelConfig = {
  kernelName: DepthToSpace,
  backendName: 'webgl',
  kernelFunc: depthToSpace
};
