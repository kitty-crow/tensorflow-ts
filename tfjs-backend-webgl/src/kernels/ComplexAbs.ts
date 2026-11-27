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

import {backend_util, ComplexAbs, ComplexAbsInputs, KernelConfig, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {ComplexAbsProgram} from '../complex_abs_gpu';

export function complexAbs(
    args: {inputs: ComplexAbsInputs, backend: MathBackendWebGL}): TensorInfo {
  const {inputs, backend} = args;
  const {x} = inputs;

  const xData = backend.texData.get(x.dataId);

  const program = new ComplexAbsProgram(x.shape);
  const parts = backend_util.requiredComplexParts(xData);
  const programInputs = backend_util.complexComponentViews(parts, x.shape);

  return backend.runWebGLProgram(
      program, programInputs, programInputs[0].dtype);
}

export const complexAbsConfig: KernelConfig = {
  kernelName: ComplexAbs,
  backendName: 'webgl',
  kernelFunc: complexAbs
};
