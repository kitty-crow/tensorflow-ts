/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {ComplexAbs, ComplexAbsInputs, KernelConfig, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

export const complexAbs =
    (args: {inputs: ComplexAbsInputs, backend: MathBackendCPU}) => {
      const {x} = args.inputs;
      const cpuBackend = args.backend;
      const resultValues = new Float32Array(util.sizeFromShape(x.shape));
      const {real, imag} = cpuBackend.requireComplexParts(x.dataId);
      const realVals = cpuBackend.requireValues(real.dataId) as Float32Array;
      const imagVals = cpuBackend.requireValues(imag.dataId) as Float32Array;
      for (let i = 0; i < realVals.length; i++) {
        const real = realVals[i];
        const imag = imagVals[i];
        resultValues[i] = Math.hypot(real, imag);
      }

      return cpuBackend.makeOutput(resultValues, x.shape, 'float32');
    };

export const complexAbsConfig: KernelConfig = {
  kernelName: ComplexAbs,
  backendName: 'cpu',
  kernelFunc: complexAbs,
};
