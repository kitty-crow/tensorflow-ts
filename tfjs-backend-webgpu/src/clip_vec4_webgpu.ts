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

import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {FlatDispatchProgram} from './flat_dispatch_program';

const workPerThread = 4;

export class ClipVec4Program extends FlatDispatchProgram implements WebGPUProgram {
  shaderKey: string;
  uniforms = 'minVal : f32, maxVal : f32,';
  workPerThread = workPerThread;
  outputComponent = 4;

  constructor(outputShape: number[]) {
    super(outputShape, [64, 1, 1], ['A'], [workPerThread, 1, 1]);
    this.shaderKey = 'clipVec4';
  }

  getUserCode(): string {
    const userCode = `
      ${main('index')} {
        if(index < uniforms.size) {
          let value = getAByOutputIndex(index);
          var clampedValue = clamp(
              value, vec4<f32>(uniforms.minVal), vec4<f32>(uniforms.maxVal));
          clampedValue = select(clampedValue, value, isnanVec4(value));
          setOutputAtIndex(index, clampedValue);
        }
      }
    `;
    return userCode;
  }
}
