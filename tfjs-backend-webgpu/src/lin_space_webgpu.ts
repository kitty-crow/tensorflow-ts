/**
 * @license
 * Copyright 2022 Google LLC.
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

export class LinSpaceProgram extends FlatDispatchProgram implements WebGPUProgram {
  shaderKey: string;
  uniforms = 'start : f32, step : f32,';

  constructor(shape: number) {
    super([shape], [64, 1, 1], []);

    this.shaderKey = 'linSpace';
  }

  getUserCode(): string {
    const userCode = `
      ${main('index')} {
        if (index < uniforms.size) {
          setOutputAtIndex(index, uniforms.start + f32(index) * uniforms.step);
        }
      }
    `;
    return userCode;
  }
}
