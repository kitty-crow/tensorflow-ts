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

import {getUnaryOpString, UnaryOpType} from './unary_op_util';
import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {FlatDispatchProgram} from './flat_dispatch_program';

export class UnaryOpProgram extends FlatDispatchProgram implements WebGPUProgram {
  shaderKey: string;
  op: UnaryOpType;
  uniforms?: string;

  constructor(outputShape: number[], op: UnaryOpType, uniforms = '') {
    // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
    const workgroupSizeX = 128;
    super(outputShape, [workgroupSizeX, 1, 1], ['A']);
    this.op = op;
    if (uniforms !== '') {
      this.uniforms = uniforms;
    }
    this.shaderKey = `unary_${op}`;
  }

  getUserCode(): string {
    return `
      fn unaryOperation(a : f32) -> f32 {
        ${getUnaryOpString(this.op, false)}
      }
      ${main('index')} {
        if (index < uniforms.size) {
          let a = getAByOutputIndex(index);
          setOutputAtIndex(index, unaryOperation(a));
        }
      }
      `;
  }
}
