/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {getCoordsDataType, getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {FlatDispatchProgram} from './flat_dispatch_program';

export class GatherNDProgram extends FlatDispatchProgram implements WebGPUProgram {
  shaderKey: string;
  uniforms: string;
  sliceDim: number;
  constructor(sliceDim: number, shape: number[]) {
    super(shape, [64, 1, 1], ['A', 'indices']);
    this.shaderKey = `gathernd_${sliceDim}`;
    this.sliceDim = sliceDim;
    this.uniforms = `sliceDim : i32, strides : ${getCoordsDataType(sliceDim)},`;
  }

  getUserCode(): string {
    let strideString: string;
    if (this.sliceDim > 1) {
      strideString = 'uniforms.strides[j]';
    } else {
      strideString = 'uniforms.strides';
    }
    const userCode = `
      ${main('index')} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          var flattenIndex = 0;
          for (var j = 0; j < uniforms.sliceDim; j = j + 1) {
            let indexTemp = i32(round(getIndices(coords[0], j)));
            let strideNum = ${strideString};
            flattenIndex = flattenIndex + indexTemp * strideNum;
          }

          setOutputAtIndex(index, getA(flattenIndex, coords[1]));
        }
      }
      `;
    return userCode;
  }
}
