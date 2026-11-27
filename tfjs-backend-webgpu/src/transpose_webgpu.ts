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

import {getCoordsDataType, getCoordsXYZ, getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {shaderEntry} from './shader_util';
import {FlatDispatchProgram} from './flat_dispatch_program';

const workPerThread = 1;

export class TransposeProgram extends FlatDispatchProgram implements WebGPUProgram {
  shaderKey: string;
  workPerThread = workPerThread;
  newDim: number[];

  constructor(aShape: number[], newDim: number[]) {
    const outputShape: number[] = new Array<number>(aShape.length);
    for (let i = 0; i < outputShape.length; i++) {
      outputShape[i] = shaderEntry(aShape, shaderEntry(newDim, i));
    }
    super(outputShape, [64, 1, 1], ['A'], [workPerThread, 1, 1]);

    this.newDim = newDim;
    this.shaderKey = `transpose_${newDim}`;
  }

  getUserCode(): string {
    const dtype = getCoordsDataType(this.outputShape.length);
    const switched = getSwitchedCoords(this.newDim);

    const userCode = `
      ${main('index')} {
        for(var i = 0; i < ${this.workPerThread}; i = i + 1) {
          let flatIndex = index * ${this.workPerThread} + i;
          if(flatIndex < uniforms.size) {
            let coords = getCoordsFromIndex(flatIndex);
            setOutputAtIndex(flatIndex, A[getIndexFromCoords${
        this.outputShape.length}D(
              ${dtype}(${switched}), uniforms.aShape)]);
          }
        }
      }
    `;
    return userCode;
  }
}

export function getSwitchedCoords(newDim: number[]): string {
  const rank = newDim.length;
  if (rank > 6) {
    throw Error(`Transpose for rank ${rank} is not yet supported`);
  }
  const switchedCoords = new Array<string>(rank);
  for (let i = 0; i < newDim.length; i++) {
    switchedCoords[shaderEntry(newDim, i)] = `coords.${getCoordsXYZ(i)}`;
  }

  return switchedCoords.join();
}
