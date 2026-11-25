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

import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {FlatDispatchProgram} from './flat_dispatch_program';

const workPerThread = 1;

export class AddNPackedProgram extends FlatDispatchProgram implements WebGPUProgram {
  shaderKey: string;
  workPerThread = workPerThread;

  constructor(shapes: number[][]) {
    super(
        shapes[0], [64, 1, 1], shapes.map((_, i) => `T${i}`),
        [workPerThread, 1, 1]);
    this.shaderKey = 'addN';
  }

  getUserCode(): string {
    const snippets: string[] = [];
    // Get target elements from every input tensor.
    this.variableNames.forEach(variable => {
      snippets.push(`let v${variable} = get${variable}ByOutputCoords(coords);`);
    });
    // Calculate the sum of all elements.
    const operation = this.variableNames
                          .map(variable => {
                            return `v${variable}`;
                          })
                          .join(' + ');

    const userCode = `
      ${main('index')} {
        for (var i = 0; i < ${this.workPerThread}; i = i + 1) {
          let flatIndex = index * ${this.workPerThread} + i;
          if (flatIndex < uniforms.size) {
            let coords = getCoordsFromIndex(flatIndex);
            ${snippets.join('\n        ')}
            setOutputAtIndex(flatIndex, ${operation});
          }
        }
      }
    `;
    return userCode;
  }
}
