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

import {Tile} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {add} from '../ops/add';
import {slice} from '../ops/slice';
import {zerosLike} from '../ops/zeros_like';
import {Tensor} from '../tensor';
import {assertArrayLength, isNumericArray} from '../util_base';

export const tileGradConfig: GradConfig = {
  kernelName: Tile,
  inputsToSave: ['x'],
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const [x] = saved;
    const {reps} = attrs;
    if (!isNumericArray(reps)) {
      throw new Error('Tile gradient requires a numeric repetition vector.');
    }

    const derX = () => {
      let xGrad = zerosLike(x);
      const shape = x.shape;
      // TODO(cais): Maybe reduce memory footprint by avoiding repeated
      // slicing.
      if (x.rank === 1) {
        assertArrayLength(shape, 1);
        assertArrayLength(reps, 1);
        for (let i = 0; i < reps[0]; ++i) {
          xGrad = add(xGrad, slice(dy, [i * shape[0]], [shape[0]]));
        }
        return xGrad;
      }
      if (x.rank === 2) {
        assertArrayLength(shape, 2);
        assertArrayLength(reps, 2);
        for (let i = 0; i < reps[0]; ++i) {
          for (let j = 0; j < reps[1]; ++j) {
            xGrad = add(xGrad, slice(dy, [i * shape[0], j * shape[1]], [
                          shape[0], shape[1]
                        ]));
          }
        }
        return xGrad;
      }
      if (x.rank === 3) {
        assertArrayLength(shape, 3);
        assertArrayLength(reps, 3);
        for (let i = 0; i < reps[0]; ++i) {
          for (let j = 0; j < reps[1]; ++j) {
            for (let k = 0; k < reps[2]; ++k) {
              xGrad =
                  add(xGrad,
                      slice(
                          dy, [i * shape[0], j * shape[1], k * shape[2]],
                          [shape[0], shape[1], shape[2]]));
            }
          }
        }
        return xGrad;
      }
      if (x.rank === 4) {
        assertArrayLength(shape, 4);
        assertArrayLength(reps, 4);
        for (let i = 0; i < reps[0]; ++i) {
          for (let j = 0; j < reps[1]; ++j) {
            for (let k = 0; k < reps[2]; ++k) {
              for (let l = 0; l < reps[3]; ++l) {
                xGrad =
                    add(xGrad,
                        slice(
                            dy,
                            [
                              i * shape[0], j * shape[1], k * shape[2],
                              l * shape[3]
                            ],
                            [shape[0], shape[1], shape[2], shape[3]]));
              }
            }
          }
        }
        return xGrad;
      }
      throw new Error(
          `Gradient for tile operation is not implemented for rank-` +
          `${x.rank} tensors yet.`);
    };
    return {x: derX};
  },
};
