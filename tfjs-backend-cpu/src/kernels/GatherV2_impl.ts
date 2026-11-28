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

import {buffer, DataType, Rank, TensorBuffer} from '@tensorflow/tfjs-core';

import {requiredCpuEntry} from '../cpu_util';

export function gatherV2Impl<R extends Rank, D extends DataType>(
    xBuf: TensorBuffer<R, D>, indicesBuf: TensorBuffer<R, D>,
    flattenOutputShape: number[]): TensorBuffer<R, D> {
  const outBuf = buffer(flattenOutputShape, xBuf.dtype);
  for (let i = 0; i < outBuf.size; ++i) {
    const newLoc = outBuf.indexToLoc(i);

    const originalLoc: number[] = newLoc.slice();
    const batchIdx = requiredCpuEntry(originalLoc, 0);
    const indicesIdx = requiredCpuEntry(originalLoc, 2);
    const indicesIndex = indicesBuf.locToIndex([batchIdx, indicesIdx]);
    const index = indicesBuf.values[indicesIndex];
    if (typeof index !== 'number') {
      throw new Error(`Missing or invalid gather index at ${indicesIndex}.`);
    }
    originalLoc[2] = index;

    const originalIndex = xBuf.locToIndex(originalLoc);

    // Out-of-bounds indices leave the default zero value in outBuf.
    if (!(0 <= originalIndex && originalIndex < xBuf.values.length)) {
      continue;
    }
    const value = xBuf.values[originalIndex];
    if (value == null) {
      throw new Error(`Missing gather value at index ${originalIndex}.`);
    }
    outBuf.values[i] = value;
  }

  return outBuf as TensorBuffer<R, D>;
}
