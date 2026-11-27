/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {TypedArray, util} from '@tensorflow/tfjs-core';
import {requiredCpuEntry} from '../cpu_util';

/** Find the first insertion point, retaining distinct equal-value policies. */
function insertionBound(array: TypedArray, value: number, side: 'left'|'right') {
  let left = 0;
  let right = array.length;
  while (left < right) {
    const mid = Math.floor((left + right) / 2);
    const entry = requiredCpuEntry(array, mid);
    const advance = side === 'left' ? entry < value : entry <= value;
    if (advance) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return right;
}

export function searchSortedImpl(
    sortedInputs: TypedArray, values: TypedArray, batchSize: number,
    numInputs: number, numValues: number, side: 'left'|'right'): TypedArray {
  const output =
      util.getArrayFromDType('int32', batchSize * numValues) as TypedArray;
  for (let b = 0; b < batchSize; ++b) {
    const sortedInputsSlice =
        sortedInputs.slice(b * numInputs, (b + 1) * numInputs);
    const outputOffset = b * numValues;
    for (let i = 0; i < numValues; ++i) {
      output[outputOffset + i] = insertionBound(
          sortedInputsSlice, requiredCpuEntry(values, i + outputOffset), side);
    }
  }
  return output;
}
