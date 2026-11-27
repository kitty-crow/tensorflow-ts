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

import {TypedArray, util} from '@tensorflow/tfjs-core';

type ReductionMode = 'all'|'any'|'min'|'max';

function combiner(mode: ReductionMode): (a: number, b: number) => number {
  switch (mode) {
    case 'all': return (a, b) => a && b;
    case 'any': return (a, b) => a || b;
    case 'min': return (a, b) => Number.isNaN(b) || b < a ? b : a;
    case 'max': return (a, b) => Number.isNaN(b) || b > a ? b : a;
  }
}

/** Reduce contiguous blocks, retaining first-value ties and NaN propagation. */
export function reduceValues(
    input: TypedArray, reduceSize: number, output: TypedArray,
    mode: ReductionMode): void {
  if (reduceSize === 0) {
    // Preserve the previous undefined-to-typed-array conversion: NaN for
    // float storage, zero for integer and boolean storage.
    output.fill(NaN);
    return;
  }
  const combine = combiner(mode);
  for (let i = 0; i < output.length; ++i) {
    const offset = i * reduceSize;
    let value = util.requiredNumericEntry(input, offset);
    for (let j = 1; j < reduceSize; ++j) {
      value = combine(value, util.requiredNumericEntry(input, offset + j));
    }
    output[i] = value;
  }
}
