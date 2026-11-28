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

import {TypedArray, util} from '@tensorflow/tfjs-core';
import {requiredCpuEntry} from '../cpu_util';

/** Keep empty tokens only when requested, for either delimiter path. */
function appendToken(
    result: Uint8Array[], token: Uint8Array, skipEmpty: boolean): void {
  if (!skipEmpty || token.length !== 0) {
    result.push(token);
  }
}

function split(
    str: Uint8Array, delimiters: Uint8Array, skipEmpty: boolean,
    result: Uint8Array[]): void {
  if (!str.length) {
    return;
  }
  // When the delimiter is empty, the input is split into individual characters.
  if (delimiters.length === 0) {
    for (let i = 0; i < str.length; ++i) {
      result.push(str.subarray(i, i + 1));
    }
    return;
  }
  // When there is one delimiter, the input is split only at that delimiter.
  if (delimiters.length === 1) {
    const delimiter = requiredCpuEntry(delimiters, 0);
    let f = str.indexOf(delimiter);
    while (f !== -1) {
      const token = str.subarray(0, f);
      appendToken(result, token, skipEmpty);
      str = str.subarray(f + 1);
      f = str.indexOf(delimiter);
    }
    appendToken(result, str, skipEmpty);
    return;
  }
  // When there are multiple delimiters, the input is split at every instance
  // one of the delimiters appears.
  let tokenStart = 0;
  for (let i = 0; i < str.length + 1; i++) {
    if (!((i === str.length) || (delimiters.indexOf(requiredCpuEntry(str, i)) !== -1))) continue;
    {
      const token = str.subarray(tokenStart, i);
      appendToken(result, token, skipEmpty);
      tokenStart = i + 1;
    }
  }
}

export function stringSplitImpl(
    input: Uint8Array[], delimiter: Uint8Array,
    skipEmpty: boolean): [TypedArray, Uint8Array[], [number, number]] {
  const batchSize = input.length;

  // Empty delimiter means split the input character by character.
  const tokens: Uint8Array[] = [];

  let outputSize = 0;
  let maxNumEntries = 0;
  const numIndices: number[] = new Array<number>(batchSize);
  for (let i = 0; i < batchSize; ++i) {
    const prevTokensLength = tokens.length;
    split(requiredCpuEntry(input, i), delimiter, skipEmpty, tokens);
    const nEntries = tokens.length - prevTokensLength;
    numIndices[i] = nEntries;
    outputSize += nEntries;
    maxNumEntries = Math.max(maxNumEntries, nEntries);
  }

  const indices = util.getArrayFromDType('int32', outputSize * 2) as TypedArray;
  const values: Uint8Array[] = new Array<Uint8Array>(outputSize);
  const shape: [number, number] = [batchSize, maxNumEntries];

  let c = 0;
  for (let i = 0; i < batchSize; ++i) {
    for (let j = 0; j < requiredCpuEntry(numIndices, i); ++j) {
      // indices is a 2d tensor with shape of [outputSize, 2]
      indices[c * 2] = i;
      indices[c * 2 + 1] = j;
      values[c] = requiredCpuEntry(tokens, c);
      ++c;
    }
  }

  return [indices, values, shape];
}
