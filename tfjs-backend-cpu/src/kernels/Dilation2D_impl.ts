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

import {backend_util} from '@tensorflow/tfjs-core';

import {requiredCpuEntry} from '../cpu_util';

/** The forward and backward kernels share the same strict maximum search. */
export function dilation2DWindow(
    info: backend_util.Conv2DInfo, hBeg: number, wBeg: number,
    readCandidate: (hIn: number, wIn: number, h: number, w: number) => number):
    {value: number, hIn: number, wIn: number, h: number, w: number} {
  // Preserve the legacy fallback coordinates when no candidate exceeds the
  // sentinel, and retain the first candidate when maxima are tied.
  let value = Number.MIN_SAFE_INTEGER;
  let bestHIn = hBeg < 0 ? 0 : hBeg;
  let bestWIn = wBeg < 0 ? 0 : wBeg;
  let bestH = 0;
  let bestW = 0;
  for (let h = 0; h < info.filterHeight; ++h) {
    const hIn = hBeg + h * info.dilationHeight;
    if (!(hIn >= 0 && hIn < info.inHeight)) {
      continue;
    }
    for (let w = 0; w < info.filterWidth; ++w) {
      const wIn = wBeg + w * info.dilationWidth;
      if (!(wIn >= 0 && wIn < info.inWidth)) {
        continue;
      }
      const candidate = readCandidate(hIn, wIn, h, w);
      if (candidate > value) {
        value = candidate;
        bestHIn = hIn;
        bestWIn = wIn;
        bestH = h;
        bestW = w;
      }
    }
  }
  return {value, hIn: bestHIn, wIn: bestWIn, h: bestH, w: bestW};
}

/** Read a channel row in a nested NHWC tensor or HWC filter. */
export function dilationFilterRow(values: number[][][], h: number, w: number):
    number[] {
  return requiredCpuEntry(requiredCpuEntry(values, h), w);
}

export function dilationInputRow(
    values: number[][][][], batch: number, h: number, w: number): number[] {
  return dilationFilterRow(requiredCpuEntry(values, batch), h, w);
}

/** Accumulate in the original JavaScript number array before dtype conversion. */
export function accumulateDilationGradient(
    row: number[], channel: number, value: number): void {
  row[channel] = requiredCpuEntry(row, channel) + value;
}
