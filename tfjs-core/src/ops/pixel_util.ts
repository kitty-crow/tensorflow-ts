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

import {requiredDimension, requiredKernelEntry} from './shape_dimension';

/** Reads height, width and channel depth for a rank-two or rank-three image. */
export function pixelGeometry(shape: readonly number[]): [number, number, number] {
  return [
    requiredDimension(shape, 0), requiredDimension(shape, 1),
    shape.length === 2 ? 1 : requiredDimension(shape, 2)
  ];
}

/** Convert tensor pixels to RGBA bytes, preserving explicit alpha and rounding. */
export function pixelBytes(
    data: ArrayLike<number>, height: number, width: number, depth: number,
    dtype: string, defaultAlpha: number): Uint8ClampedArray {
  const multiplier = dtype === 'float32' ? 255 : 1;
  const bytes = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < height * width; ++i) {
    const rgba: [number, number, number, number] = [0, 0, 0, defaultAlpha];
    for (let d = 0; d < depth; d++) {
      const value = requiredKernelEntry(data, i * depth + d);
      validatePixelComponent(dtype, value);
      if (depth === 1) {
        rgba[0] = value * multiplier;
        rgba[1] = value * multiplier;
        rgba[2] = value * multiplier;
      } else {
        rgba[d] = value * multiplier;
      }
    }
    const j = i * 4;
    bytes[j] = Math.round(rgba[0]);
    bytes[j + 1] = Math.round(rgba[1]);
    bytes[j + 2] = Math.round(rgba[2]);
    bytes[j + 3] = Math.round(rgba[3]);
  }
  return bytes;
}

/** Validate the native pixel range for the Tensor's element type. */
function validatePixelComponent(dtype: string, value: number): void {
  if (dtype === 'float32' && (value < 0 || value > 1)) {
    throw new Error(
        `Tensor values for a float32 Tensor must be in the ` +
        `range [0 - 1] but encountered ${value}.`);
  }
  if (dtype === 'int32' && (value < 0 || value > 255)) {
    throw new Error(
        `Tensor values for a int32 Tensor must be in the ` +
        `range [0 - 255] but encountered ${value}.`);
  }
}
