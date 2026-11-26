/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

/** Reads a required numeric shape coordinate without silently using undefined. */
export function requiredDimension(values: readonly number[], index: number): number {
  const dimension = values[index];
  if (dimension == null) {
    throw new Error(`Missing dimension at index ${index}.`);
  }
  return dimension;
}

/** Returns the two trailing dimensions of a matrix or batched matrix. */
export function matrixDimensions(shape: readonly number[]): [number, number] {
  return [
    requiredDimension(shape, shape.length - 2),
    requiredDimension(shape, shape.length - 1)
  ];
}

/** Reads the four dimensions of a rank-four image. */
export function imageDimensions(shape: readonly number[]):
    [number, number, number, number] {
  return [
    requiredDimension(shape, 0), requiredDimension(shape, 1),
    requiredDimension(shape, 2), requiredDimension(shape, 3)
  ];
}

/** Computes the spatial expansion and channel reduction in layout order. */
export function depthToSpaceOutputShape(
    shape: readonly number[], blockSize: number, dataFormat: 'NHWC'|'NCHW'):
    [number, number, number, number] {
  const [batch, d1, d2, d3] = imageDimensions(shape);
  const depth = dataFormat === 'NHWC' ? d3 : d1;
  const height = dataFormat === 'NHWC' ? d1 : d2;
  const width = dataFormat === 'NHWC' ? d2 : d3;
  const outputDepth = depth / (blockSize * blockSize);
  return dataFormat === 'NHWC' ?
      [batch, height * blockSize, width * blockSize, outputDepth] :
      [batch, outputDepth, height * blockSize, width * blockSize];
}
