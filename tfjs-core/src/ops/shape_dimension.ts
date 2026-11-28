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

/** Required kernel inputs and metadata must not silently become absent. */
export function requiredKernelEntry<T>(values: ArrayLike<T>, index: number): T {
  const value = values[index];
  if (value == null) {
    throw new Error(`Missing kernel entry at index ${index}.`);
  }
  return value;
}

/** Permute a dense shape without losing zero dimensions or accepting missing axes. */
export function permutedShape(shape: readonly number[], permutation: readonly number[]): number[] {
  if (shape.length !== permutation.length) {
    throw new Error('Permutation length must match the shape rank.');
  }
  const result: number[] = [];
  for (let i = 0; i < shape.length; i++) {
    result.push(requiredDimension(shape, requiredDimension(permutation, i)));
  }
  return result;
}

/** Compute the output shape and number of slices for a normalized unpack axis. */
export function unpackShape(shape: readonly number[], axis: number):
    {num: number, outShape: number[]} {
  const num = requiredDimension(shape, axis);
  const outShape: number[] = [];
  for (let i = 0; i < shape.length; i++) {
    if (i !== axis) {
      outShape.push(requiredDimension(shape, i));
    }
  }
  return {num, outShape};
}

/** Read the spatial result of a kernel invoked with an inserted batch axis. */
export function unbatchedShape(shape: readonly number[], rank: 4):
    [number, number, number];
export function unbatchedShape(shape: readonly number[], rank: 5):
    [number, number, number, number];
export function unbatchedShape(shape: readonly number[], rank: 4|5):
    [number, number, number]|[number, number, number, number];
export function unbatchedShape(shape: readonly number[], rank: 4|5):
    [number, number, number]|[number, number, number, number] {
  if (shape.length !== rank) {
    throw new Error(`Expected a rank-${rank} batched kernel result.`);
  }
  // Preserve reshape's fixed-length contract while checking each spatial axis.
  const spatial: [number, number, number] = [
    requiredDimension(shape, 1), requiredDimension(shape, 2),
    requiredDimension(shape, 3)
  ];
  if (rank === 4) {
    return spatial;
  }
  return [...spatial, requiredDimension(shape, 4)];
}

/** Insert a singleton batch axis while checking spatial dimensions. */
export function batchedImageShape(shape: readonly number[], rank: 4):
    [number, number, number, number];
export function batchedImageShape(shape: readonly number[], rank: 5):
    [number, number, number, number, number];
export function batchedImageShape(shape: readonly number[], rank: 4|5):
    [number, number, number, number]|[number, number, number, number, number] {
  if (shape.length !== rank - 1) {
    throw new Error(`Expected a rank-${rank - 1} unbatched image.`);
  }
  const spatial: [number, number, number, number] = [
    1, requiredDimension(shape, 0), requiredDimension(shape, 1),
    requiredDimension(shape, 2)
  ];
  if (rank === 4) {
    return spatial;
  }
  return [...spatial, requiredDimension(shape, 3)];
}
