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

import {TensorInfo, util} from '@tensorflow/tfjs-core';

export function assertNotComplex(
    tensor: TensorInfo|TensorInfo[], opName: string): void {
  if (!Array.isArray(tensor)) {
    tensor = [tensor];
  }
  tensor.forEach(t => {
    if (t != null) {
      util.assert(
          t.dtype !== 'complex64',
          () => `${
              opName} does not support complex64 tensors in the CPU backend.`);
    }
  });
}

/** Read the two trailing dimensions of a matrix or a batch of matrices. */
export function matrixDimensions(shape: readonly number[]): [number, number] {
  const [rows, columns] = shape.slice(-2);
  if (rows == null || columns == null) {
    throw new Error('Matrix shape must have two trailing dimensions.');
  }
  return [rows, columns];
}

/** Strides for a rank-three matrix batch, excluding its implicit unit stride. */
export function matrixBatchStrides(strides: readonly number[]): [number, number] {
  const [batch, row] = strides;
  if (batch == null || row == null) {
    throw new Error('Matrix batch must have two explicit strides.');
  }
  return [batch, row];
}

/** Read the three explicit strides of a rank-four tensor. */
export function rankFourStrides(strides: readonly number[]):
    [number, number, number] {
  const [first, second, third] = strides;
  if (first == null || second == null || third == null) {
    throw new Error('Rank-four tensor must have three explicit strides.');
  }
  return [first, second, third];
}

/** Convert physical strides to batch, row, column and channel order. */
export function spatialStrides2D(
    strides: readonly number[], channelsLast: boolean):
    [number, number, number, number] {
  const [batch, second, third] = rankFourStrides(strides);
  return channelsLast ? [batch, second, third, 1] : [batch, third, 1, second];
}

/** Read the four explicit strides of a rank-five tensor. */
export function rankFiveStrides(strides: readonly number[]):
    [number, number, number, number] {
  const [first, second, third, fourth] = strides;
  if (first == null || second == null || third == null || fourth == null) {
    throw new Error('Rank-five tensor must have four explicit strides.');
  }
  return [first, second, third, fourth];
}

/** Read the four dimensions of an NHWC image tensor. */
export function imageDimensions(shape: readonly number[]):
    [number, number, number, number] {
  const [batch, height, width, channels] = shape;
  if (batch == null || height == null || width == null || channels == null) {
    throw new Error('Image shape must have four dimensions.');
  }
  return [batch, height, width, channels];
}
