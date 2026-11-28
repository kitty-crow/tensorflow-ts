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
import {Tensor, Tensor1D, Tensor4D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {Rank, ShapeMap, TensorLike} from '../types';
import * as util from '../util';
import {reshape} from './reshape';

/**
 * The fused batch-normalization kernel expects leading singleton axes for
 * ranks below four. A null result preserves the existing no-reshape path.
 */
export function getBatchNormReshapeShape(
    shape: readonly number[], size: number): ShapeMap[Rank.R4]|null {
  if (shape.length >= 4) {
    return null;
  }
  if (shape.length <= 1) {
    return [1, 1, 1, size];
  }

  const first = shape[0] ?? null;
  const second = shape[1] ?? null;
  if (first === null || second === null) {
    throw new Error('Batch normalization input shape is missing a dimension.');
  }
  if (shape.length === 2) {
    return [1, 1, first, second];
  }

  const third = shape[2] ?? null;
  if (third === null) {
    throw new Error('Batch normalization input shape is missing a dimension.');
  }
  return [1, first, second, third];
}

export function xAs4D<R extends Rank>(x: Tensor<R>): Tensor4D {
  const shape = getBatchNormReshapeShape(x.shape, x.size);
  if (shape === null) {
    return x as Tensor4D;
  }
  return reshape<Rank.R4>(x, shape);
}

/** Check a rank-specific batch normalisation input with the original error text. */
export function assertBatchNormInputRank(
    rank: number, name: 'x'|'mean'|'variance'|'scale'|'offset',
    actualRank: number): void {
  if (name === 'x') {
    util.assert(
        actualRank === rank,
        () => `Error in batchNorm${rank}D: x must be rank ${rank} but got rank ${actualRank}.`);
    return;
  }
  util.assert(
      actualRank === rank || actualRank === 1,
      () => `Error in batchNorm${rank}D: ${name} must be rank ${rank} or rank 1 but got rank ${actualRank}.`);
}

/** Convert and validate in the same order for all three rank-specific ops. */
export function prepareRankedBatchNormInputs<R extends Rank>(
    rank: number, x: Tensor<R>|TensorLike, mean: Tensor<R>|Tensor1D|TensorLike,
    variance: Tensor<R>|Tensor1D|TensorLike,
    offset: Tensor<R>|Tensor1D|TensorLike,
    scale: Tensor<R>|Tensor1D|TensorLike) {
  const $x = convertToTensor(x, 'x', 'batchNorm');
  const $mean = convertToTensor(mean, 'mean', 'batchNorm');
  const $variance = convertToTensor(variance, 'variance', 'batchNorm');
  let $scale: Tensor<R>|Tensor1D;
  if (scale != null) {
    $scale = convertToTensor(scale, 'scale', 'batchNorm');
  }
  let $offset: Tensor<R>|Tensor1D;
  if (offset != null) {
    $offset = convertToTensor(offset, 'offset', 'batchNorm');
  }
  assertBatchNormInputRank(rank, 'x', $x.rank);
  assertBatchNormInputRank(rank, 'mean', $mean.rank);
  assertBatchNormInputRank(rank, 'variance', $variance.rank);
  if ($scale != null) {
    assertBatchNormInputRank(rank, 'scale', $scale.rank);
  }
  if ($offset != null) {
    assertBatchNormInputRank(rank, 'offset', $offset.rank);
  }
  return {x: $x, mean: $mean, variance: $variance, offset: $offset, scale: $scale};
}
