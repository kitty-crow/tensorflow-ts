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
import {Tensor, Tensor4D} from '../tensor';
import {Rank, ShapeMap} from '../types';
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
