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
import {assertAndGetBroadcastShape, getReductionAxes} from '../ops/broadcast_util';
import {reshape} from '../ops/reshape';
import {sum} from '../ops/sum';
import {Tensor} from '../tensor';
import {assertArrayLength} from '../util_base';

/** Sum only broadcasted axes; retain the original tensor when no sum is needed. */
export function sumBroadcastGradient(
    gradient: Tensor, inputShape: number[], outputShape: number[]): Tensor {
  const axes = getReductionAxes(inputShape, outputShape);
  if (axes.length === 0) {
    return gradient;
  }
  return sum(gradient, axes);
}

/** Some gradient contracts reshape only after an actual reduction. */
export function reduceBroadcastGradient(
    gradient: Tensor, inputShape: number[], outputShape: number[]): Tensor {
  const axes = getReductionAxes(inputShape, outputShape);
  if (axes.length === 0) {
    return gradient;
  }
  return reshape(sum(gradient, axes), inputShape);
}

/** Checks saved operands and calculates their common broadcast shape. */
export function savedBroadcastOperands(saved: Tensor[]):
    {a: Tensor, b: Tensor, outShape: number[]} {
  assertArrayLength(saved, 2);
  const [a, b] = saved;
  return {a, b, outShape: assertAndGetBroadcastShape(a.shape, b.shape)};
}
