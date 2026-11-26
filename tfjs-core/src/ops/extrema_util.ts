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

import {ENGINE} from '../engine';
import {Tensor} from '../tensor';
import {TensorLike} from '../types';

import {assertAndGetBroadcastShape} from './broadcast_util';
import {cast} from './cast';
import {matchedBinaryInputs} from './operation';

/** Minimum and Maximum share numeric matching and bool promotion. */
export function runExtremaKernel<T extends Tensor>(
    a: Tensor|TensorLike, b: Tensor|TensorLike, operationName: string,
    kernelName: string): T {
  let {a: left, b: right} = matchedBinaryInputs(a, b, operationName);
  if (left.dtype === 'bool') {
    left = cast(left, 'int32');
    right = cast(right, 'int32');
  }
  assertAndGetBroadcastShape(left.shape, right.shape);
  return ENGINE.runKernel<T>(kernelName, {a: left, b: right});
}
