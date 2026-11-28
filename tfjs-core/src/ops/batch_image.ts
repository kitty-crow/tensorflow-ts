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

import {Tensor, Tensor4D, Tensor5D} from '../tensor';
import {Rank} from '../types';
import {reshape} from './reshape';
import {batchedImageShape} from './shape_dimension';

/** Inserts a batch axis only for an unbatched 2D image. */
export function batchImage4D(x: Tensor): {tensor: Tensor4D, reshaped: boolean} {
  if (x.rank === 3) {
    return {tensor: reshape<Rank.R4>(x, batchedImageShape(x.shape, 4)), reshaped: true};
  }
  // The caller retains its operation-specific rank validation.
  return {tensor: x as Tensor4D, reshaped: false};
}

/** Inserts a batch axis only for an unbatched 3D image. */
export function batchImage5D(x: Tensor): {tensor: Tensor5D, reshaped: boolean} {
  if (x.rank === 4) {
    return {tensor: reshape<Rank.R5>(x, batchedImageShape(x.shape, 5)), reshaped: true};
  }
  // The caller retains its operation-specific rank validation.
  return {tensor: x as Tensor5D, reshaped: false};
}
