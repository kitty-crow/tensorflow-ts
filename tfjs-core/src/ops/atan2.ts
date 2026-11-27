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
import {Atan2, Atan2Inputs} from '../kernel_names';
import {Tensor} from '../tensor';
import {TensorLike} from '../types';

import {matchedBinaryInputs, op} from './operation';

/**
 * Computes arctangent of `tf.Tensor`s a / b element-wise: `atan2(a, b)`.
 * Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([1.0, 1.0, -1.0, .7]);
 * const b = tf.tensor1d([2.0, 13.0, 3.5, .21]);
 *
 * tf.atan2(a, b).print()
 * ```
 *
 * @param a The first tensor.
 * @param b The second tensor. Must have the same dtype as `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function atan2_<T extends Tensor>(
    a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  const {a: $a, b: $b} = matchedBinaryInputs(a, b, 'atan2');

  const inputs = {a: $a, b: $b} satisfies Atan2Inputs;

  return ENGINE.runKernel(Atan2, inputs);
}

export const atan2 = /* @__PURE__ */ op({atan2_});
