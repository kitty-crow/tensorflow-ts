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
import {cast} from '../ops/cast';
import {mul} from '../ops/mul';
import {Tensor} from '../tensor';
import {assertArrayLength} from '../util_base';

/** Keep each comparator explicit, including asymmetric tie and NaN behaviour. */
export function createExtremaGradient(
    compareA: (a: Tensor, b: Tensor) => Tensor,
    compareB: (a: Tensor, b: Tensor) => Tensor) {
  return (dy: Tensor, saved: Tensor[]) => {
    assertArrayLength(saved, 2);
    const [a, b] = saved;
    return {
      a: () => mul(dy, cast(compareA(a, b), 'float32')),
      b: () => mul(dy, cast(compareB(a, b), 'float32'))
    };
  };
}
