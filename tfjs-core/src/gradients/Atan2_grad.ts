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

import {Atan2} from '../kernel_names';
import {GradConfig} from '../kernel_registry';
import {add} from '../ops/add';
import {div} from '../ops/div';
import {mul} from '../ops/mul';
import {neg} from '../ops/neg';
import {reshape} from '../ops/reshape';
import {square} from '../ops/square';
import {Tensor} from '../tensor';
import {sumBroadcastGradient, savedBroadcastOperands} from './broadcast_grad_util';

export const atan2GradConfig: GradConfig = {
  kernelName: Atan2,
  inputsToSave: ['a', 'b'],
  gradFunc: (dy: Tensor, saved: Tensor[]) => {
    const {a, b, outShape} = savedBroadcastOperands(saved);

    const derA = () => {
      const d = add(square(a), square(b));
      const res = mul(dy, div(b, d));
      const reduced = sumBroadcastGradient(res, a.shape, outShape);
      return reshape(reduced, a.shape);
    };
    const derB = () => {
      const d = add(square(a), square(b));
      const res = neg(mul(dy, div(a, d)));
      const reduced = sumBroadcastGradient(res, b.shape, outShape);
      return reshape(reduced, b.shape);
    };
    return {a: derA, b: derB};
  }
};
