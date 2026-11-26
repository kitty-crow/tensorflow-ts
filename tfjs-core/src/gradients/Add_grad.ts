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
import {Add} from '../kernel_names';
import {GradConfig} from '../kernel_registry';
import * as broadcast_util from '../ops/broadcast_util';
import {reshape} from '../ops/reshape';
import {Tensor} from '../tensor';
import {assertArrayLength} from '../util_base';
import {sumBroadcastGradient} from './broadcast_grad_util';

export const addGradConfig: GradConfig = {
  kernelName: Add,
  inputsToSave: ['a', 'b'],
  gradFunc: (dy: Tensor, saved: Tensor[]) => {
    assertArrayLength(saved, 2);
    const [a, b] = saved;
    const outShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);

    const derA = () => {
      const res = dy;
      const reduced = sumBroadcastGradient(res, a.shape, outShape);
      return reshape(reduced, a.shape);
    };
    const derB = () => {
      const res = dy;
      const reduced = sumBroadcastGradient(res, b.shape, outShape);
      return reshape(reduced, b.shape);
    };

    return {a: derA, b: derB};
  }
};
