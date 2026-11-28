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

import {Mod} from '../kernel_names';
import {GradConfig} from '../kernel_registry';
import {div} from '../ops/div';
import {floor} from '../ops/floor';
import {mul} from '../ops/mul';
import {neg} from '../ops/neg';
import {Tensor} from '../tensor';
import {reduceBroadcastGradient, savedBroadcastOperands} from './broadcast_grad_util';

export const modGradConfig: GradConfig = {
  kernelName: Mod,
  inputsToSave: ['a', 'b'],
  gradFunc: (dy: Tensor, saved: Tensor[]) => {
    const {a, b, outShape} = savedBroadcastOperands(saved);

    const derA = () => {
      return reduceBroadcastGradient(dy, a.shape, outShape);
    };
    const derB = () => {
      const res = mul(dy, neg(floor(div(a, b))));
      return reduceBroadcastGradient(res, b.shape, outShape);
    };
    return {a: derA, b: derB};
  }
};
