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
import {Pow} from '../kernel_names';
import {GradConfig} from '../kernel_registry';
import * as broadcast_util from '../ops/broadcast_util';
import {cast} from '../ops/cast';
import {greater} from '../ops/greater';
import {log} from '../ops/log';
import {mul} from '../ops/mul';
import {pow} from '../ops/pow';
import {reshape} from '../ops/reshape';
import {scalar} from '../ops/scalar';
import {sub} from '../ops/sub';
import {where} from '../ops/where';
import {zerosLike} from '../ops/zeros_like';
import {Tensor} from '../tensor';
import {assertArrayLength} from '../util_base';
import {sumBroadcastGradient} from './broadcast_grad_util';

export const powGradConfig: GradConfig = {
  kernelName: Pow,
  inputsToSave: ['a', 'b'],
  outputsToSave: [true],
  gradFunc: (dy: Tensor, saved: Tensor[]) => {
    assertArrayLength(saved, 3);
    const [a, b, y] = saved;
    const base = a;
    const exp = b;
    const outShape =
        broadcast_util.assertAndGetBroadcastShape(base.shape, exp.shape);

    const derBase = () => {
      const expFloat = cast(exp, 'float32');
      const res = mul(dy, mul(expFloat, pow(base, sub(expFloat, scalar(1)))));
      const reduced = sumBroadcastGradient(res, base.shape, outShape);
      return reshape(reduced, base.shape);
    };
    const derExp = () => {
      const condition = greater(base, 0);
      const logBase = where(condition, log(base), zerosLike(base));
      const res = mul(dy, mul(y, logBase));
      const reduced = sumBroadcastGradient(res, exp.shape, outShape);
      return reshape(reduced, exp.shape);
    };
    return {a: derBase, b: derExp};
  }
};
