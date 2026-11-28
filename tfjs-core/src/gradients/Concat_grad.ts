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
import {Concat, ConcatAttrs} from '../kernel_names';
import {GradConfig, KernelAttributes} from '../kernel_registry';
import {split} from '../ops/split';
import {Tensor} from '../tensor';
import {parseAxisParam, requiredNumericEntry} from '../util';
import {requiredKernelEntry} from '../ops/shape_dimension';

export const concatGradConfig: GradConfig = {
  kernelName: Concat,
  saveAllInputs: true,
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: KernelAttributes<ConcatAttrs>) => {
    const shapes = saved.map(t => t.shape);
    const {axis} = attrs;
    const first = requiredKernelEntry(saved, 0);
    const $axis = requiredNumericEntry(parseAxisParam(axis, first.shape), 0);
    const sizeSplits = shapes.map(s => requiredNumericEntry(s, $axis));
    const derTensors = split(dy, sizeSplits, $axis);
    return derTensors.map(t => () => t) as {};
  }
};
