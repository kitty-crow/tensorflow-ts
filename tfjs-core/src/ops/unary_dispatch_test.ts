/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import {Asinh, Ceil, Identity, Imag, Real, Relu} from '../kernel_names';
import {Tensor} from '../tensor';
import {Rank} from '../types';

import {asinh} from './asinh';
import {ceil} from './ceil';
import {clone} from './clone';
import {imag} from './imag';
import {real} from './real';
import {relu} from './relu';

describe('unary operation dispatch contracts', () => {
  const float = new Tensor<Rank.R1>([1], 'float32', {}, 0);
  const integer = new Tensor<Rank.R1>([1], 'int32', {}, 1);
  const string = new Tensor<Rank.R1>([1], 'string', {}, 2);

  beforeEach(() => {
    spyOn(ENGINE, 'startScope');
    spyOn(ENGINE, 'endScope');
    spyOn(ENGINE, 'runKernel').and.returnValue(float);
  });

  it('retains operation identity, input name and scope for numeric kernels', () => {
    expect(asinh(integer)).toBe(float);
    expect(ENGINE.startScope).toHaveBeenCalledWith('asinh__op');
    expect(ENGINE.runKernel).toHaveBeenCalledWith(Asinh, {x: integer});
    expect(relu(integer)).toBe(float);
    expect(ENGINE.startScope).toHaveBeenCalledWith('relu__op');
    expect(ENGINE.runKernel).toHaveBeenCalledWith(Relu, {x: integer});
  });

  it('preserves the float32 requirement and failure scope', () => {
    expect(ceil(float)).toBe(float);
    expect(ENGINE.runKernel).toHaveBeenCalledWith(Ceil, {x: float});
    expect(() => ceil(integer)).toThrowError(
        "Argument 'x' passed to 'ceil' must be float32 tensor, but got int32 tensor");
    expect(ENGINE.endScope).toHaveBeenCalledWith(null);
  });

  it('allows string tensors for identity but not numeric operations', () => {
    expect(clone(string)).toBe(float);
    expect(ENGINE.runKernel).toHaveBeenCalledWith(Identity, {x: string});
    expect(() => relu(string)).toThrowError(
        "Argument 'x' passed to 'relu' must be numeric tensor, but got string tensor");
  });

  it('preserves the input key and diagnostic argument for complex components', () => {
    expect(imag(float)).toBe(float);
    expect(ENGINE.runKernel).toHaveBeenCalledWith(Imag, {input: float});
    expect(real(float)).toBe(float);
    expect(ENGINE.runKernel).toHaveBeenCalledWith(Real, {input: float});
    expect(() => imag(string)).toThrowError(
        "Argument 'input' passed to 'imag' must be numeric tensor, but got string tensor");
  });
});
