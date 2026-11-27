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

import {TensorInfo} from '../tensor_info';
import {sequentialConvolutionTestInputs, requiredTestElement, requiredTestEntry} from '../test_util';

import {prepareSplitSize} from './split_util';
import {checkPadOnDimRoundingMode} from './conv_util';

describe('conditional branch regression coverage', () => {
  const input: TensorInfo = {dataId: {}, shape: [2, 6], dtype: 'float32'};

  it('supports equal numeric splits on the selected axis', () => {
    expect(prepareSplitSize(input, 3, 1)).toEqual([2, 2, 2]);
    expect(() => prepareSplitSize(input, 4, 1)).toThrowError(
        'Number of splits must evenly divide the axis.');
  });
  it('preserves inferred split size and array identity', () => {
    const sizes = [2, -1, 1];
    expect(prepareSplitSize(input, sizes, 1)).toBe(sizes);
    expect(sizes).toEqual([2, 3, 1]);
  });
  it('rejects multiple inferred splits or mismatched totals', () => {
    expect(() => prepareSplitSize(input, [-1, -1], 1)).toThrowError();
    expect(() => prepareSplitSize(input, [1, 2], 1)).toThrowError();
  });
  it('checks padding only when a rounding mode is specified', () => {
    expect(() => checkPadOnDimRoundingMode('conv', 'same')).not.toThrow();
    expect(() => checkPadOnDimRoundingMode('conv', 'same', 'ceil'))
        .toThrowError();
    expect(() => checkPadOnDimRoundingMode('conv', 2, 'floor')).not.toThrow();
    expect(() => checkPadOnDimRoundingMode('conv', 0.5, 'floor'))
        .toThrowError();
  });
});


describe('checked split dimensions', () => {
  it('preserves a zero-sized axis and rejects missing dimensions', () => {
    const empty: TensorInfo = {dataId: {}, shape: [0, 3], dtype: 'float32'};
    expect(prepareSplitSize(empty, 2, 0)).toEqual([0, 0]);
    expect(() => prepareSplitSize(empty, 2, 2)).toThrowError(/Missing dimension/);
  });
});

describe('typed numeric and tensor test fixtures', () => {
  it('generates independent sequential arrays including empty inputs', () => {
    const result = sequentialConvolutionTestInputs(3, 2);
    expect(result.input).toEqual([1, 2, 3]);
    expect(result.filter).toEqual([1, 2]);
    result.input[0] = 9;
    expect(result.filter).toEqual([1, 2]);
    expect(sequentialConvolutionTestInputs(0, 0)).toEqual(
        {input: [], filter: []});
  });

  it('retains zero and rejects missing output entries', () => {
    expect(requiredTestElement([0], 0)).toBe(0);
    expect(requiredTestEntry({value: 0}, 'value')).toBe(0);
    expect(() => requiredTestElement([], 0)).toThrowError(/Missing/);
    expect(() => requiredTestEntry({value: 0}, 'missing')).toThrowError(/Missing/);
  });
});
