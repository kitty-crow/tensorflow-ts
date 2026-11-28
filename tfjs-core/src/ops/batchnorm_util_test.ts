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
import {assertBatchNormInputRank, getBatchNormReshapeShape} from './batchnorm_util';

describe('batchnorm_util.getBatchNormReshapeShape', () => {
  it('uses size for scalar and vector inputs', () => {
    expect(getBatchNormReshapeShape([], 1)).toEqual([1, 1, 1, 1]);
    expect(getBatchNormReshapeShape([5], 5)).toEqual([1, 1, 1, 5]);
    expect(getBatchNormReshapeShape([0], 0)).toEqual([1, 1, 1, 0]);
  });

  it('prepends two singleton axes for rank two', () => {
    const shape = [2, 3];
    expect(getBatchNormReshapeShape(shape, 6)).toEqual([1, 1, 2, 3]);
    expect(shape).toEqual([2, 3]);
    expect(getBatchNormReshapeShape([0, 3], 0)).toEqual([1, 1, 0, 3]);
  });

  it('prepends one singleton axis for rank three', () => {
    expect(getBatchNormReshapeShape([2, 3, 4], 24)).toEqual([1, 2, 3, 4]);
    expect(getBatchNormReshapeShape([2, 0, 4], 0)).toEqual([1, 2, 0, 4]);
  });

  it('leaves rank four and higher on the no-reshape path', () => {
    expect(getBatchNormReshapeShape([1, 2, 3, 4], 24)).toBeNull();
    expect(getBatchNormReshapeShape([1, 2, 3, 4, 5], 120)).toBeNull();
    expect(getBatchNormReshapeShape([1, 2, 3, 4, 5, 6], 720)).toBeNull();
  });

  it('rejects missing dimensions rather than passing an incomplete shape', () => {
    const missingSecond = new Array<number>(2);
    missingSecond[0] = 2;
    const missingThird = new Array<number>(3);
    missingThird[0] = 2;
    missingThird[1] = 3;
    const message = 'Batch normalization input shape is missing a dimension.';
    expect(() => getBatchNormReshapeShape(missingSecond, 0))
        .toThrowError(message);
    expect(() => getBatchNormReshapeShape(missingThird, 0))
        .toThrowError(message);
  });
});

describe('rank-specific batch normalization validation', () => {
  it('accepts vectors for parameters but requires the exact input rank', () => {
    for (const rank of [2, 3, 4]) {
      expect(() => assertBatchNormInputRank(rank, 'x', rank)).not.toThrow();
      expect(() => assertBatchNormInputRank(rank, 'mean', 1)).not.toThrow();
      expect(() => assertBatchNormInputRank(rank, 'variance', rank)).not.toThrow();
      expect(() => assertBatchNormInputRank(rank, 'scale', 1)).not.toThrow();
      expect(() => assertBatchNormInputRank(rank, 'offset', rank)).not.toThrow();
      expect(() => assertBatchNormInputRank(rank, 'x', 1)).toThrowError(
          `Error in batchNorm${rank}D: x must be rank ${rank} but got rank 1.`);
      expect(() => assertBatchNormInputRank(rank, 'scale', 0)).toThrowError(
          `Error in batchNorm${rank}D: scale must be rank ${rank} or rank 1 but got rank 0.`);
    }
  });
});
