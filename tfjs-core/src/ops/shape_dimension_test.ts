/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

import * as axis from './axis_util';
import * as broadcast from './broadcast_util';
import * as concat from './concat_util';
import * as slice from './slice_util';

describe('checked shape utility regressions', () => {
  it('interleaves coordinates including zero', () => {
    expect(axis.combineLocations([0, 2], [3, 0], [1, 3]))
        .toEqual([0, 3, 2, 0]);
  });
  it('preserves reduction axis order and zero dimensions', () => {
    expect(axis.computeOutAndReduceShapes([0, 2, 3], [2, 0]))
        .toEqual([[2], [3, 0]]);
    expect(() => axis.computeOutAndReduceShapes([2], [1])).toThrowError();
  });
  it('inverts a permutation without mutating it', () => {
    const permutation = [2, 0, 1, 3];
    expect(axis.getUndoAxesPermutation(permutation)).toEqual([1, 2, 0, 3]);
    expect(permutation).toEqual([2, 0, 1, 3]);
  });
  it('concatenates zero-length dimensions without mutating shapes', () => {
    const shapes = [[0, 3], [2, 3], [4, 3]];
    expect(() => concat.assertParamsConsistent(shapes, 0)).not.toThrow();
    expect(concat.computeOutShape(shapes, 0)).toEqual([6, 3]);
    expect(shapes).toEqual([[0, 3], [2, 3], [4, 3]]);
  });
  it('rejects missing concatenation shapes or dimensions', () => {
    expect(() => concat.computeOutShape([], 0)).toThrowError();
    expect(() => concat.computeOutShape([[2], []], 0)).toThrowError();
    expect(() => concat.assertParamsConsistent([[2, 3], [2, 4]], 0))
        .toThrowError();
  });
  it('preserves the historical broadcast dimension fallback', () => {
    expect(broadcast.getBroadcastDims([0, 1], [3, 4])).toEqual([0, 1]);
    expect(broadcast.getBroadcastDims([1], [0])).toEqual([]);
    expect(broadcast.getBroadcastDims([1, 3], [3])).toEqual([]);
  });
  it('calculates strided shapes and flat offsets', () => {
    expect(slice.computeOutShape([0, 1], [5, 7], [2, 3])).toEqual([3, 2]);
    expect(slice.computeOutShape([4], [0], [-2])).toEqual([2]);
    expect(slice.computeFlatOffset([1, 2, 3], [12, 4])).toBe(23);
    expect(slice.computeFlatOffset([], [])).toBe(1);
  });
  it('preserves contiguous slice detection', () => {
    expect(slice.isSliceContinous([5, 3], [1, 0], [2, 3])).toBe(true);
    expect(slice.isSliceContinous([5, 3], [0, 1], [5, 2])).toBe(false);
    expect(slice.isSliceContinous([], [], [])).toBe(true);
  });
  it('rejects missing required slice coordinates', () => {
    expect(() => slice.computeOutShape([0], [], [1])).toThrowError();
    expect(() => slice.computeFlatOffset([1, 2], [])).toThrowError();
  });
});
