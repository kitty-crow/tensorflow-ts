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

import {requiredTestEntry} from './test_util';
import {assertArrayLength, inferFromImplicitShape, roundUpToPow2, squeezeShape, toNestedArray} from './util_base';

describe('shape utility regression coverage', () => {
  it('infers a missing dimension without mutating the input', () => {
    const shape = [2, -1, 3];
    expect(inferFromImplicitShape(shape, 24)).toEqual([2, 4, 3]);
    expect(shape).toEqual([2, -1, 3]);
  });
  it('preserves explicit shape identity and zero-size behaviour', () => {
    const shape = [2, 3];
    expect(inferFromImplicitShape(shape, 6)).toBe(shape);
    expect(inferFromImplicitShape(shape, 0)).toBe(shape);
  });
  it('rejects invalid implicit shapes', () => {
    expect(() => inferFromImplicitShape([-1, -1], 6)).toThrowError();
    expect(() => inferFromImplicitShape([-2], 6)).toThrowError();
    expect(() => inferFromImplicitShape([2, 3], 5)).toThrowError();
    expect(() => inferFromImplicitShape([0, -1], 0)).toThrowError();
    expect(() => inferFromImplicitShape([2, -1], 5)).toThrowError();
  });
  it('keeps unselected singleton dimensions', () => {
    expect(squeezeShape([1, 2, 1, 1], [2])).toEqual({
      newShape: [1, 2, 1], keptDims: [0, 1, 3]
    });
    expect(squeezeShape([1, 2, 1], [])).toEqual({
      newShape: [2], keptDims: [1]
    });
    expect(() => squeezeShape([1, 2], [1])).toThrowError();
  });
  it('preserves scalar zero and rejects missing scalar values', () => {
    expect(toNestedArray([], new Float32Array([0])) === 0).toBe(true);
    expect(() => toNestedArray([], new Float32Array(0))).toThrowError();
  });
});

describe('saved tensor array contract', () => {
  it('retains valid references and order', () => {
    const first = {id: 1};
    const second = {id: 2};
    const saved = [first, second];
    assertArrayLength(saved, 2);
    const [a, b] = saved;
    expect(a).toBe(first);
    expect(b).toBe(second);
  });
  it('rejects too few, too many and sparse entries', () => {
    expect(() => assertArrayLength([1], 2)).toThrowError();
    expect(() => assertArrayLength([1, 2, 3], 2)).toThrowError();
    expect(() => assertArrayLength(new Array<number>(2), 2)).toThrowError();
  });
});

describe('bitonic TopK size', () => {
  it('retains the shared power-of-two rounding contract', () => {
    expect([0, 1, 2, 3, 8, 9, 1025].map(roundUpToPow2))
        .toEqual([1, 1, 2, 4, 8, 16, 2048]);
  });
});

describe('required named test fixture', () => {
  it('preserves present values including falsy and null values', () => {
    const values: {[key: string]: number|string|boolean|null} = {
      zero: 0, empty: '', flag: false, absent: null
    };
    expect(requiredTestEntry(values, 'zero')).toBe(0);
    expect(requiredTestEntry(values, 'empty')).toBe('');
    expect(requiredTestEntry(values, 'flag')).toBe(false);
    expect(requiredTestEntry(values, 'absent')).toBeNull();
  });
  it('fails at the missing key instead of passing an absent value', () => {
    expect(() => requiredTestEntry({present: 1}, 'missing'))
        .toThrowError('Missing required test entry: missing.');
  });
  it('rejects an explicitly undefined fixture entry', () => {
    const values: {[key: string]: number|undefined} = {present: undefined};
    expect(() => requiredTestEntry(values, 'present'))
        .toThrowError('Missing required test entry: present.');
  });
});
