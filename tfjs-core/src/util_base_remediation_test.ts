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

import {inferFromImplicitShape, squeezeShape, toNestedArray} from './util_base';

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
    expect(toNestedArray([], new Float32Array([0]))).toBe(0);
    expect(() => toNestedArray([], new Float32Array(0))).toThrowError();
  });
});
