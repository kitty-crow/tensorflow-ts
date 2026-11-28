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

// Standalone Jasmine specs need the same platform registration as core.
import './base_side_effects';

import {flatten} from './util';
import {TensorBuffer} from './tensor';
import {Rank} from './types';
import {env} from './environment';
import {getGradient, registerGradient, unregisterGradient} from './kernel_registry';
import {basename, concatenateTypedArrays, decodeWeights} from './io/io_utils';
import {requiredTestElement, requiredTestEntry, requiredTestValue} from './test_util';
import {requiredNamedEntry, assertArrayLength, inferFromImplicitShape, indexToLoc, locToIndex, roundUpToPow2, sizeFromShape, squeezeShape, sum, toNestedArray} from './util_base';

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

describe('checked coordinate conversion', () => {
  it('round trips non-square shapes and scalar coordinates', () => {
    expect(indexToLoc(17, 3, [12, 4])).toEqual([1, 1, 1]);
    expect(locToIndex([1, 1, 1], 3, [12, 4])).toBe(17);
    expect(indexToLoc(0, 0, [])).toEqual([]);
    expect(locToIndex([], 0, [])).toBe(0);
  });
  it('rejects missing stride data rather than producing NaN', () => {
    expect(() => indexToLoc(17, 3, [12])).toThrowError();
    expect(() => locToIndex([1, 1, 1], 3, [12])).toThrowError();
  });
});


describe('required runtime named entry', () => {
  it('preserves zero, false and object identity', () => {
    const object = {value: 1};
    expect(requiredNamedEntry({zero: 0}, 'zero')).toBe(0);
    expect(requiredNamedEntry({flag: false}, 'flag')).toBe(false);
    expect(requiredNamedEntry({object}, 'object')).toBe(object);
  });
  it('rejects absent or null entries', () => {
    expect(() => requiredNamedEntry({}, 'missing')).toThrowError(/Missing required entry/);
    expect(() => requiredNamedEntry({missing: null}, 'missing')).toThrowError(/Missing required entry/);
  });
});


describe('required test values', () => {
  it('retains zero, false and null, but rejects undefined', () => {
    expect(requiredTestValue(0, 'count')).toBe(0);
    expect(requiredTestValue(false, 'flag')).toBe(false);
    expect(requiredTestValue(null, 'nullable')).toBeNull();
    expect(() => requiredTestValue(undefined, 'missing')).toThrowError(/missing/);
  });
});


describe('flatten traversal', () => {
  it('preserves destination identity and depth-first ordering', () => {
    const result = [0];
    expect(flatten([1, [2, 3], 4], result)).toBe(result);
    expect(result).toEqual([0, 1, 2, 3, 4]);
    expect(flatten([[], [5]])).toEqual([5]);
  });

  it('flattens typed arrays or retains them when requested', () => {
    const values = new Float32Array([0, 2]);
    expect(flatten<number|Float32Array>([1, values])).toEqual([1, 0, 2]);
    const preserved = flatten<Float32Array>(values, [], true);
    expect(preserved.length).toBe(1);
    expect(requiredTestValue(preserved[0], 'preserved typed array')).toBe(values);
  });
});

describe('Registered platform typed utility contract', () => {
  it('preserves UTF-8 bytes, empty decoding and typed array detection', () => {
    const platform = env().platform;
    const bytes = platform.encode('é', 'utf-8');
    expect(Array.from(bytes)).toEqual([195, 169]);
    expect(platform.decode(bytes, 'utf-8')).toBe('é');
    expect(platform.decode(new Uint8Array(0), 'utf-8')).toBe('');
    expect(platform.isTypedArray(bytes)).toBe(true);
    expect(platform.isTypedArray([195, 169])).toBe(false);
  });
});

describe('weight byte and metadata validation', () => {
  it('copies exactly the bytes of mixed subviews including empty views', () => {
    const bytes = new Uint8Array([9, 1, 2, 8]);
    const ints = new Int32Array([7, 0x01020304, 9]);
    const floats = new Float32Array([7, -0, 9]);
    const output = concatenateTypedArrays([
      bytes.subarray(1, 3), ints.subarray(1, 2),
      floats.subarray(1, 2), bytes.subarray(0, 0)
    ]);
    const expected = [
      1, 2, ...Array.from(new Uint8Array(ints.buffer, 4, 4)),
      ...Array.from(new Uint8Array(floats.buffer, 4, 4))
    ];
    expect(Array.from(new Uint8Array(output))).toEqual(expected);
    expect(Array.from(bytes)).toEqual([9, 1, 2, 8]);
    expect(concatenateTypedArrays([]).byteLength).toBe(0);
  });

  it('rejects missing or undefined integer quantisation metadata', () => {
    const bytes = new Uint8Array([1, 0]).buffer;
    expect(() => decodeWeights(bytes, [{
      name: 'missingMin', shape: [1], dtype: 'float32',
      quantization: {dtype: 'uint8', scale: 0}
    }])).toThrowError(/min and scale/);
    expect(() => decodeWeights(bytes, [{
      name: 'missingScale', shape: [1], dtype: 'int32',
      quantization: {dtype: 'uint16', min: 0, scale: undefined}
    }])).toThrowError(/min and scale/);
  });
});


describe('weight framing and basename regressions', () => {
  it('rejects a missing string length prefix before decoding a tensor', () => {
    expect(() => decodeWeights(new ArrayBuffer(0), [{
      name: 'missingLength', shape: [1], dtype: 'string'
    }])).toThrowError();
  });
  it('retains path trimming, root and trailing separator behavior', () => {
    expect(basename(' /a/b/// ')).toBe('b');
    expect(basename('/')).toBe('');
    expect(basename('')).toBe('');
    expect(basename('plain')).toBe('plain');
  });
});

describe('required numeric reductions', () => {
  it('retains zero, empty products and NaN values', () => {
    expect(sum([0, -2, 2])).toBe(0);
    expect(sum([])).toBe(0);
    expect(sum([NaN])).toBeNaN();
    expect(sizeFromShape([])).toBe(1);
    expect(sizeFromShape([2, 0, 3])).toBe(0);
    expect(sizeFromShape([2, 3, 4])).toBe(24);
  });
  it('rejects missing concrete coordinates', () => {
    expect(() => sum(new Array<number>(1))).toThrowError();
    expect(() => sizeFromShape(new Array<number>(1))).toThrowError();
    expect(() => inferFromImplicitShape(new Array<number>(1), 1))
        .toThrowError();
  });
});

describe('required nested test fixtures', () => {
  it('preserves present elements at every level', () => {
    const rows = [[0, 2], [3, 4]];
    expect(requiredTestElement(requiredTestElement(rows, 0), 0)).toBe(0);
    expect(requiredTestElement(requiredTestElement(rows, 1), 1)).toBe(4);
    const signatures = {default: {inputs: {x: {name: 'x:0'}}}};
    const signature = requiredTestEntry(signatures, 'default');
    expect(requiredTestEntry(signature.inputs, 'x').name).toBe('x:0');
  });
  it('rejects missing nested elements and named entries', () => {
    const rows: number[][] = [[1], []];
    expect(() => requiredTestElement(requiredTestElement(rows, 1), 0))
        .toThrowError();
    expect(() => requiredTestEntry({x: {name: 'x:0'}}, 'missing'))
        .toThrowError();
  });
});

describe('TensorBuffer coordinate contracts', () => {
  it('preserves scalar, zero, NaN and empty string values', () => {
    const scalar = new TensorBuffer<Rank.R0, 'float32'>(
        [], 'float32', new Float32Array([7]));
    expect(scalar.get()).toBe(7);
    expect(scalar.get(0)).toBe(7);
    const matrix = new TensorBuffer<Rank.R2, 'float32'>(
        [2, 2], 'float32', new Float32Array([0, NaN, 2, 3]));
    expect(matrix.get(0, 0)).toBe(0);
    expect(Number.isNaN(matrix.get(0, 1))).toBe(true);
    expect(matrix.locToIndex([1, 1])).toBe(3);
    expect(matrix.get(1, 1)).toBe(3);
    const strings = new TensorBuffer<Rank.R1, 'string'>([1], 'string', ['']);
    expect(strings.get(0)).toBe('');
  });

  it('rejects missing coordinates and inaccessible buffer entries', () => {
    const matrix = new TensorBuffer<Rank.R2, 'float32'>(
        [2, 2], 'float32', new Float32Array(4));
    expect(() => matrix.get(2, 0)).toThrowError();
    expect(() => matrix.get(0, 0, 0)).toThrowError();
    expect(() => matrix.get(0, 0.5)).toThrowError();
    expect(() => matrix.locToIndex([])).toThrowError();
  });
});

describe('gradient registry conditional warning', () => {
  it('consults DEBUG only for an existing gradient and still replaces it', () => {
    const kernelName = 'RemediationGradientDebugCheck';
    const initial = {kernelName, gradFunc: () => null};
    const replacement = {kernelName, gradFunc: () => ({})};
    const debug = spyOn(env(), 'getBool').and.returnValue(false);
    registerGradient(initial);
    try {
      expect(debug).not.toHaveBeenCalled();
      registerGradient(replacement);
      expect(debug).toHaveBeenCalledTimes(1);
      expect(debug).toHaveBeenCalledWith('DEBUG');
      expect(getGradient(kernelName)).toBe(replacement);
    } finally {
      unregisterGradient(kernelName);
    }
  });
});
