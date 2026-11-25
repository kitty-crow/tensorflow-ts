/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {requireComplexTensorInfos, requireTensorDataValues} from './tensor_data_values';

describe('CPU tensor storage payload', () => {
  it('preserves the numeric array identity, including zero and NaN', () => {
    const values = new Float32Array([0, NaN, -2]);
    expect(requireTensorDataValues({values, dtype: 'float32', refCount: 1})).toBe(values);
  });

  it('accepts empty numeric payloads', () => {
    const values = new Int32Array(0);
    expect(requireTensorDataValues({values, dtype: 'int32', refCount: 1})).toBe(values);
  });

  it('preserves encoded string arrays without conversion', () => {
    const values = [new Uint8Array([65]), new Uint8Array(0)];
    expect(requireTensorDataValues({values, dtype: 'string', refCount: 1})).toBe(values);
  });

  it('rejects a complex container without a direct payload', () => {
    expect(() => requireTensorDataValues({dtype: 'complex64', refCount: 1}))
        .toThrowError('CPU tensor data has no stored values.');
  });
});

describe('CPU complex tensor storage', () => {
  it('preserves the owned component records', () => {
    const real = {dataId: {}, shape: [1], dtype: 'float32' as const};
    const imag = {dataId: {}, shape: [1], dtype: 'float32' as const};
    const parts = {real, imag};
    expect(requireComplexTensorInfos({dtype: 'complex64', refCount: 1,
      complexTensorInfos: parts})).toBe(parts);
  });

  it('rejects missing component ownership', () => {
    expect(() => requireComplexTensorInfos({dtype: 'complex64', refCount: 1}))
        .toThrowError('CPU tensor data has no complex components.');
  });
});
