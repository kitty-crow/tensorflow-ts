/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

// Generates WGSL that computes strides.
export function symbolicallyComputeStrides(
    indicesArr: number[], variableName: string): string[] {
  if (Math.max(...indicesArr) > 5) {
    throw new Error('Cannot symbolically compute strides for rank > 6 tensor.');
  }

  const numCoords = indicesArr.length;
  const indicesStr = 'xyzwuv';
  if (numCoords === 1) {
    return [];
  }
  const shape = indicesArr.map(d => {
    const field = indicesStr[d];
    if (field == null) {
      throw new Error(`Invalid symbolic stride axis ${d}.`);
    }
    return `${variableName}.${field}`;
  });
  const strides: string[] = new Array<string>(numCoords - 1);
  let stride = shape[numCoords - 1];
  if (stride == null) {
    throw new Error('Missing final symbolic stride dimension.');
  }
  strides[numCoords - 2] = stride;
  for (let i = numCoords - 3; i >= 0; --i) {
    const dimension = shape[i + 1];
    if (dimension == null) {
      throw new Error(`Missing symbolic stride dimension ${i + 1}.`);
    }
    stride = `(${stride} * ${dimension})`;
    strides[i] = stride;
  }

  return strides;
}

export const atomicAddSnippet =
    (ptr: string, v: string, type: 'int32'|'float32') => {
      if (type === 'int32') {
        return `atomicAdd(${ptr}, bitcast<i32>(${v}));`;
      } else {
        // atomicAdd only supports uint/int type. For float, we use
        // atomicCompareExchangeWeak to simulate.
        return `
          {
            var oldValue = 0;
            loop {
              let newValueF32 = bitcast<f32>(oldValue) + (${v});
              let newValue = bitcast<i32>(newValueF32);
              let res = atomicCompareExchangeWeak(${ptr}, oldValue, newValue);
              if res.exchanged {
                break;
              }
              oldValue = res.old_value;
            }
          }`;
      }
    };

/** Read a required host-side shader entry, retaining zero and empty strings. */
export function shaderEntry<T>(entries: readonly T[], index: number): T {
  const entry = entries[index];
  if (entry == null) {
    throw new Error(`Missing shader entry at index ${index}.`);
  }
  return entry;
}
