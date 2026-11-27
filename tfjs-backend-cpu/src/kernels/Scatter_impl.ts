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
import {buffer, Rank, ShapeMap, TensorBuffer, TypedArray, util} from '@tensorflow/tfjs-core';

interface DefaultValueTypeMap {
  bool: boolean;
  int32: number;
  float32: number;
  string: string;
}

export function
scatterImpl<R extends Rank, D extends 'float32'|'int32'|'bool'|'string'>(
    indices: TensorBuffer<R, 'int32'>, updates: TensorBuffer<R, D>,
    shape: number[], outputSize: number, sliceSize: number, numUpdates: number,
    sliceRank: number, strides: number[],
    defaultValue: TensorBuffer<R, D>|DefaultValueTypeMap[D],
    sumDupeIndices: boolean): TensorBuffer<R, D> {
  const flattenShape = [outputSize / sliceSize, sliceSize];

  const indicesData = indices.values as TypedArray;
  const updatesData = updates.values;

  if (outputSize === 0) {
    return buffer(shape as ShapeMap[R], updates.dtype);
  }

  const outBuf = (defaultValue instanceof TensorBuffer) ?
      defaultValue :
      buffer(flattenShape, updates.dtype);
  if (typeof defaultValue === 'string') {
    (outBuf.values as string[]).fill(defaultValue);
  }
  if (typeof defaultValue === 'number') {
    (outBuf.values as TypedArray).fill(defaultValue);
  }
  if (typeof defaultValue === 'boolean') {
    (outBuf.values as TypedArray).fill(+defaultValue);
  }

  for (let i = 0; i < numUpdates; i++) {
    const index: number[] = [];
    let flattenIndex = 0;
    for (let j = 0; j < sliceRank; j++) {
      const dim = util.requiredNumericEntry(indicesData, i * sliceRank + j);
      index.push(dim);
      flattenIndex += dim * util.requiredNumericEntry(strides, j);
    }

    if (flattenIndex < 0 || flattenIndex >= outputSize / sliceSize) {
      throw new Error(`Invalid indices: ${index} does not index into ${shape}`);
    }

    for (let k = 0; k < sliceSize; k++) {
      const outputIndex = flattenIndex * sliceSize + k;
      if (sumDupeIndices) {
        const outputValues = outBuf.values as TypedArray;
        outputValues[outputIndex] =
            util.requiredNumericEntry(outputValues, outputIndex) +
            util.requiredNumericEntry(updatesData as TypedArray, i * sliceSize + k);
      } else {
        const updateIndex = updates.rank === 0 ? 0 : i * sliceSize + k;
        const value = updatesData[updateIndex];
        if (value == null) {
          throw new Error(`Missing scatter update at index ${updateIndex}.`);
        }
        outBuf.values[outputIndex] = value;
      }
    }
  }

  return outBuf as TensorBuffer<R, D>;
}
