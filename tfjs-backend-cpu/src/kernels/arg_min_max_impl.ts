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

import {backend_util, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';
import {transpose} from './Transpose';

/** Reduces an axis to its first extremum index, retaining strict comparison and NaN semantics. */
export function argMinMaxImpl(
    x: TensorInfo, axis: number, backend: MathBackendCPU,
    findMax: boolean): TensorInfo {
  const opName = findMax ? 'argMax' : 'argMin';
  assertNotComplex(x, opName);

  let axes = util.parseAxisParam(axis, x.shape);
  const permutedAxes = backend_util.getAxesPermutation(axes, x.shape.length);
  let $x = x;
  const intermediateTensorInfos: TensorInfo[] = [];
  if (permutedAxes != null) {
    $x = transpose({inputs: {x}, backend, attrs: {perm: permutedAxes}});
    intermediateTensorInfos.push($x);
    axes = backend_util.getInnerMostAxes(axes.length, $x.shape.length);
  }

  axes = [axes[0]];
  backend_util.assertAxesAreInnerMostDims(opName, axes, $x.shape.length);
  const [outShape, reduceShape] =
      backend_util.computeOutAndReduceShapes($x.shape, axes);

  const outSize = util.sizeFromShape(outShape);
  const vals = util.makeZerosTypedArray(outSize, 'int32');
  const reduceSize = util.sizeFromShape(reduceShape);

  const aVals = backend.requireValues($x.dataId) as TypedArray;
  for (let i = 0; i < vals.length; ++i) {
    const offset = i * reduceSize;
    let best = aVals[offset] ?? NaN;
    let bestIndex = 0;
    for (let j = 0; j < reduceSize; ++j) {
      const value = aVals[offset + j] ?? NaN;
      if (findMax ? value > best : value < best) {
        best = value;
        bestIndex = j;
      }
    }
    vals[i] = bestIndex;
  }

  intermediateTensorInfos.forEach(
      t => backend.disposeIntermediateTensorInfo(t));

  return backend.makeTensorInfo(outShape, 'int32', vals);
}
