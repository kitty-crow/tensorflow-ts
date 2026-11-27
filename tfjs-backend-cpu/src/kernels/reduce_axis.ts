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
import {reshape} from './Reshape';
import {transpose} from './Transpose';
import {reduceValues} from './reduce_values';

export function reduceAxis(
    x: TensorInfo, backend: MathBackendCPU, axis: number|number[],
    keepDims: boolean, mode: 'all'|'any'|'min'): TensorInfo {
  assertNotComplex(x, mode);

  const origAxes = util.parseAxisParam(axis, x.shape);
  let axes = origAxes;
  const permutedAxes = backend_util.getAxesPermutation(axes, x.shape.length);
  let $x = x;
  if (permutedAxes != null) {
    $x = transpose({inputs: {x}, backend, attrs: {perm: permutedAxes}});
    axes = backend_util.getInnerMostAxes(axes.length, x.shape.length);
  }

  backend_util.assertAxesAreInnerMostDims(mode, axes, $x.shape.length);
  const [outShape, reduceShape] =
      backend_util.computeOutAndReduceShapes($x.shape, axes);
  const reduceSize = util.sizeFromShape(reduceShape);
  // This allocator rejects string dtype rather than returning string storage.
  const vals = util.makeZerosTypedArray(util.sizeFromShape(outShape), $x.dtype) as TypedArray;

  const aVals = backend.requireValues($x.dataId) as TypedArray;
  reduceValues(aVals, reduceSize, vals, mode);

  if (permutedAxes != null) {
    backend.disposeIntermediateTensorInfo($x);
  }

  const result = backend.makeTensorInfo(outShape, $x.dtype, vals);

  if (keepDims) {
    const expandedShape = backend_util.expandShapeToKeepDim(outShape, origAxes);
    const reshapedResult =
        reshape({inputs: {x: result}, backend, attrs: {shape: expandedShape}});

    backend.disposeIntermediateTensorInfo(result);

    return reshapedResult;
  }

  return result;
}
