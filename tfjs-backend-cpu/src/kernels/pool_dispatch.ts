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

import {AvgPoolAttrs, AvgPool3DAttrs, backend_util, KernelAttributes, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';
import {pool, pool3d} from '../utils/pool_utils';
import {identity} from './Identity';

type PoolType = 'avg'|'max';

export function dispatchPool2D(args: {
  inputs: {x: TensorInfo}, backend: MathBackendCPU,
  attrs: KernelAttributes<AvgPoolAttrs>
}, poolType: PoolType): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  assertNotComplex(x, `${poolType}Pool`);
  const {filterSize, strides, pad, dimRoundingMode} = attrs;
  const dilations = 1;

  util.assert(
      backend_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () => `Error in ${poolType}Pool: Either strides or dilations must be 1. ` +
          `Got strides ${strides} and dilations '${dilations}'`);

  const convInfo = backend_util.computePool2DInfo(
      x.shape as [number, number, number, number], filterSize, strides,
      dilations, pad, dimRoundingMode);
  let res: TensorInfo;

  if (convInfo.filterWidth === 1 && convInfo.filterHeight === 1 &&
      util.arraysEqual(convInfo.inShape, convInfo.outShape)) {
    res = identity({inputs: {x}, backend});
  } else {
    const xValues = backend.requireValues(x.dataId) as TypedArray;
    const strides = util.computeStrides(x.shape);
    const buffer = pool(xValues, x.shape, x.dtype, strides, convInfo, poolType);
    res = backend.makeTensorInfo(
        convInfo.outShape, x.dtype, buffer.values as TypedArray);
  }
  return res;
}

export function dispatchPool3D(args: {
  inputs: {x: TensorInfo}, backend: MathBackendCPU,
  attrs: KernelAttributes<AvgPool3DAttrs>
}, poolType: PoolType): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {filterSize, strides, pad, dimRoundingMode, dataFormat} = attrs;

  assertNotComplex(x, `${poolType}Pool3d`);

  const convInfo = backend_util.computePool3DInfo(
      x.shape as [number, number, number, number, number], filterSize, strides,
      1 /* dilations */, pad, dimRoundingMode, dataFormat);

  const xValues = backend.requireValues(x.dataId) as TypedArray;
  const outBuf = pool3d(
      xValues, x.shape, x.dtype, util.computeStrides(x.shape), convInfo, poolType);

  return backend.makeTensorInfo(outBuf.shape, 'float32', outBuf.values);
}
