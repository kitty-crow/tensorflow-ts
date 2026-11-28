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

import {buffer, KernelConfig, KernelAttributes, Rank, slice_util, StridedSlice, StridedSliceAttrs, StridedSliceInputs, TensorBuffer, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {stridedSliceImplCPU} from '../kernel_utils/shared';
import {StridedSliceProgram} from '../strided_slice_gpu';

import {reshape} from './Reshape';
import {slice} from './Slice';

export function stridedSlice(args: {
  inputs: StridedSliceInputs,
  backend: MathBackendWebGL,
  attrs: KernelAttributes<StridedSliceAttrs>
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {
    begin,
    end,
    strides,
    beginMask,
    endMask,
    ellipsisMask,
    newAxisMask,
    shrinkAxisMask
  } = attrs;

  const {
    finalShapeSparse,
    finalShape,
    isIdentity,
    sliceDim0,
    isSimpleSlice,
    begin: $begin,
    end: $end,
    strides: $strides
  } =
      slice_util.sliceInfo(
          x.shape, begin, end, strides, beginMask, endMask, ellipsisMask,
          newAxisMask, shrinkAxisMask);

  const result = sliceWithShape(x, backend, {
    finalShapeSparse, finalShape, isIdentity, sliceDim0, isSimpleSlice,
    begin: $begin, end: $end, strides: $strides
  });

  const resultReshaped =
      reshape({inputs: {x: result}, backend, attrs: {shape: finalShape}});

  backend.disposeIntermediateTensorInfo(result);

  return resultReshaped;
}

/** Retain the WebGL intermediate reshape and disposal contract. */
function sliceWithShape(
    x: TensorInfo, backend: MathBackendWebGL,
    info: {
      finalShapeSparse: number[], finalShape: number[], isIdentity: boolean,
      sliceDim0: boolean, isSimpleSlice: boolean, begin: number[],
      end: number[], strides: number[]
    }): TensorInfo {
  const {finalShapeSparse, finalShape, isIdentity, sliceDim0, isSimpleSlice,
    begin, end, strides} = info;
  if (isIdentity) {
    return reshape({inputs: {x}, backend, attrs: {shape: finalShape}});
  }
  if (sliceDim0 || isSimpleSlice) {
    util.assert(
        x.shape.length >= 1,
        () => `Input must have rank at least 1, got: ${x.shape.length}`);
    const size = slice_util.computeOutShape(begin, end, strides);
    const sliced = slice({inputs: {x}, backend, attrs: {begin, size}});
    const result = reshape({inputs: {x: sliced}, backend, attrs: {shape: finalShape}});
    backend.disposeIntermediateTensorInfo(sliced);
    return result;
  }
  if (backend.shouldExecuteOnCPU([x])) {
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const values = backend.readSync(x.dataId) as TypedArray;
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const xBuf = buffer(x.shape, x.dtype, values) as TensorBuffer<Rank>;
    const resultValues = stridedSliceImplCPU(finalShapeSparse, xBuf, strides, begin);
    return backend.makeTensorInfo(finalShape, x.dtype, resultValues.values);
  }
  const program = new StridedSliceProgram(begin, strides, finalShapeSparse);
  return backend.runWebGLProgram(program, [x], x.dtype);
}

export const stridedSliceConfig: KernelConfig = {
  kernelName: StridedSlice,
  backendName: 'webgl',
  kernelFunc: stridedSlice
};
