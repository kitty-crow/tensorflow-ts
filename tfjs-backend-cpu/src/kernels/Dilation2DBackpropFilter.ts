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

import {backend_util, Dilation2DAttrs, Dilation2DBackpropFilter, Tensor3D, Tensor4D, TypedArray, util} from '@tensorflow/tfjs-core';
import {KernelAttributes, KernelConfig} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {requiredCpuEntry} from '../cpu_util';
import {accumulateDilationGradient, dilation2DWindow, dilationFilterRow, dilationInputRow} from './Dilation2D_impl';

export const dilation2DBackpropFilterConfig: KernelConfig = {
  kernelName: Dilation2DBackpropFilter,
  backendName: 'cpu',
  kernelFunc: ({inputs, backend, attrs}) => {
    const {x, filter, dy} =
        inputs as {x: Tensor4D, filter: Tensor3D, dy: Tensor4D};
    const {strides, pad, dilations} = attrs as KernelAttributes<Dilation2DAttrs>;
    const cpuBackend = backend as MathBackendCPU;

    const $x =
        util.toNestedArray(
            x.shape, cpuBackend.requireValues(x.dataId) as TypedArray) as
        number[][][][];

    const $filter = util.toNestedArray(
                        filter.shape,
                        cpuBackend.requireValues(filter.dataId) as
                            TypedArray) as number[][][];

    const info = backend_util.computeDilation2DInfo(
        x.shape as [number, number, number, number],
        filter.shape as [number, number, number], strides, pad,
        'NHWC' /* dataFormat */, dilations);

    const {
      batchSize, inChannels, outHeight, outWidth, padInfo,
      strideHeight, strideWidth, outShape
    } = info;

    util.assert(
        dy.rank === outShape.length,
        () => `Error in ${Dilation2DBackpropFilter}, dy ` +
            `must have the same rank as output ${outShape.length}, but got ` +
            `${dy.rank}`);

    const $dy =
        util.toNestedArray(
            outShape, cpuBackend.requireValues(dy.dataId) as TypedArray) as
        number[][][][];

    // The computed filter gradients has the same dimensions as the filter:
    // [filterHeight, filterWidth, depth]
    const gradients = util.makeZerosNestedTypedArray(
                          filter.shape, filter.dtype) as number[][][];

    // Strict comparison retains the first maximum in traversal order.
    // This implementation follows the TF c++ implementation:
    // https://github.com/tensorflow/tensorflow/blob/d9a3a849edc198e90172bc58eb293de457f9d986/tensorflow/core/kernels/dilation_ops.cc
    for (let b = 0; b < batchSize; ++b) {
      for (let hOut = 0; hOut < outHeight; ++hOut) {
        const hBeg = hOut * strideHeight - padInfo.top;
        for (let wOut = 0; wOut < outWidth; ++wOut) {
          const wBeg = wOut * strideWidth - padInfo.left;
          for (let d = 0; d < inChannels; ++d) {
            const maximum = dilation2DWindow(info, hBeg, wBeg,
                (hIn, wIn, h, w) =>
                    requiredCpuEntry(dilationInputRow($x, b, hIn, wIn), d) +
                    requiredCpuEntry(dilationFilterRow($filter, h, w), d));
            const gradRow = dilationFilterRow(gradients, maximum.h, maximum.w);
            accumulateDilationGradient(gradRow, d,
                requiredCpuEntry(dilationInputRow($dy, b, hOut, wOut), d));
          }
        }
      }
    }

    const dataId = cpuBackend.write(
        util.toTypedArray(gradients, x.dtype), filter.shape, filter.dtype);

    return {dataId, shape: filter.shape, dtype: filter.dtype};
  }
};
