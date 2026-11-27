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

import {backend_util, Conv3D, Conv3DAttrs, Conv3DInputs, KernelConfig, KernelAttributes, TensorBuffer, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex, rankFiveStrides, requiredCpuEntry} from '../cpu_util';

export function conv3D(
    args: {inputs: Conv3DInputs, backend: MathBackendCPU, attrs: KernelAttributes<Conv3DAttrs>}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, filter} = inputs;
  const {strides, pad, dilations} = attrs;

  assertNotComplex([x, filter], 'conv3d');

  const convInfo = backend_util.computeConv3DInfo(
      x.shape as [number, number, number, number, number],
      filter.shape as [number, number, number, number, number], strides,
      dilations, pad);

  const {
    filterDepth,
    filterHeight,
    filterWidth,
    dilationDepth,
    dilationHeight,
    dilationWidth,
    padInfo
  } = convInfo;
  const padFront = padInfo.front;
  const padLeft = padInfo.left;
  const padTop = padInfo.top;
  const y = new TensorBuffer(convInfo.outShape, x.dtype as 'float32');

  const xVals = backend.requireValues(x.dataId) as TypedArray;
  const wVals = backend.requireValues(filter.dataId) as TypedArray;
  const yVals = y.values;

  const xStrides = util.computeStrides(x.shape);
  const filterStrides = util.computeStrides(filter.shape);
  const [xBatchStride, xDepthStride, xRowStride] = rankFiveStrides(xStrides);
  const [yBatchStride, yDepthStride, yRowStride] = rankFiveStrides(y.strides);
  const [filterDepthStride, filterRowStride, filterColStride] =
      rankFiveStrides(filterStrides);

  for (let b = 0; b < convInfo.batchSize; ++b) {
    const xOffset1 = b * xBatchStride;
    const yOffset1 = b * yBatchStride;
    for (let yF = 0; yF < convInfo.outDepth; ++yF) {
      const yOffset2 = yOffset1 + yF * yDepthStride;
      const xFCorner = yF * convInfo.strideDepth - padFront;
      for (let wF = 0; wF < filterDepth; ++wF) {
        const xF = xFCorner + wF * dilationDepth;
        if (xF < 0 || xF >= convInfo.inDepth) {
          continue;
        }
        const wOffset1 = wF * filterDepthStride;
        const xOffset2 = xOffset1 + xF * xDepthStride;

        for (let yR = 0; yR < convInfo.outHeight; ++yR) {
          const yOffset3 = yOffset2 + yR * yRowStride;
          const xRCorner = yR * convInfo.strideHeight - padTop;
          for (let wR = 0; wR < filterHeight; ++wR) {
            const xR = xRCorner + wR * dilationHeight;
            if (xR < 0 || xR >= convInfo.inHeight) {
              continue;
            }
            const wOffset2 = wOffset1 + wR * filterRowStride;
            const xOffset3 = xOffset2 + xR * xRowStride;
            for (let yC = 0; yC < convInfo.outWidth; ++yC) {
              const yOffset4 = yOffset3 + yC * convInfo.outChannels;
              const xCCorner = yC * convInfo.strideWidth - padLeft;
              for (let wC = 0; wC < filterWidth; ++wC) {
                const xC = xCCorner + wC * dilationWidth;
                if (xC < 0 || xC >= convInfo.inWidth) {
                  continue;
                }
                const wOffset3 = wOffset2 + wC * filterColStride;
                const xOffset4 = xOffset3 + xC * convInfo.inChannels;
                let wOffset4 = wOffset3;
                for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
                  const xVal = requiredCpuEntry(xVals, xOffset4 + d1);
                  for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
                    const index = yOffset4 + d2;
                    yVals[index] = requiredCpuEntry(yVals, index) +
                        xVal * requiredCpuEntry(wVals, wOffset4 + d2);
                  }
                  wOffset4 += convInfo.outChannels;
                }
              }
            }
          }
        }
      }
    }
  }

  return backend.makeTensorInfo(y.shape, y.dtype, y.values);
}

export const conv3DConfig: KernelConfig = {
  kernelName: Conv3D,
  backendName: 'cpu',
  kernelFunc: conv3D
};
