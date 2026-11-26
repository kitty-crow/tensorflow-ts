/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {BatchMatMul, BatchMatMulAttrs, BatchMatMulInputs, broadcast_util, buffer, KernelConfig, KernelAttributes, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex, matrixBatchStrides, matrixDimensions} from '../cpu_util';

import {reshape} from './Reshape';

export function batchMatMul(args: {
  inputs: BatchMatMulInputs,
  attrs: KernelAttributes<BatchMatMulAttrs>,
  backend: MathBackendCPU
}) {
  const {inputs, backend, attrs} = args;
  const {a, b} = inputs;
  const {transposeA, transposeB} = attrs;

  assertNotComplex([a, b], 'matMul');

  const [aRows, aColumns] = matrixDimensions(a.shape);
  const [bRows, bColumns] = matrixDimensions(b.shape);

  const innerShapeA = transposeA ? aRows : aColumns;
  const innerShapeB = transposeB ? bColumns : bRows;

  const outerShapeA = transposeA ? aColumns : aRows;
  const outerShapeB = transposeB ? bRows : bColumns;

  const outerDimsA = a.shape.slice(0, -2);
  const outerDimsB = b.shape.slice(0, -2);

  const batchDimA = util.sizeFromShape(outerDimsA);
  const batchDimB = util.sizeFromShape(outerDimsB);

  const outShapeOuterDims = broadcast_util.assertAndGetBroadcastShape(
      a.shape.slice(0, -2), b.shape.slice(0, -2));
  const outShape = outShapeOuterDims.concat([outerShapeA, outerShapeB]);

  util.assert(
      innerShapeA === innerShapeB,
      () => `Error in matMul: inner shapes (${innerShapeA}) and (` +
          `${innerShapeB}) of Tensors with shapes ${a.shape} and ` +
          `${b.shape} and transposeA=${transposeA}` +
          ` and transposeB=${transposeB} must match.`);

  const a3dShape = transposeA ? [batchDimA, innerShapeA, outerShapeA] :
                                [batchDimA, outerShapeA, innerShapeA];
  const b3dShape = transposeB ? [batchDimB, outerShapeB, innerShapeB] :
                                [batchDimB, innerShapeB, outerShapeB];

  // The rest of the implementation is designed to operate on rank-3 tensors
  const a3d = reshape({inputs: {x: a}, backend, attrs: {shape: a3dShape}});
  const b3d = reshape({inputs: {x: b}, backend, attrs: {shape: b3dShape}});

  const sharedDim = innerShapeA;
  const leftDim = outerShapeA;
  const rightDim = outerShapeB;
  const batchDim = Math.max(batchDimA, batchDimB);

  const a3dValues = backend.requireValues(a3d.dataId) as TypedArray;
  const b3dValues = backend.requireValues(b3d.dataId) as TypedArray;

  const a3dStrides = util.computeStrides(a3d.shape);
  const b3dStrides = util.computeStrides(b3d.shape);

  const [aBatch, aRowStride] = matrixBatchStrides(a3dStrides);
  const [bBatch, bRowStride] = matrixBatchStrides(b3dStrides);
  const aOuterStep = transposeA ? 1 : aRowStride;
  const aInnerStep = transposeA ? aRowStride : 1;
  const bInnerStep = transposeB ? 1 : bRowStride;
  const bOuterStep = transposeB ? bRowStride : 1;

  const size = leftDim * rightDim;
  const result = buffer([batchDim, leftDim, rightDim], a3d.dtype);

  const resVals = result.values as TypedArray;
  const blockSize = backend.blockSize;

  for (let bi = 0; bi < batchDim; bi++) {
    const batchIndexA = bi % batchDimA;
    const batchIndexB = bi % batchDimB;
    for (let i0 = 0; i0 < leftDim; i0 += blockSize) {
      // for when blockSize doesn't evenly divide the input
      const iBlock = Math.min(i0 + blockSize, leftDim);
      for (let j0 = 0; j0 < rightDim; j0 += blockSize) {
        const jBlock = Math.min(j0 + blockSize, rightDim);
        for (let k0 = 0; k0 < sharedDim; k0 += blockSize) {
          const kBlock = Math.min(k0 + blockSize, sharedDim);

          for (let i = i0; i < iBlock; i++) {
            for (let j = j0; j < jBlock; j++) {
              let sum = 0.0;

              for (let k = k0; k < kBlock; k++) {
                const aVal =
                    // tslint:disable-next-line: max-line-length
                    a3dValues[batchIndexA * aBatch + i * aOuterStep + k * aInnerStep];
                const bVal =
                    // tslint:disable-next-line: max-line-length
                    b3dValues[k * bInnerStep + j * bOuterStep + batchIndexB * bBatch];
                sum += aVal * bVal;
              }
              resVals[bi * size + (i * rightDim + j)] += sum;
            }
          }
        }
      }
    }
  }

  backend.disposeIntermediateTensorInfo(a3d);
  backend.disposeIntermediateTensorInfo(b3d);

  // set correct shape on output.
  return backend.makeTensorInfo(
      outShape, result.dtype, result.values as TypedArray);
}

export const batchMatMulConfig: KernelConfig = {
  kernelName: BatchMatMul,
  backendName: 'cpu',
  kernelFunc: batchMatMul,
};
