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
import {DataType, TensorInfo, util} from '@tensorflow/tfjs-core';

/** Select supported optional device features in request order. */
export function requiredWebGPUFeatures(
    features: Pick<GPUSupportedFeatures, 'has'>): GPUFeatureName[] {
  const required: GPUFeatureName[] = [];
  if (features.has('timestamp-query')) {
    required.push('timestamp-query');
  }
  if (features.has('bgra8unorm-storage')) {
    required.push('bgra8unorm-storage');
  }
  return required;
}

/** Multiply the validated dimensions selected by a dispatch axis. */
function dispatchAxisSize(shape: number[], dimensions: number[]): number {
  let product = 1;
  for (const axis of dimensions) {
    const dimension = shape[axis];
    if (dimension == null) {
      throw new Error(`Missing output shape dimension ${axis}.`);
    }
    product *= dimension;
  }
  return product;
}

export function tilesFitEvenlyIntoShape(
    tileSize: number[], shape: number[]): boolean {
  if (tileSize.length !== shape.length) {
    throw new Error(
        `Cannot compute whether rank ${tileSize.length}` +
        ` tiles fit evenly into rank ${shape.length} shape` +
        ` - ranks must match.`);
  }
  return shape.every((dim, dimIdx) => {
    const tile = tileSize[dimIdx];
    if (tile == null) {
      throw new Error(`Missing tile size for dimension ${dimIdx}.`);
    }
    return dim % tile === 0;
  });
}

// Computes dispatch geometry based on layout of output dimensions and
// workgroupSize.
export function computeDispatch(
    layout: {x: number[], y?: number[], z?: number[]}, outputShape: number[],
    workgroupSize: [number, number, number] = [1, 1, 1],
    elementsPerThread: [number, number, number] =
        [1, 1, 1]): [number, number, number] {
  const [dispatchX, dispatchY, dispatchZ] = [
    Math.ceil(
        dispatchAxisSize(outputShape, layout.x) /
        (workgroupSize[0] * elementsPerThread[0])),
    layout.y ? Math.ceil(
                   dispatchAxisSize(outputShape, layout.y) /
                   (workgroupSize[1] * elementsPerThread[1])) :
               1,
    layout.z ? Math.ceil(
                   dispatchAxisSize(outputShape, layout.z) /
                   (workgroupSize[2] * elementsPerThread[2])) :
               1
  ];
  return [dispatchX, dispatchY, dispatchZ];
}

export type WorkgroupInfo = {
  workgroupSize: [number, number, number],
  elementsPerThread: [number, number, number],
};

export function computeWorkgroupInfoForMatMul(
    dimAOuter: number, dimInner: number, dimBOuter: number,
    transposeA = false): WorkgroupInfo {
  // These are experimental values. Usually, we need to adjust the work group
  // size based on the input shapes to improve the EU occupancy.
  // TODO: WebGPU limits the maximum allowed shared memory size as 16K. To make
  // sure it doesn't exceed this limitations. Temporarily reduce the work group
  // size to [8, 8, 1] and the work per thread size is [4, 4, 1]. But we should
  // revisit it and find the balance between work group size and work per thread
  // size.
  const workgroupSize: [number, number, number] = [8, 8, 1];
  const elementsPerThread: [number, number, number] = [4, 4, 1];

  if (!transposeA && dimAOuter <= 8) {
    elementsPerThread[1] = 1;
  }

  if (!transposeA && dimInner <= 16 && dimBOuter <= 16) {
    workgroupSize[0] = 4;
  }

  return {workgroupSize, elementsPerThread};
}

export function computeWorkgroupSizeForConv2d(
    layout: {x: number[], y?: number[], z?: number[]}, outputShape: number[],
    isVec4 = false): [number, number, number] {
  if (isVec4) {
    return [8, 8, 1];
  }

  const y = layout.y;
  if (y == null) {
    throw new Error('Conv2D dispatch requires a y axis.');
  }
  const dim0 = dispatchAxisSize(outputShape, layout.x);
  const dim1 = dispatchAxisSize(outputShape, y);
  // TODO(jiajia.qin@intel.com): More fine tune based on outputShape.
  // These are experimental values. Usually, we need to adjust the work group
  // size based on the output shape. For example, when one dimension is smaller
  // than 4, it will be wasteful if we assign a larger size for this dimension,
  // which results lots of threads doing useless work and reduces parallelism
  // of hardware threads. But it is always a balance between work group size
  // and shared memory. If one dimension is too small, such as 1, shared memory
  // will won't be fully utilized.
  if (dim0 <= 4) {
    return [4, 16, 1];
  }
  if (dim1 <= 4) {
    return [16, 4, 1];
  }

  return [16, 16, 1];
}

export function computeWorkPerThreadForConv2d(
    layout: {x: number[], y?: number[], z?: number[]}, outputShape: number[],
    isVec4 = false): [number, number, number] {
  if (isVec4) {
    return [4, 4, 1];
  }

  const y = layout.y;
  if (y == null) {
    throw new Error('Conv2D dispatch requires a y axis.');
  }
  const dim0 = dispatchAxisSize(outputShape, layout.x);
  const dim1 = dispatchAxisSize(outputShape, y);
  // TODO(jiajia.qin@intel.com): More fine tune based on outputShape.
  // The following conditions correspond to the values set in
  // computeWorkgroupSizeForConv2d.
  if (dim0 <= 4) {
    return [1, 2, 1];
  }
  if (dim1 <= 4) {
    return [2, 1, 1];
  }

  return [2, 2, 1];
}

export function flatDispatchLayout(shape: number[]) {
  return {x: shape.map((d, i) => i)};
}

export function GPUBytesPerElement(dtype: DataType): number {
  if (dtype === 'float32' || dtype === 'int32' || dtype === 'bool' ||
      dtype === 'string') {
    return 4;
  }
  if (dtype === 'complex64') {
    return 8;
  } else {
    throw new Error(`Unknown dtype ${dtype}`);
  }
}

export function isWebGPUSupported(): boolean {
  return !!(typeof globalThis !== 'undefined' && (globalThis.navigator)
    && (globalThis.navigator.gpu));
}

export function assertNotComplex(
    tensor: TensorInfo|TensorInfo[], opName: string): void {
  if (!Array.isArray(tensor)) {
    tensor = [tensor];
  }
  tensor.forEach(t => {
    if (t != null) {
      util.assert(
          t.dtype !== 'complex64',
          () => `${opName} does not support complex64 tensors ` +
              'in the WebGPU backend.');
    }
  });
}

export enum MatMulProgramType {
  MatMulReduceProgram,
  MatMulSplitKProgram,
  MatMulSmallOutputSizeProgram,
  MatMulPackedProgram,
  MatMulMax
}
