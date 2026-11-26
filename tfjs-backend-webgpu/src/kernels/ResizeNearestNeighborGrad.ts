/**
 * @license
 * Copyright 2023 Google LLC.
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

import {KernelConfig, KernelAttributes, ResizeNearestNeighborGrad, ResizeNearestNeighborGradAttrs, ResizeNearestNeighborGradInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {ResizeNearestNeigborBackpropProgram} from '../resize_nearest_neighbor_backprop_webgpu';

import {resizeBackpropGeometry} from './resize_backprop_geometry';

export function resizeNearestNeighborGrad(args: {
  inputs: ResizeNearestNeighborGradInputs,
  backend: WebGPUBackend,
  attrs: KernelAttributes<ResizeNearestNeighborGradAttrs>
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {images, dy} = inputs;
  const {alignCorners} = attrs;

  const {effectiveXSize, effectiveYSize, invHeightScale, invWidthScale,
    winHeight, winWidth} = resizeBackpropGeometry(
      images.shape as [number, number, number, number],
      dy.shape as [number, number, number, number], alignCorners);

  const program = new ResizeNearestNeigborBackpropProgram(
      images.shape as [number, number, number, number], alignCorners);
  const uniformData = [
    {type: 'int32', data: effectiveXSize},
    {type: 'int32', data: effectiveYSize},
    {type: 'float32', data: [invHeightScale]},
    {type: 'float32', data: [invWidthScale]},
    {type: 'int32', data: [winHeight]}, {type: 'int32', data: [winWidth]}
  ];
  return backend.runWebGPUProgram(program, [dy], dy.dtype, uniformData);
}

export const resizeNearestNeighborGradConfig: KernelConfig = {
  kernelName: ResizeNearestNeighborGrad,
  backendName: 'webgpu',
  kernelFunc: resizeNearestNeighborGrad
};
