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

import {backend_util, Draw, DrawAttrs, DrawInputs, KernelConfig, KernelFunc, TypedArray} from '@tensorflow/tfjs-core';
import {TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

export function draw(
    args: {inputs: DrawInputs, backend: MathBackendCPU, attrs: DrawAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {image} = inputs;
  const {canvas, options} = attrs;
  const {contextOptions, imageOptions} = options || {};
  const alpha = imageOptions ?.alpha || 1;

  const contextType = contextOptions ?.contextType || '2d';
  if (contextType !== '2d') {
    throw new Error(`Context type ${
        contextOptions.contextType} is not supported by the CPU backend.`);
  }
  const ctx = canvas.getContext(contextType,
    contextOptions?.contextAttributes || {}) as CanvasRenderingContext2D ;
  if (ctx == null) {
    throw new Error(`Could not get the context with ${contextType} type.`);
  }

  const [height, width, depth] = backend_util.pixelGeometry(image.shape);
  const data = backend.requireValues(image.dataId) as TypedArray;
  const bytes = backend_util.pixelBytes(
      data, height, width, depth, image.dtype, 255 * alpha);

  canvas.width = width;
  canvas.height = height;
  const imageData = new ImageData(bytes, width, height);
  ctx.putImageData(imageData, 0, 0);
  return image;
}

export const drawConfig: KernelConfig = {
  kernelName: Draw,
  backendName: 'cpu',
  kernelFunc: draw as unknown as KernelFunc
};
