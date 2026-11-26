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

import {KernelAttributes, KernelConfig, ResizeNearestNeighborGrad, ResizeNearestNeighborGradAttrs, ResizeNearestNeighborGradInputs} from '@tensorflow/tfjs';

import {NodeJSKernelBackend} from '../nodejs_kernel_backend';
import {executeWithInt32Vector, resizeGradientAttrs} from './native_dispatch';

export const resizeNearestNeighborGradConfig: KernelConfig = {
  kernelName: ResizeNearestNeighborGrad,
  backendName: 'tensorflow',
  kernelFunc: (args: {
    inputs: ResizeNearestNeighborGradInputs, backend: NodeJSKernelBackend,
    attrs: KernelAttributes<ResizeNearestNeighborGradAttrs>
  }) => {
    const {images, dy} = args.inputs;
    const backend = args.backend;
    const {alignCorners} =
        args.attrs;

    const opAttrs = resizeGradientAttrs(backend, images.dtype, alignCorners);
    const [, origHeight, origWidth, ] = images.shape;
    return executeWithInt32Vector(backend, ResizeNearestNeighborGrad, opAttrs, [dy], [origHeight, origWidth]);
  }
};
