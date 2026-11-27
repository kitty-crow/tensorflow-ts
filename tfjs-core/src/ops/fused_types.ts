/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

import type {KernelBackend} from '../backends/backend';
import {Tensor, Tensor3D, Tensor4D} from '../tensor';
import type {TensorInfo} from '../tensor_info';
import {Conv2DInfo} from './conv_util';

/** Shared convolution kernel inputs, retaining the concrete backend type. */
export type Conv2DKernelConfig<B extends KernelBackend> = {
  x: TensorInfo,
  filter: TensorInfo,
  convInfo: Conv2DInfo,
  backend: B,
  // CG305: Fused operations support execution without a bias tensor.
  //@cg-expect-error
  bias?: TensorInfo,
  // CG305: Only PReLU activation requires its weights tensor.
  //@cg-expect-error
  preluActivationWeights?: TensorInfo,
  // CG305: An alpha override is only relevant for leaky ReLU activation.
  //@cg-expect-error
  leakyreluAlpha?: number,
  // CG305: Fused operations default to linear activation when not requested.
  //@cg-expect-error
  activation?: Activation
};

export type FusedConv2DConfig = {
  input: Tensor4D,
  filter: Tensor4D,
  convInfo: Conv2DInfo,
  // CG305: Fused operations support execution without a bias tensor.
  //@cg-expect-error
  bias?: Tensor,
  // CG305: Fused operations default to linear activation when not requested.
  //@cg-expect-error
  activation?: Activation,
  // CG305: Only PReLU activation requires its weights tensor.
  //@cg-expect-error
  preluActivationWeights?: Tensor,
  // CG305: An alpha override is only relevant for leaky ReLU activation.
  //@cg-expect-error
  leakyreluAlpha?: number
};

export type FusedBatchMatMulConfig = {
  a: Tensor3D,
  b: Tensor3D,
  transposeA: boolean,
  transposeB: boolean,
  // CG305: Fused operations support execution without a bias tensor.
  //@cg-expect-error
  bias?: Tensor,
  // CG305: Fused operations default to linear activation when not requested.
  //@cg-expect-error
  activation?: Activation,
  // CG305: Only PReLU activation requires its weights tensor.
  //@cg-expect-error
  preluActivationWeights?: Tensor,
  // CG305: An alpha override is only relevant for leaky ReLU activation.
  //@cg-expect-error
  leakyreluAlpha?: number
};

export type Activation =
    'linear'|'relu'|'prelu'|'elu'|'relu6'|'leakyrelu'|'sigmoid';
