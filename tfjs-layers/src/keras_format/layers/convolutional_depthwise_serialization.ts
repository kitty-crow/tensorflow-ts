/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {ConstraintSerialization} from '../constraint_config';
import {InitializerSerialization} from '../initializer_config';
import {RegularizerSerialization} from '../regularizer_config';
import {BaseLayerSerialization} from '../topology_config';
import {BaseConvLayerConfig} from './convolutional_serialization';

export interface DepthwiseConv2DLayerConfig extends BaseConvLayerConfig {
  kernel_size: number|[number, number];
  // CG305: Depthwise convolution supplies the default depth multiplier; serialized JSON may omit depth_multiplier.
  //@cg-expect-error
  depth_multiplier?: number;
  // CG305: Depthwise convolution supplies the default initializer; serialized JSON may omit depthwise_initializer.
  //@cg-expect-error
  depthwise_initializer?: InitializerSerialization;
  // CG305: Depthwise convolution need not have a weight constraint; serialized JSON may omit depthwise_constraint.
  //@cg-expect-error
  depthwise_constraint?: ConstraintSerialization;
  // CG305: Depthwise convolution need not have a weight regularizer; serialized JSON may omit depthwise_regularizer.
  //@cg-expect-error
  depthwise_regularizer?: RegularizerSerialization;
}

// Update depthwiseConv2DLayerClassNames below in concert with this.
export type DepthwiseConv2DLayerSerialization =
    BaseLayerSerialization<'DepthwiseConv2D', DepthwiseConv2DLayerConfig>;

export type ConvolutionalDepthwiseLayerSerialization =
    DepthwiseConv2DLayerSerialization;

export type ConvolutionalDepthwiseLayerClassName =
    ConvolutionalDepthwiseLayerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid.

/**
 * A string array of valid ConvolutionalDepthwiseLayer class names.
 *
 * This is guaranteed to match the `ConvolutionalDepthwiseLayerClassName` union
 * type.
 */
export const convolutionalDepthwiseLayerClassNames:
    ConvolutionalDepthwiseLayerClassName[] = [
      'DepthwiseConv2D',
    ];
