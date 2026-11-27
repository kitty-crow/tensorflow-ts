/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {DataFormatSerialization, PaddingMode} from '../common';
import {ConstraintSerialization} from '../constraint_config';
import {InitializerSerialization} from '../initializer_config';
import {RegularizerSerialization} from '../regularizer_config';
import {BaseLayerSerialization, LayerConfig} from '../topology_config';

export interface BaseConvLayerConfig extends LayerConfig {
  kernel_size: number|number[];
  // CG305: The convolution constructor defaults strides to one; Keras JSON may omit strides.
  //@cg-expect-error
  strides?: number|number[];
  // CG305: The convolution constructor defaults padding to valid; Keras JSON may omit padding.
  //@cg-expect-error
  padding?: PaddingMode;
  // CG305: The layer constructor selects its default channel ordering; Keras JSON may omit data_format.
  //@cg-expect-error
  data_format?: DataFormatSerialization;
  // CG305: The convolution constructor defaults dilation to one; Keras JSON may omit dilation_rate.
  //@cg-expect-error
  dilation_rate?: number|[number]|[number, number];
  // CG305: The constructor supplies its activation default; Keras JSON may omit activation.
  //@cg-expect-error
  activation?: string;
  // CG305: The constructor supplies the bias-use default; Keras JSON may omit use_bias.
  //@cg-expect-error
  use_bias?: boolean;
  // CG305: The layer supplies the default initializer; Keras JSON may omit kernel_initializer.
  //@cg-expect-error
  kernel_initializer?: InitializerSerialization;
  // CG305: The layer supplies the default initializer; Keras JSON may omit bias_initializer.
  //@cg-expect-error
  bias_initializer?: InitializerSerialization;
  // CG305: No constraint is required; Keras JSON may omit kernel_constraint.
  //@cg-expect-error
  kernel_constraint?: ConstraintSerialization;
  // CG305: No constraint is required; Keras JSON may omit bias_constraint.
  //@cg-expect-error
  bias_constraint?: ConstraintSerialization;
  // CG305: No regularizer is required; Keras JSON may omit kernel_regularizer.
  //@cg-expect-error
  kernel_regularizer?: RegularizerSerialization;
  // CG305: No regularizer is required; Keras JSON may omit bias_regularizer.
  //@cg-expect-error
  bias_regularizer?: RegularizerSerialization;
  // CG305: No regularizer is required; Keras JSON may omit activity_regularizer.
  //@cg-expect-error
  activity_regularizer?: RegularizerSerialization;
}

export interface ConvLayerConfig extends BaseConvLayerConfig {
  filters: number;
}

export type Conv1DLayerSerialization =
    BaseLayerSerialization<'Conv1D', ConvLayerConfig>;

export type Conv2DLayerSerialization =
    BaseLayerSerialization<'Conv2D', ConvLayerConfig>;

export type Conv2DTransposeLayerSerialization =
    BaseLayerSerialization<'Conv2DTranspose', ConvLayerConfig>;

export interface SeparableConvLayerConfig extends ConvLayerConfig {
  // CG305: The separable convolution constructor defaults the depth multiplier to one; Keras JSON may omit depth_multiplier.
  //@cg-expect-error
  depth_multiplier?: number;
  // CG305: The layer supplies the default initializer; Keras JSON may omit depthwise_initializer.
  //@cg-expect-error
  depthwise_initializer?: InitializerSerialization;
  // CG305: The layer supplies the default initializer; Keras JSON may omit pointwise_initializer.
  //@cg-expect-error
  pointwise_initializer?: InitializerSerialization;
  // CG305: No regularizer is required; Keras JSON may omit depthwise_regularizer.
  //@cg-expect-error
  depthwise_regularizer?: RegularizerSerialization;
  // CG305: No regularizer is required; Keras JSON may omit pointwise_regularizer.
  //@cg-expect-error
  pointwise_regularizer?: RegularizerSerialization;
  // CG305: No constraint is required; Keras JSON may omit depthwise_constraint.
  //@cg-expect-error
  depthwise_constraint?: ConstraintSerialization;
  // CG305: No constraint is required; Keras JSON may omit pointwise_constraint.
  //@cg-expect-error
  pointwise_constraint?: ConstraintSerialization;
}

export type SeparableConv2DLayerSerialization =
    BaseLayerSerialization<'SeparableConv2D', ConvLayerConfig>;

export interface Cropping2DLayerConfig extends LayerConfig {
  cropping: number|[number, number]|[[number, number], [number, number]];
  // CG305: The layer constructor selects its default channel ordering; Keras JSON may omit data_format.
  //@cg-expect-error
  data_format?: DataFormatSerialization;
}

export type Cropping2DLayerSerialization =
    BaseLayerSerialization<'Cropping2D', Cropping2DLayerConfig>;

// CG103: UpSampling2DLayerConfig has no independently matching fields in its reported pairs.
// Serialised size and data_format describe spatial resizing, not activation
// coefficients, recurrent cells, runtime activation, permutation axes,
// concatenation axis or runtime dataFormat. Shared LayerMetadata is already
// inherited.
//@cg-expect-error
export interface UpSampling2DLayerConfig extends LayerConfig {
  // CG305: The upsampling constructor supplies its default scale; Keras JSON may omit size.
  //@cg-expect-error
  size?: number[];
  // CG305: The layer constructor selects its default channel ordering; Keras JSON may omit data_format.
  //@cg-expect-error
  data_format?: DataFormatSerialization;
}

export type UpSampling2DLayerSerialization =
    BaseLayerSerialization<'UpSampling2D', UpSampling2DLayerConfig>;

// Update convolutionalLayerClassNames below in concert with this.
export type ConvolutionalLayerSerialization =
    Conv1DLayerSerialization|Conv2DLayerSerialization|
    Conv2DTransposeLayerSerialization|SeparableConv2DLayerSerialization|
    Cropping2DLayerSerialization|UpSampling2DLayerSerialization;

export type ConvolutionalLayerClassName =
    ConvolutionalLayerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid.

/**
 * A string array of valid ConvolutionalLayer class names.
 *
 * This is guaranteed to match the `ConvolutionalLayerClassName` union type.
 */
export const convolutionalLayerClassNames: ConvolutionalLayerClassName[] = [
  'Conv1D',
  'Conv2D',
  'Conv2DTranspose',
  'Cropping2D',
  'SeparableConv2D',
  'UpSampling2D',
];
