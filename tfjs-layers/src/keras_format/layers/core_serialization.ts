/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {ActivationSerialization} from '../activation_config';
import {Shape} from '../common';
import {ConstraintSerialization} from '../constraint_config';
import {InitializerSerialization} from '../initializer_config';
import {RegularizerSerialization} from '../regularizer_config';
import {BaseLayerSerialization, LayerConfig} from '../topology_config';

export interface DropoutLayerConfig extends LayerConfig {
  rate: number;
  // CG305: Dropout derives the default mask shape from the input; serialized JSON may omit noise_shape.
  //@cg-expect-error
  noise_shape?: number[];
  // CG305: Dropout can use an automatically selected seed; serialized JSON may omit seed.
  //@cg-expect-error
  seed?: number;
}

export type DropoutLayerSerialization =
    BaseLayerSerialization<'Dropout', DropoutLayerConfig>;

export interface DenseLayerConfig extends LayerConfig {
  units: number;
  // CG305: Dense defaults the activation; serialized JSON may omit activation.
  //@cg-expect-error
  activation?: ActivationSerialization;
  // CG305: Dense defaults the use of bias; serialized JSON may omit use_bias.
  //@cg-expect-error
  use_bias?: boolean;
  // CG305: Dense can infer the input dimension from the graph; serialized JSON may omit input_dim.
  //@cg-expect-error
  input_dim?: number;
  // CG305: Dense defaults the kernel initializer; serialized JSON may omit kernel_initializer.
  //@cg-expect-error
  kernel_initializer?: InitializerSerialization;
  // CG305: Dense defaults the bias initializer; serialized JSON may omit bias_initializer.
  //@cg-expect-error
  bias_initializer?: InitializerSerialization;
  // CG305: Dense does not require a kernel constraint; serialized JSON may omit kernel_constraint.
  //@cg-expect-error
  kernel_constraint?: ConstraintSerialization;
  // CG305: Dense does not require a bias constraint; serialized JSON may omit bias_constraint.
  //@cg-expect-error
  bias_constraint?: ConstraintSerialization;
  // CG305: Dense does not require a kernel regularizer; serialized JSON may omit kernel_regularizer.
  //@cg-expect-error
  kernel_regularizer?: RegularizerSerialization;
  // CG305: Dense does not require a bias regularizer; serialized JSON may omit bias_regularizer.
  //@cg-expect-error
  bias_regularizer?: RegularizerSerialization;
  // CG305: Dense does not require an activity regularizer; serialized JSON may omit activity_regularizer.
  //@cg-expect-error
  activity_regularizer?: RegularizerSerialization;
}

export type DenseLayerSerialization =
    BaseLayerSerialization<'Dense', DenseLayerConfig>;

export type FlattenLayerSerialization =
    BaseLayerSerialization<'Flatten', LayerConfig>;

// CG103: ActivationLayerConfig has no independently matching fields in its reported pairs.
// Serialised activation identifier/config is not runtime layer arguments,
// activation coefficients, recurrent cells, repetition count, permutation
// pattern, masking value, concatenation axis, noise settings, dataFormat,
// wrapped Layer or seed. LayerMetadata is already shared.
//@cg-expect-error
export interface ActivationLayerConfig extends LayerConfig {
  activation: ActivationSerialization;
}

export type ActivationLayerSerialization =
    BaseLayerSerialization<'Activation', ActivationLayerConfig>;

export interface RepeatVectorLayerConfig extends LayerConfig {
  n: number;
}

export type RepeatVectorLayerSerialization =
    BaseLayerSerialization<'RepeatVector', RepeatVectorLayerConfig>;

// CG103: ReshapeLayerConfig has no independently matching fields in its reported pairs.
// Serialised target_shape is a wire key for reshape, not runtime targetShape
// or the unrelated activation, repeat, masking, concatenation, noise, data
// format, wrapped layer or seed fields. Shared LayerMetadata is inherited.
//@cg-expect-error
export interface ReshapeLayerConfig extends LayerConfig {
  target_shape: Shape;
}

export type ReshapeLayerSerialization =
    BaseLayerSerialization<'Reshape', ReshapeLayerConfig>;

// CG103: PermuteLayerConfig has no independently matching fields in its reported pairs.
// Serialised dims is a permutation pattern, not runtime activation
// coefficients, activation identifier, repetition count, mask value,
// concatenation axis, noise/dropout, dataFormat or seed. Shared LayerMetadata
// is inherited.
//@cg-expect-error
export interface PermuteLayerConfig extends LayerConfig {
  dims: number[];
}

export type PermuteLayerSerialization =
    BaseLayerSerialization<'Permute', PermuteLayerConfig>;

export interface MaskingLayerConfig extends LayerConfig {
  maskValue: number;
}

export type MaskingLayerSerialization =
    BaseLayerSerialization<'Masking', MaskingLayerConfig>;

// Update coreLayerClassNames below in concert with this.
export type CoreLayerSerialization =
    DropoutLayerSerialization|DenseLayerSerialization|FlattenLayerSerialization|
    ActivationLayerSerialization|RepeatVectorLayerSerialization|
    ReshapeLayerSerialization|PermuteLayerSerialization|
    MaskingLayerSerialization;

export type CoreLayerClassName = CoreLayerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid.

/**
 * A string array of valid CoreLayer class names.
 *
 * This is guaranteed to match the `CoreLayerClassName` union type.
 */
export const coreLayerClassNames: CoreLayerClassName[] = [
  'Activation',
  'Dense',
  'Dropout',
  'Flatten',
  'Permute',
  'RepeatVector',
  'Reshape',
];
