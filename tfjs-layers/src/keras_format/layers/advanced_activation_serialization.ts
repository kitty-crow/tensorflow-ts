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
import {BaseLayerSerialization, LayerConfig} from '../topology_config';

// CG103: ReLULayerConfig has no independently matching fields in its reported pairs.
// The wire-only max_value field is not alpha, theta, cells, activation, n,
// dims, maskValue, axis, stddev, rate, dataFormat or seed. Runtime LayerArgs
// already shares LayerMetadata; the remaining fields represent different
// contracts.
//@cg-expect-error
export interface ReLULayerConfig extends LayerConfig {
  // CG305: ReLU permits no maximum cap; serialized JSON may omit max_value.
  //@cg-expect-error
  max_value?: number;
}

export type ReLULayerSerialization =
    BaseLayerSerialization<'ReLU', ReLULayerConfig>;

export interface LeakyReLULayerConfig extends LayerConfig {
  // CG305: LeakyReLU supplies its default negative slope; serialized JSON may omit alpha.
  //@cg-expect-error
  alpha?: number;
}

export type LeakyReLULayerSerialization =
    BaseLayerSerialization<'LeakyReLU', LeakyReLULayerConfig>;

export interface PReLULayerConfig extends LayerConfig {
  // CG305: PReLU supplies its default alpha initializer; serialized JSON may omit alpha_initializer.
  //@cg-expect-error
  alpha_initializer?: InitializerSerialization;
  // CG305: PReLU does not require alpha regularization; serialized JSON may omit alpha_regularizer.
  //@cg-expect-error
  alpha_regularizer?: RegularizerSerialization;
  // CG305: PReLU does not require an alpha constraint; serialized JSON may omit alpha_constraint.
  //@cg-expect-error
  alpha_constraint?: ConstraintSerialization;
  // CG305: PReLU supports no shared axes; serialized JSON may omit shared_axes.
  //@cg-expect-error
  shared_axes?: number|number[];
}

export type PReLULayerSerialization =
    BaseLayerSerialization<'PReLU', PReLULayerConfig>;

export interface ELULayerConfig extends LayerConfig {
  // CG305: ELU supplies its default alpha; serialized JSON may omit alpha.
  //@cg-expect-error
  alpha?: number;
}

export type ELULayerSerialization =
    BaseLayerSerialization<'ELU', ELULayerConfig>;

export interface ThresholdedReLULayerConfig extends LayerConfig {
  // CG305: ThresholdedReLU supplies its default threshold; serialized JSON may omit theta.
  //@cg-expect-error
  theta?: number;
}

export type ThresholdedReLULayerSerialization =
    BaseLayerSerialization<'ThresholdedReLU', ThresholdedReLULayerConfig>;

export interface SoftmaxLayerConfig extends LayerConfig {
  // CG305: Softmax supplies its default axis; serialized JSON may omit axis.
  //@cg-expect-error
  axis?: number;
}

export type SoftmaxLayerSerialization =
    BaseLayerSerialization<'Softmax', SoftmaxLayerConfig>;

// Update advancedActivationLayerClassNames below in concert with this.
export type AdvancedActivationLayerSerialization = ReLULayerSerialization|
    LeakyReLULayerSerialization|PReLULayerSerialization|ELULayerSerialization|
    ThresholdedReLULayerSerialization|SoftmaxLayerSerialization;

export type AdvancedActivationLayerClassName =
    AdvancedActivationLayerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid.

/**
 * A string array of valid AdvancedActivationLayer class names.
 *
 * This is guaranteed to match the `AdvancedActivationLayerClassName` union
 * type.
 */
export const advancedActivationLayerClassNames:
    AdvancedActivationLayerClassName[] = [
      'ReLU',
      'LeakyReLU',
      'PReLU',
      'ELU',
      'ThresholdedReLU',
      'Softmax',
    ];
