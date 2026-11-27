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

export interface BatchNormalizationLayerConfig extends LayerConfig {
  // CG305: BatchNormalization defaults axis to -1; serialized JSON may omit axis.
  //@cg-expect-error
  axis?: number;
  // CG305: BatchNormalization defaults momentum to 0.99; serialized JSON may omit momentum.
  //@cg-expect-error
  momentum?: number;
  // CG305: BatchNormalization defaults epsilon to 0.001; serialized JSON may omit epsilon.
  //@cg-expect-error
  epsilon?: number;
  // CG305: BatchNormalization defaults center to true; serialized JSON may omit center.
  //@cg-expect-error
  center?: boolean;
  // CG305: BatchNormalization defaults scale to true; serialized JSON may omit scale.
  //@cg-expect-error
  scale?: boolean;
  // CG305: BatchNormalization defaults beta to zeros; serialized JSON may omit beta_initializer.
  //@cg-expect-error
  beta_initializer?: InitializerSerialization;
  // CG305: BatchNormalization defaults gamma to ones; serialized JSON may omit gamma_initializer.
  //@cg-expect-error
  gamma_initializer?: InitializerSerialization;
  // CG305: BatchNormalization defaults moving mean to zeros; serialized JSON may omit moving_mean_initializer.
  //@cg-expect-error
  moving_mean_initializer?: InitializerSerialization;
  // CG305: BatchNormalization defaults moving variance to ones; serialized JSON may omit moving_variance_initializer.
  //@cg-expect-error
  moving_variance_initializer?: InitializerSerialization;
  // CG305: BatchNormalization does not require a beta constraint; serialized JSON may omit beta_constraint.
  //@cg-expect-error
  beta_constraint?: ConstraintSerialization;
  // CG305: BatchNormalization does not require a gamma constraint; serialized JSON may omit gamma_constraint.
  //@cg-expect-error
  gamma_constraint?: ConstraintSerialization;
  // CG305: BatchNormalization does not require a beta regularizer; serialized JSON may omit beta_regularizer.
  //@cg-expect-error
  beta_regularizer?: RegularizerSerialization;
  // CG305: BatchNormalization does not require a gamma regularizer; serialized JSON may omit gamma_regularizer.
  //@cg-expect-error
  gamma_regularizer?: RegularizerSerialization;
}

// Update batchNormalizationLayerClassNames below in concert with this.
export type BatchNormalizationLayerSerialization =
    BaseLayerSerialization<'BatchNormalization', BatchNormalizationLayerConfig>;

export type NormalizationLayerSerialization =
    BatchNormalizationLayerSerialization;

export type NormalizationLayerClassName =
    NormalizationLayerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid.

/**
 * A string array of valid NormalizationLayer class names.
 *
 * This is guaranteed to match the `NormalizationLayerClassName` union
 * type.
 */
export const normalizationLayerClassNames: NormalizationLayerClassName[] = [
  'BatchNormalization',
];
