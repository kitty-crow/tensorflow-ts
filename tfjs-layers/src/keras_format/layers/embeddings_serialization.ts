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

export interface EmbeddingLayerConfig extends LayerConfig {
  input_dim: number;
  output_dim: number;
  // CG305: Embedding supplies a default initializer; serialized JSON may omit embeddings_initializer.
  //@cg-expect-error
  embeddings_initializer?: InitializerSerialization;
  // CG305: Embedding does not require a weight regularizer; serialized JSON may omit embeddings_regularizer.
  //@cg-expect-error
  embeddings_regularizer?: RegularizerSerialization;
  // CG305: Embedding does not require activity regularization; serialized JSON may omit activity_regularizer.
  //@cg-expect-error
  activity_regularizer?: RegularizerSerialization;
  // CG305: Embedding does not require a weight constraint; serialized JSON may omit embeddings_constraint.
  //@cg-expect-error
  embeddings_constraint?: ConstraintSerialization;
  // CG305: Embedding does not require zero masking; serialized JSON may omit mask_zero.
  //@cg-expect-error
  mask_zero?: boolean;
  // CG305: Input sequence length may be inferred; serialized JSON may omit input_length.
  //@cg-expect-error
  input_length?: number|number[];
}

// Update embeddingLayerClassNames below in concert with this.
export type EmbeddingLayerSerialization =
    BaseLayerSerialization<'Embedding', EmbeddingLayerConfig>;

export type EmbeddingLayerClassName = EmbeddingLayerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid.

/**
 * A string array of valid EmbeddingLayer class names.
 *
 * This is guaranteed to match the `EmbeddingLayerClassName` union type.
 */
export const embeddingLayerClassNames: EmbeddingLayerClassName[] = [
  'Embedding',
];
