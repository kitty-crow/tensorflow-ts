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
import {ConstraintSerialization} from '../constraint_config';
import {InitializerSerialization} from '../initializer_config';
import {RegularizerSerialization} from '../regularizer_config';
import {BaseLayerSerialization, LayerConfig} from '../topology_config';
import {BaseSerialization} from '../types';

export interface BaseRNNLayerConfig extends LayerConfig {
  // CG305: Concrete recurrent layer classes construct their own cell; Keras JSON may omit cell.
  //@cg-expect-error
  cell?: RNNCellSerialization|RNNCellSerialization[];
  // CG305: Omission selects the final output rather than a sequence; Keras JSON may omit return_sequences.
  //@cg-expect-error
  return_sequences?: boolean;
  // CG305: Omission selects output without a returned state; Keras JSON may omit return_state.
  //@cg-expect-error
  return_state?: boolean;
  // CG305: Omission selects forward iteration; Keras JSON may omit go_backwards.
  //@cg-expect-error
  go_backwards?: boolean;
  // CG305: Omission selects non-stateful execution; Keras JSON may omit stateful.
  //@cg-expect-error
  stateful?: boolean;
  // CG305: Omission selects the default non-unrolled execution; Keras JSON may omit unroll.
  //@cg-expect-error
  unroll?: boolean;
  // CG305: Input shape may be supplied or inferred elsewhere; Keras JSON may omit input_dim.
  //@cg-expect-error
  input_dim?: number;
  // CG305: Sequence length may be supplied or inferred elsewhere; Keras JSON may omit input_length.
  //@cg-expect-error
  input_length?: number;
}

export interface SimpleRNNCellConfig extends LayerConfig {
  units: number;
  // CG305: The constructor supplies its activation default; Keras JSON may omit activation.
  //@cg-expect-error
  activation?: ActivationSerialization;
  // CG305: The constructor supplies the bias-use default; Keras JSON may omit use_bias.
  //@cg-expect-error
  use_bias?: boolean;
  // CG305: The layer supplies the default initializer; Keras JSON may omit kernel_initializer.
  //@cg-expect-error
  kernel_initializer?: InitializerSerialization;
  // CG305: The layer supplies the default initializer; Keras JSON may omit recurrent_initializer.
  //@cg-expect-error
  recurrent_initializer?: InitializerSerialization;
  // CG305: The layer supplies the default initializer; Keras JSON may omit bias_initializer.
  //@cg-expect-error
  bias_initializer?: InitializerSerialization;
  // CG305: No regularizer is required; Keras JSON may omit kernel_regularizer.
  //@cg-expect-error
  kernel_regularizer?: RegularizerSerialization;
  // CG305: No regularizer is required; Keras JSON may omit recurrent_regularizer.
  //@cg-expect-error
  recurrent_regularizer?: RegularizerSerialization;
  // CG305: No regularizer is required; Keras JSON may omit bias_regularizer.
  //@cg-expect-error
  bias_regularizer?: RegularizerSerialization;
  // CG305: No constraint is required; Keras JSON may omit kernel_constraint.
  //@cg-expect-error
  kernel_constraint?: ConstraintSerialization;
  // CG305: No constraint is required; Keras JSON may omit recurrent_constraint.
  //@cg-expect-error
  recurrent_constraint?: ConstraintSerialization;
  // CG305: No constraint is required; Keras JSON may omit bias_constraint.
  //@cg-expect-error
  bias_constraint?: ConstraintSerialization;
  // CG305: Omission selects the default zero input dropout; Keras JSON may omit dropout.
  //@cg-expect-error
  dropout?: number;
  // CG305: Omission selects the default zero recurrent dropout; Keras JSON may omit recurrent_dropout.
  //@cg-expect-error
  recurrent_dropout?: number;
}

export type SimpleRNNCellSerialization =
    BaseSerialization<'SimpleRNNCell', SimpleRNNCellConfig>;

export interface SimpleRNNLayerConfig extends BaseRNNLayerConfig {
  units: number;
  // CG305: The constructor supplies its activation default; Keras JSON may omit activation.
  //@cg-expect-error
  activation?: ActivationSerialization;
  // CG305: The constructor supplies the bias-use default; Keras JSON may omit use_bias.
  //@cg-expect-error
  use_bias?: boolean;
  // CG305: The layer supplies the default initializer; Keras JSON may omit kernel_initializer.
  //@cg-expect-error
  kernel_initializer?: InitializerSerialization;
  // CG305: The layer supplies the default initializer; Keras JSON may omit recurrent_initializer.
  //@cg-expect-error
  recurrent_initializer?: InitializerSerialization;
  // CG305: The layer supplies the default initializer; Keras JSON may omit bias_initializer.
  //@cg-expect-error
  bias_initializer?: InitializerSerialization;
  // CG305: No regularizer is required; Keras JSON may omit kernel_regularizer.
  //@cg-expect-error
  kernel_regularizer?: RegularizerSerialization;
  // CG305: No regularizer is required; Keras JSON may omit recurrent_regularizer.
  //@cg-expect-error
  recurrent_regularizer?: RegularizerSerialization;
  // CG305: No regularizer is required; Keras JSON may omit bias_regularizer.
  //@cg-expect-error
  bias_regularizer?: RegularizerSerialization;
  // CG305: No constraint is required; Keras JSON may omit kernel_constraint.
  //@cg-expect-error
  kernel_constraint?: ConstraintSerialization;
  // CG305: No constraint is required; Keras JSON may omit recurrent_constraint.
  //@cg-expect-error
  recurrent_constraint?: ConstraintSerialization;
  // CG305: No constraint is required; Keras JSON may omit bias_constraint.
  //@cg-expect-error
  bias_constraint?: ConstraintSerialization;
  // CG305: Omission selects the default zero input dropout; Keras JSON may omit dropout.
  //@cg-expect-error
  dropout?: number;
  // CG305: Omission selects the default zero recurrent dropout; Keras JSON may omit recurrent_dropout.
  //@cg-expect-error
  recurrent_dropout?: number;
}

export type SimpleRNNLayerSerialization =
    BaseLayerSerialization<'SimpleRNN', SimpleRNNLayerConfig>;

export interface GRUCellConfig extends SimpleRNNCellConfig {
  // CG305: The recurrent cell supplies its default gate activation; Keras JSON may omit recurrent_activation.
  //@cg-expect-error
  recurrent_activation?: string;
  // CG305: The recurrent implementation setting has a default; Keras JSON may omit implementation.
  //@cg-expect-error
  implementation?: number;
}

export type GRUCellSerialization = BaseSerialization<'GRUCell', GRUCellConfig>;

export interface GRULayerConfig extends SimpleRNNLayerConfig {
  // CG305: The recurrent cell supplies its default gate activation; Keras JSON may omit recurrent_activation.
  //@cg-expect-error
  recurrent_activation?: ActivationSerialization;
  // CG305: The recurrent implementation setting has a default; Keras JSON may omit implementation.
  //@cg-expect-error
  implementation?: number;
}

export type GRULayerSerialization =
    BaseLayerSerialization<'GRU', GRULayerConfig>;

export interface LSTMCellConfig extends SimpleRNNCellConfig {
  // CG305: The recurrent cell supplies its default gate activation; Keras JSON may omit recurrent_activation.
  //@cg-expect-error
  recurrent_activation?: ActivationSerialization;
  // CG305: The LSTM cell supplies its default forget-bias behaviour; Keras JSON may omit unit_forget_bias.
  //@cg-expect-error
  unit_forget_bias?: boolean;
  // CG305: The recurrent implementation setting has a default; Keras JSON may omit implementation.
  //@cg-expect-error
  implementation?: number;
}

export type LSTMCellSerialization =
    BaseSerialization<'LSTMCell', LSTMCellConfig>;

export interface LSTMLayerConfig extends SimpleRNNLayerConfig {
  // CG305: The recurrent cell supplies its default gate activation; Keras JSON may omit recurrent_activation.
  //@cg-expect-error
  recurrent_activation?: ActivationSerialization;
  // CG305: The LSTM cell supplies its default forget-bias behaviour; Keras JSON may omit unit_forget_bias.
  //@cg-expect-error
  unit_forget_bias?: boolean;
  // CG305: The recurrent implementation setting has a default; Keras JSON may omit implementation.
  //@cg-expect-error
  implementation?: number;
}
export type LSTMLayerSerialization =
    BaseLayerSerialization<'LSTM', LSTMLayerConfig>;

// CG103: StackedRNNCellsConfig has no independently matching fields in its reported pairs.
// Serialised cells are RNNCellSerialization records rather than live RNNCell
// instances or a wrapped Layer; unrelated activation coefficients, flatten
// format, activation identifier, concatenation axis and global pooling format
// are not cell stacks. Shared LayerMetadata is inherited.
//@cg-expect-error
export interface StackedRNNCellsConfig extends LayerConfig {
  // TODO(soergel): consider whether we can avoid improperly mixing
  // Simple / LSTM / GRU cells here and in the above Layer serializations.
  cells: RNNCellSerialization[];
}

export type StackedRNNCellsSerialization =
    BaseSerialization<'StackedRNNCells', StackedRNNCellsConfig>;

export type RNNCellSerialization = SimpleRNNCellSerialization|
    GRUCellSerialization|LSTMCellSerialization|StackedRNNCellsSerialization;

// Update recurrentLayerClassNames below in concert with this.
export type RecurrentLayerSerialization =
    SimpleRNNLayerSerialization|LSTMLayerSerialization|GRULayerSerialization;

export type RecurrentLayerClassName = RecurrentLayerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid.

/**
 * A string array of valid RecurrentLayer class names.
 *
 * This is guaranteed to match the `RecurrentLayerClassName` union type.
 */
export const recurrentLayerClassNames: RecurrentLayerClassName[] = [
  'GRU',
  'LSTM',
  'SimpleRNN',
];
