/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {LayerSerialization} from './layers/layer_serialization';
import {TensorKeyArray} from './node_config';
import {TrainingConfig} from './training_config';
import {BaseSerialization} from './types';

export type ModelConfig = {
  name: string,
  layers: LayerSerialization[],
  input_layers: TensorKeyArray[],
  output_layers: TensorKeyArray[],
};

/**
 * A standard Keras JSON 'Model' configuration.
 */
export interface ModelSerialization extends
    BaseSerialization<'Model', ModelConfig> {
  // CG305: Model JSON may omit legacy backend metadata; serialized JSON may omit backend.
  //@cg-expect-error
  backend?: string;
  // CG305: Model JSON may omit Keras version metadata; serialized JSON may omit keras_version.
  //@cg-expect-error
  keras_version?: string;
}

export type SequentialConfig = {
  layers: LayerSerialization[]
};

/**
 * A standard Keras JSON 'Sequential' configuration.
 */
export interface SequentialSerialization extends
    BaseSerialization<'Sequential', SequentialConfig> {
  // CG305: Sequential JSON may omit legacy backend metadata; serialized JSON may omit backend.
  //@cg-expect-error
  backend?: string;
  // CG305: Sequential JSON may omit Keras version metadata; serialized JSON may omit keras_version.
  //@cg-expect-error
  keras_version?: string;
}

/**
 * A legacy Keras JSON 'Sequential' configuration.
 *
 * It was a bug that Keras Sequential models were recorded with
 * model_config.config as an array of layers, instead of a dict containing a
 * 'layers' entry.  While the bug has been fixed, we still need to be able to
 * read this legacy format.
 */
export type LegacySequentialSerialization = {
  // Note this cannot extend `BaseSerialization` because of the bug.
  class_name: 'Sequential';

  config: LayerSerialization[];
  // CG305: Legacy Sequential JSON may omit backend metadata; serialized JSON may omit backend.
  //@cg-expect-error
  backend?: string;
  // CG305: Legacy Sequential JSON may omit Keras version metadata; serialized JSON may omit keras_version.
  //@cg-expect-error
  keras_version?: string;
};

/**
 * Contains the description of a KerasModel, as well as the configuration
 * necessary to train that model.
 */
export type KerasFileSerialization = {
  // aka ModelTopology?
  model_config: ModelSerialization|SequentialSerialization|
  LegacySequentialSerialization;
  training_config: TrainingConfig;
};
