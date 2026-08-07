/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
import {SampleWeightMode} from './common';
import {LossIdentifier} from './loss_config';
import {OptimizerSerialization} from './optimizer_config';
import {PyJsonDict} from './types';

// TODO(soergel): flesh out known metrics options
export type MetricsIdentifier = string;

/**
 * A valid Keras `loss_weights` value.
 *
 * Keras accepts a scalar for a single-output model, or one weight per output
 * as an Array/tuple or output-name dictionary for a multi-output model.
 */
export type LossWeights = number|number[]|{[key: string]: number};

/**
 * Configuration of the Keras trainer. This includes the configuration to the
 * optimizer, the loss, any metrics to be calculated, etc.
 */
export interface TrainingConfig extends PyJsonDict {
  // tslint:disable-next-line:no-any
  optimizer_config: OptimizerSerialization;
  loss: LossIdentifier|LossIdentifier[]|{[key: string]: LossIdentifier};
  metrics?: MetricsIdentifier[]|{[key: string]: MetricsIdentifier};
  weighted_metrics?: MetricsIdentifier[];
  sample_weight_mode?: SampleWeightMode;
  loss_weights?: LossWeights;
}
