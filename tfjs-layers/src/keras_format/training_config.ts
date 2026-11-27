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
  // CG305: Training may omit metrics; serialized JSON may omit metrics.
  //@cg-expect-error
  metrics?: MetricsIdentifier[]|{[key: string]: MetricsIdentifier};
  // CG305: Training may omit weighted metrics; serialized JSON may omit weighted_metrics.
  //@cg-expect-error
  weighted_metrics?: MetricsIdentifier[];
  // CG305: Training may omit sample weighting mode; serialized JSON may omit sample_weight_mode.
  //@cg-expect-error
  sample_weight_mode?: SampleWeightMode;
  // CG305: Training may omit loss weighting; serialized JSON may omit loss_weights.
  //@cg-expect-error
  loss_weights?: LossWeights;
}
