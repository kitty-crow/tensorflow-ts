/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Optimizers.
 */

import {Optimizer, train} from '@tensorflow/tfjs-core';

import {epsilon} from './backend/common';

import {ValueError} from './errors';

// Add (de)serialize()

// Porting note: This diverges from the PyKeras implementation and may need to
// change based on (de)serialization requirements.
export function getOptimizer(identifier: string): Optimizer {
  switch (identifier) {
    case 'Adagrad':
    case 'adagrad':
      return train.adagrad(0.01);
    case 'Adadelta':
    case 'adadelta':
      return train.adadelta(1, 0.95, epsilon());
    case 'Adam':
    case 'adam':
      return train.adam(0.001, 0.9, 0.999, epsilon());
    case 'Adamax':
    case 'adamax':
      return train.adamax(0.002, 0.9, 0.999, epsilon(), 0);
    case 'RMSProp':
    case 'rmsprop':
      return train.rmsprop(0.001, 0.9, 0, epsilon());
    case 'SGD':
    case 'sgd':
      return train.sgd(0.01);
  }
  throw new ValueError(`Unknown Optimizer ${identifier}`);
}
