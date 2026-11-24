/**
 * @license
 * Copyright 2023 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import { LayerArgs, Layer } from './topology';
import { RandomSeed } from '../backend/random_seed';
import { serialization } from '@tensorflow/tfjs-core';

export declare interface BaseRandomLayerArgs extends LayerArgs {
  seed?: number;
}

/** Shared options for random height and width transformations. */
export interface RandomResizeArgs extends BaseRandomLayerArgs {
  factor: number | [number, number];
  interpolation?: 'bilinear' | 'nearest';
  autoVectorize?: boolean;
}

export abstract class BaseRandomLayer extends Layer {
  // A layer handle the random number creation and savemodel behavior.
  /** @nocollapse */
  static className = 'BaseRandomLayer';
  protected randomGenerator: RandomSeed;

  constructor(args: BaseRandomLayerArgs) {
    super(args);
    this.randomGenerator = new RandomSeed(args.seed);
  }

  override getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      'seed': this.randomGenerator.seed
    };

    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
