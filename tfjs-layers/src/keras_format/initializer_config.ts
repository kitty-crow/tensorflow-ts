/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {BaseSerialization} from './types';

// TODO(soergel): Move the CamelCase versions back out of keras_format
// e.g. to src/common.ts.  Maybe even duplicate *all* of these to be pedantic?
/** @docinline */
export type FanMode = 'fanIn'|'fanOut'|'fanAvg';
export const VALID_FAN_MODE_VALUES = ['fanIn', 'fanOut', 'fanAvg'];

// These constants have a snake vs. camel distinction.
export type FanModeSerialization = 'fan_in'|'fan_out'|'fan_avg';

/** @docinline */
export type Distribution = 'normal'|'uniform'|'truncatedNormal';
export const VALID_DISTRIBUTION_VALUES =
    ['normal', 'uniform', 'truncatedNormal'];
// These constants have a snake vs. camel distinction.
export type DistributionSerialization = 'normal'|'uniform'|'truncated_normal';

export type ZerosSerialization = BaseSerialization<'Zeros', {}>;

export type OnesSerialization = BaseSerialization<'Ones', {}>;

export type ConstantConfig = {
  value: number;
};

export type ConstantSerialization =
    BaseSerialization<'Constant', ConstantConfig>;

export type RandomNormalConfig = {
  /** Mean of the random values to generate. */
  mean?: number;
  /** Standard deviation of the random values to generate. */
  stddev?: number;
  /** Used to seed the random generator. */
  seed?: number;
};

export type RandomNormalSerialization =
    BaseSerialization<'RandomNormal', RandomNormalConfig>;

export type RandomUniformConfig = {
  /** Lower bound of the range of random values to generate. */
  minval?: number;
  /** Upper bound of the range of random values to generate. */
  maxval?: number;
  /** Used to seed the random generator. */
  seed?: number;
};

export type RandomUniformSerialization =
    BaseSerialization<'RandomUniform', RandomUniformConfig>;

export type {RandomNormalConfig as TruncatedNormalConfig};

export type TruncatedNormalSerialization =
    BaseSerialization<'TruncatedNormal', RandomNormalConfig>;

export type VarianceScalingConfig = {
  scale?: number;

  mode?: FanModeSerialization;
  distribution?: DistributionSerialization;
  seed?: number;
};

export type VarianceScalingSerialization =
    BaseSerialization<'VarianceScaling', VarianceScalingConfig>;

export type OrthogonalConfig = {
  /** Random number generator seed. */
  seed?: number;
  /** Multiplicative factor to apply to the orthogonal matrix. Defaults to 1. */
  gain?: number;
};

export type OrthogonalSerialization =
    BaseSerialization<'Orthogonal', OrthogonalConfig>;

export type IdentityConfig = {
  gain?: number;
};

export type IdentitySerialization =
    BaseSerialization<'Identity', IdentityConfig>;

// Update initializerClassNames below in concert with this.
export type InitializerSerialization = ZerosSerialization|OnesSerialization|
    ConstantSerialization|RandomUniformSerialization|RandomNormalSerialization|
    TruncatedNormalSerialization|IdentitySerialization|
    VarianceScalingSerialization|OrthogonalSerialization;

export type InitializerClassName = InitializerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid
// and that we have the right number of them.

/**
 * A string array of valid Initializer class names.
 *
 * This is guaranteed to match the `InitializerClassName` union type.
 */
export const initializerClassNames: InitializerClassName[] = [
  'Zeros', 'Ones', 'Constant', 'RandomNormal', 'RandomUniform',
  'TruncatedNormal', 'VarianceScaling', 'Orthogonal', 'Identity'
];
