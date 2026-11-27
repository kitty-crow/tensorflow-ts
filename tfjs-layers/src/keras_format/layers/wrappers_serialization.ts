/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {BidirectionalMergeMode} from '../common';
import {BaseLayerSerialization, LayerConfig} from '../topology_config';
import {LayerSerialization} from './layer_serialization';
import {RecurrentLayerSerialization} from './recurrent_serialization';

export type TimeDistributedLayerSerialization =
    BaseLayerSerialization<'TimeDistributed', TimeDistributedLayerConfig>;

// CG103: TimeDistributedLayerConfig has no independently matching fields in its reported pairs.
// Serialised layer is a recursive LayerSerialization record, not a live Layer
// or runtime activation, recurrent cell, permutation, mask, concatenation,
// noise or format options. Shared LayerMetadata is inherited.
//@cg-expect-error
export interface TimeDistributedLayerConfig extends LayerConfig {
  layer: LayerSerialization;
}

export type BidirectionalLayerSerialization =
    BaseLayerSerialization<'Bidirectional', BidirectionalLayerConfig>;

// CG103: BidirectionalLayerConfig has no independently matching fields in its reported pairs.
// Serialised layer is a RecurrentLayerSerialization record with wire
// merge_mode, not a live RNN and runtime mergeMode. Other reported constructor
// arguments have unrelated activation, cell, reshape, masking, merge, noise,
// format and seed fields. Shared LayerMetadata is inherited.
//@cg-expect-error
export interface BidirectionalLayerConfig extends LayerConfig {
  layer: RecurrentLayerSerialization;
  // CG305: Bidirectional supplies a default merge mode; serialized JSON may omit merge_mode.
  //@cg-expect-error
  merge_mode?: BidirectionalMergeMode;
}

// Update wrapperLayerClassNames below in concert with this.
export type WrapperLayerSerialization =
    TimeDistributedLayerSerialization|BidirectionalLayerSerialization;

export type WrapperLayerClassName = WrapperLayerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid.

/**
 * A string array of valid WrapperLayer class names.
 *
 * This is guaranteed to match the `WrapperLayerClassName` union type.
 */
export const wrapperLayerClassNames: WrapperLayerClassName[] = [
  'Bidirectional',
  'TimeDistributed',
];
