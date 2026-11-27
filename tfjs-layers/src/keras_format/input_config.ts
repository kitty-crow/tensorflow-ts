/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {DataType} from '@tensorflow/tfjs-core';
import {Shape} from './common';
import {BaseLayerSerialization} from './topology_config';

export type InputLayerConfig = {
  // CG305: Input names may be generated automatically; serialized JSON may omit name.
  //@cg-expect-error
  name?: string;
  // CG305: Input shape can be supplied via batch_input_shape; serialized JSON may omit input_shape.
  //@cg-expect-error
  input_shape?: Shape;
  // CG305: The batch dimension may remain unspecified; serialized JSON may omit batch_size.
  //@cg-expect-error
  batch_size?: number;
  // CG305: Batch shape can be constructed from input_shape and batch_size; serialized JSON may omit batch_input_shape.
  //@cg-expect-error
  batch_input_shape?: Shape;
  // CG305: Input dtype may use the runtime default; serialized JSON may omit dtype.
  //@cg-expect-error
  dtype?: DataType;
  // CG305: Inputs default to dense placeholders; serialized JSON may omit sparse.
  //@cg-expect-error
  sparse?: boolean;
};

// This really should be BaseSerialization because an input layer has no
// inbound_nodes. But, that makes type safety more difficult.

// Update inputLayerClassNames below in concert with this.
export type InputLayerSerialization =
    BaseLayerSerialization<'InputLayer', InputLayerConfig>;

export type InputLayerClassName = InputLayerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid.

/**
 * A string array of valid InputLayer class names.
 *
 * This is guaranteed to match the `InputLayerClassName` union type.
 */
export const inputLayerClassNames: InputLayerClassName[] = [
  'InputLayer',
];
