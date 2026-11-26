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
import {NodeConfig} from './node_config';
import {BaseSerialization, PyJson, PyJsonDict} from './types';

/** Metadata shared by runtime layer arguments and serialised layer configs. */
export interface LayerMetadata {
  /**
   * The data-type for this layer. Defaults to 'float32'.
   * This argument is only applicable to input layers (the first layer of a
   * model).
   */
  dtype?: DataType;
  /** Name for this layer. */
  name?: string;
  /**
   * Whether the weights of this layer are updatable by `fit`.
   * Defaults to true.
   */
  trainable?: boolean;
}

/** Constructor arguments for Layer. */
export interface LayerConfig extends PyJsonDict, LayerMetadata {
  input_shape?: Shape;
  batch_input_shape?: Shape;
  batch_size?: number;
  input_dtype?: DataType;
}

/**
 * Converts a subtype of `LayerConfig` to a variant with restricted keys.
 *
 * This is a bit tricky because `keyof` obtains only local fields, not inherited
 * fields.  Thus, this type combines the keys from the `LayerConfig` supertype
 * with those of the specific subtype.
 *
 * See ./types.ts for an explanation of the PyJson type.
 */
export type JsonLayer<C extends LayerConfig> = C&LayerConfig&
    PyJson<Extract<keyof C, string>|Extract<keyof LayerConfig, string>>;

/**
 * A Keras JSON entry representing a layer.
 *
 * The Keras JSON convention is to provide the `class_name` (i.e., the layer
 * type) at the top level, and then to place the layer-specific configuration in
 * a `config` subtree.  These layer-specific configurations are provided by
 * subtypes of `LayerConfig`.  Thus, this `*Serialization` has a type parameter
 * giving the specific type of the wrapped `LayerConfig`.
 */
export interface BaseLayerSerialization<N extends string, C extends LayerConfig>
    extends BaseSerialization<N, JsonLayer<C>> {
  name: string;
  inbound_nodes?: NodeConfig[];
}
