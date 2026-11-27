/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {DataFormatSerialization, PaddingMode} from '../common';
import {BaseLayerSerialization, LayerConfig} from '../topology_config';

export interface Pooling1DLayerConfig extends LayerConfig {
  // CG305: Pooling1D supplies its default window size; serialized JSON may omit pool_size.
  //@cg-expect-error
  pool_size?: [number];
  // CG305: Pooling1D defaults strides to its window size; serialized JSON may omit strides.
  //@cg-expect-error
  strides?: [number];
  // CG305: Pooling1D defaults padding; serialized JSON may omit padding.
  //@cg-expect-error
  padding?: PaddingMode;
}

export type MaxPooling1DLayerSerialization =
    BaseLayerSerialization<'MaxPooling1D', Pooling1DLayerConfig>;

export type AveragePooling1DLayerSerialization =
    BaseLayerSerialization<'AveragePooling1D', Pooling1DLayerConfig>;

export interface Pooling2DLayerConfig extends LayerConfig {
  // CG305: Pooling2D supplies its default window size; serialized JSON may omit pool_size.
  //@cg-expect-error
  pool_size?: number|[number, number];
  // CG305: Pooling2D defaults strides to its window size; serialized JSON may omit strides.
  //@cg-expect-error
  strides?: number|[number, number];
  // CG305: Pooling2D defaults padding; serialized JSON may omit padding.
  //@cg-expect-error
  padding?: PaddingMode;
  // CG305: Pooling2D defaults the channel ordering; serialized JSON may omit data_format.
  //@cg-expect-error
  data_format?: DataFormatSerialization;
}

export type MaxPooling2DLayerSerialization =
    BaseLayerSerialization<'MaxPooling2D', Pooling2DLayerConfig>;

export type AveragePooling2DLayerSerialization =
    BaseLayerSerialization<'AveragePooling2D', Pooling2DLayerConfig>;

export type GlobalAveragePooling1DLayerSerialization =
    BaseLayerSerialization<'GlobalAveragePooling1D', LayerConfig>;

export type GlobalMaxPooling1DLayerSerialization =
    BaseLayerSerialization<'GlobalMaxPooling1D', LayerConfig>;

// CG103: GlobalPooling2DLayerConfig has no independently matching fields in its reported pairs.
// Wire data_format uses the serialisation enum rather than runtime dataFormat;
// activation parameters, recurrent cells, runtime activation, mask,
// concatenation axis and dropout rate do not describe pooling. Shared
// LayerMetadata is inherited.
//@cg-expect-error
export interface GlobalPooling2DLayerConfig extends LayerConfig {
  // CG305: GlobalPooling2D defaults the channel ordering; serialized JSON may omit data_format.
  //@cg-expect-error
  data_format?: DataFormatSerialization;
}

export type GlobalAveragePooling2DLayerSerialization = BaseLayerSerialization<
    'GlobalAveragePooling2D', GlobalPooling2DLayerConfig>;

export type GlobalMaxPooling2DLayerSerialization =
    BaseLayerSerialization<'GlobalMaxPooling2D', GlobalPooling2DLayerConfig>;

// Update poolingLayerClassNames below in concert with this.
export type PoolingLayerSerialization = MaxPooling1DLayerSerialization|
    AveragePooling1DLayerSerialization|MaxPooling2DLayerSerialization|
    AveragePooling2DLayerSerialization|GlobalAveragePooling1DLayerSerialization|
    GlobalMaxPooling1DLayerSerialization|
    GlobalAveragePooling2DLayerSerialization|
    GlobalMaxPooling2DLayerSerialization;

export type PoolingLayerClassName = PoolingLayerSerialization['class_name'];

// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid.

/**
 * A string array of valid PoolingLayer class names.
 *
 * This is guaranteed to match the `PoolingLayerClassName` union type.
 */
export const poolingLayerClassNames: PoolingLayerClassName[] = [
  'AveragePooling1D',
  'AveragePooling2D',
  'GlobalAveragePooling1D',
  'GlobalAveragePooling2D',
  'GlobalMaxPooling1D',
  'GlobalMaxPooling2D',
  'MaxPooling1D',
  'MaxPooling2D',
];
