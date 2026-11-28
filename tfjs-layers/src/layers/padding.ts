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
 * Padding Layers.
 */

// Porting Note: In Python Keras, the padding layers are in convolutional.py,
//   but we decided to put them in a separate file (padding.ts) for clarity.

import * as tfc from '@tensorflow/tfjs-core';
import {serialization, Tensor, tidy} from '@tensorflow/tfjs-core';

import {imageDataFormat} from '../backend/common';
import {InputSpec, Layer, LayerArgs} from '../engine/topology';
import {ValueError} from '../errors';
import {DataFormat, Shape} from '../keras_format/common';
import {Kwargs} from '../types';
import {requiredElement} from '../utils/generic_utils';
import {getExactlyOneShape, getExactlyOneTensor} from '../utils/types_utils';

/**
 * Pads the middle dimension of a 3D tensor.
 *
 * @param x Input `tf.Tensor` to be padded.
 * @param padding `Array` of 2 integers, how many zeros to add at the start and
 *   end of the middle dimension (i.e., dimension 1).
 * @return A padded 3D `tf.Tensor`.
 */
export function temporalPadding(x: Tensor, padding?: [number, number]): Tensor {
  return tidy(() => {
    if (x.rank !== 3) {
      throw new ValueError(
          `temporalPadding expects input tensor to be 3-D, but received a ` +
          `${x.rank}-D tensor.`);
    }

    if (padding == null) {
      padding = [1, 1];
    }
    if (padding.length !== 2) {
      throw new ValueError(
          `temporalPadding expects input padding pattern to be a length-2 ` +
          `array, but received a length-${padding.length} array.`);
    }

    const pattern: Array<[number, number]> = [[0, 0], padding, [0, 0]];
    return tfc.pad(x, pattern);
  });
}

/**
 * Pads the 2nd and 3rd dimensions of a 4D tensor.
 *
 * @param x Input `tf.Tensor` to be padded.
 * @param padding `Array` of two `Array`s, each of which is an `Array` of two
 *   integers. The amount of padding at the beginning and end of the 2nd and 3rd
 *   dimensions, respectively.
 * @param dataFormat 'channelsLast' (default) or 'channelsFirst'.
 * @return Padded 4D `tf.Tensor`.
 */
export function spatial2dPadding(
    x: Tensor, padding?: [[number, number], [number, number]],
    dataFormat?: DataFormat): Tensor {
  return tidy(() => {
    if (x.rank !== 4) {
      throw new ValueError(
          `temporalPadding expects input tensor to be 4-D, but received a ` +
          `${x.rank}-D tensor.`);
    }

    if (padding == null) {
      padding = [[1, 1], [1, 1]];
    }
    if (padding.length !== 2 || padding[0].length !== 2 ||
        padding[1].length !== 2) {
      throw new ValueError(
          'spatial2dPadding expects `padding` to be an Array of two Arrays, ' +
          'each of which is an Array of two integers.');
    }

    if (dataFormat == null) {
      dataFormat = imageDataFormat();
    }
    if (dataFormat !== 'channelsLast' && dataFormat !== 'channelsFirst') {
      throw new ValueError(
          `Unknown data format: ${dataFormat}. ` +
          `Supported data formats are 'channelsLast' and 'channelsFirst.`);
    }

    let pattern: Array<[number, number]>;
    if (dataFormat === 'channelsFirst') {
      pattern = [[0, 0], [0, 0], padding[0], padding[1]];
    } else {
      pattern = [[0, 0], padding[0], padding[1], [0, 0]];
    }

    return tfc.pad(x, pattern);
  });
}

export declare interface ZeroPadding2DLayerArgs extends LayerArgs {
  /**
   * Integer, or `Array` of 2 integers, or `Array` of 2 `Array`s, each of
   * which is an `Array` of 2 integers.
   * - If integer, the same symmetric padding is applied to width and height.
   * - If `Array` of 2 integers, interpreted as two different symmetric values
   *   for height and width:
   *   `[symmetricHeightPad, symmetricWidthPad]`.
   * - If `Array` of 2 `Array`s, interpreted as:
   *   `[[topPad, bottomPad], [leftPad, rightPad]]`.
   */
  padding?: number|[number, number]|[[number, number], [number, number]];

  /**
   * One of `'channelsLast'` (default) and `'channelsFirst'`.
   *
   * The ordering of the dimensions in the inputs.
   * `channelsLast` corresponds to inputs with shape
   * `[batch, height, width, channels]` while `channelsFirst`
   * corresponds to inputs with shape
   * `[batch, channels, height, width]`.
   */
  dataFormat?: DataFormat;
}

/** Normalize the three supported padding representations. */
function normalizeZeroPadding2D(
    padding: number|[number, number]|[[number, number], [number, number]]):
    [[number, number], [number, number]] {
  if (padding == null) {
    return [[1, 1], [1, 1]];
  }
  if (typeof padding === 'number') {
    return [[padding, padding], [padding, padding]];
  }
  if (padding.length !== 2) {
    throw new ValueError(
        `ZeroPadding2D expects padding to be a length-2 array, but ` +
        `received a length-${padding.length} array.`);
  }
  const [height, width] = padding;
  if (typeof height === 'number' && typeof width === 'number') {
    return [[height, height], [width, width]];
  }
  if (!Array.isArray(height) || !Array.isArray(width)) {
    throw new ValueError('ZeroPadding2D expects two numbers or two padding arrays.');
  }
  if (height.length !== 2) {
    throw new ValueError(
        `ZeroPadding2D expects height padding to be a length-2 array, ` +
        `but received a length-${height.length} array.`);
  }
  if (width.length !== 2) {
    throw new ValueError(
        `ZeroPadding2D expects width padding to be a length-2 array, ` +
        `but received a length-${width.length} array.`);
  }
  return [height, width];
}

/** Preserve unknown symbolic dimensions when applying spatial padding. */
function paddedDimension(dimension: number, padding: [number, number]): number {
  return dimension != null && dimension >= 0 ?
      dimension + padding[0] + padding[1] : null;
}

export class ZeroPadding2D extends Layer {
  /** @nocollapse */
  static className = 'ZeroPadding2D';
  readonly dataFormat: DataFormat;
  readonly padding: [[number, number], [number, number]];

  constructor(args?: ZeroPadding2DLayerArgs) {
    if (args == null) {
      args = {};
    }
    super(args);

    this.dataFormat = args.dataFormat ?? imageDataFormat();
    this.padding = normalizeZeroPadding2D(args.padding ?? null);
    this.inputSpec = [new InputSpec({ndim: 4})];
  }

  override computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);

    const batch = requiredElement(inputShape, 0);
    if (this.dataFormat === 'channelsFirst') {
      const channels = requiredElement(inputShape, 1);
      const rows = paddedDimension(inputShape[2] ?? null, this.padding[0]);
      const cols = paddedDimension(inputShape[3] ?? null, this.padding[1]);
      return [batch, channels, rows, cols];
    }
    const rows = paddedDimension(inputShape[1] ?? null, this.padding[0]);
    const cols = paddedDimension(inputShape[2] ?? null, this.padding[1]);
    return [batch, rows, cols, requiredElement(inputShape, 3)];
  }

  override call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(
        () => spatial2dPadding(
            getExactlyOneTensor(inputs), this.padding, this.dataFormat));
  }

  override getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      padding: this.padding,
      dataFormat: this.dataFormat,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(ZeroPadding2D);
