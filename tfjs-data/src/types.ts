/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */

import {DataType, TensorContainer} from '@tensorflow/tfjs-core';

// Maybe this should be called 'NestedContainer'-- that's just a bit unwieldy.
export type Container<T> = ContainerObject<T>|ContainerArray<T>;

export type ContainerOrT<T> = Container<T>|T;

export interface ContainerObject<T> {
  [x: string]: ContainerOrT<T>;
}
export interface ContainerArray<T> extends Array<ContainerOrT<T>> {}

/**
 * Types supported by FileChunkIterator in both Browser and Node Environment.
 */
export type FileElement = File|Blob|Uint8Array;

/**
 * A dictionary containing column level configurations when reading and decoding
 * CSV file(s) from csv source.
 * Has the following fields:
 * - `required` If value in this column is required. If set to `true`, throw an
 * error when it finds an empty value.
 *
 * - `dtype` Data type of this column. Could be int32, float32, bool, or string.
 *
 * - `default` Default value of this column.
 *
 * - `isLabel` Whether this column is label instead of features. If isLabel is
 * `true` for at least one column, the .csv() API will return an array of two
 * items: the first item is a dict of features key/value pairs, the second item
 * is a dict of labels key/value pairs. If no column is marked as label returns
 * a dict of features only.
 */
export interface ColumnConfig {
  // CG305: CSV columns need not be marked required.
  //@cg-expect-error
  required?: boolean;
  // CG305: The CSV parser can infer a column's type without a dtype override.
  //@cg-expect-error
  dtype?: DataType;
  // CG305: A missing column value need not have a configured replacement.
  //@cg-expect-error
  default?: TensorContainer;
  // CG305: CSV columns are features unless marked as labels.
  //@cg-expect-error
  isLabel?: boolean;
}

/**
 * Interface for configuring dataset when reading and decoding from CSV file(s).
 */
export interface CSVConfig {
  /**
   * A boolean value that indicates whether the first row of provided CSV file
   * is a header line with column names, and should not be included in the data.
   */
  // CG305: CSV header handling can use its default when unspecified.
  //@cg-expect-error
  hasHeader?: boolean;

  /**
   * A list of strings that corresponds to the CSV column names, in order. If
   * provided, it ignores the column names inferred from the header row. If not
   * provided, infers the column names from the first row of the records. If
   * `hasHeader` is false and `columnNames` is not provided, this method will
   * throw an error.
   */
  // CG305: The CSV header supplies column names when no override is given.
  //@cg-expect-error
  columnNames?: string[];

  /**
   * A dictionary whose key is column names, value is an object stating if this
   * column is required, column's data type, default value, and if this column
   * is label. If provided, keys must correspond to names provided in
   * `columnNames` or inferred from the file header lines. If any column is
   * marked as label, the .csv() API will return an array of two items: the
   * first item is a dict of features key/value pairs, the second item is a dict
   * of labels key/value pairs. If no column is marked as label returns a dict
   * of features only.
   *
   * Has the following fields:
   * - `required` If value in this column is required. If set to `true`, throw
   * an error when it finds an empty value.
   *
   * - `dtype` Data type of this column. Could be int32, float32, bool, or
   * string.
   *
   * - `default` Default value of this column.
   *
   * - `isLabel` Whether this column is label instead of features. If isLabel is
   * `true` for at least one column, the element in returned `CSVDataset` will
   * be an object of {xs: features, ys: labels}: xs is a dict of features
   * key/value pairs, ys is a dict of labels key/value pairs. If no column is
   * marked as label, returns a dict of features only.
   */
  // CG305: CSV may be parsed without per-column configuration.
  //@cg-expect-error
  columnConfigs?: {[key: string]: ColumnConfig};

  /**
   * If true, only columns provided in `columnConfigs` will be parsed and
   * provided during iteration.
   */
  // CG305: CSV may include all columns without a configured-only filter.
  //@cg-expect-error
  configuredColumnsOnly?: boolean;

  /**
   * The string used to parse each line of the input file.
   */
  // CG305: The parser supports its default delimiter when unspecified.
  //@cg-expect-error
  delimiter?: string;

  /**
   * If true, delimiter field should be null. Parsing delimiter is whitespace
   * and treat continuous multiple whitespace as one delimiter.
   */
  // CG305: Whitespace parsing is an optional alternative to ordinary delimiters.
  //@cg-expect-error
  delimWhitespace?: boolean;
}

/**
 * Interface configuring data from webcam video stream.
 */
export interface WebcamConfig {
  /**
   * A string specifying which camera to use on device. If the value is
   * 'user', it will use front camera. If the value is 'environment', it will
   * use rear camera.
   */
  // CG305: Default camera selection does not require a facing mode.
  //@cg-expect-error
  facingMode?: 'user'|'environment';

  /**
   * A string used to request a specific camera. The deviceId can be obtained by
   * calling `mediaDevices.enumerateDevices()`.
   */
  // CG305: Default camera selection does not require a specific device identifier.
  //@cg-expect-error
  deviceId?: string;

  /**
   * Specifies the width of the output tensor. The actual width of the
   * HTMLVideoElement (if provided) can be different and the final image will be
   * resized to match resizeWidth.
   */
  // CG305: Native camera width can be retained without a resize request.
  //@cg-expect-error
  resizeWidth?: number;

  /**
   * Specifies the height of the output tensor. The actual height of the
   * HTMLVideoElement (if provided) can be different and the final image will be
   * resized to match resizeHeight.
   */
  // CG305: Native camera height can be retained without a resize request.
  //@cg-expect-error
  resizeHeight?: number;

  /**
   * A boolean value that indicates whether to crop the video frame from center.
   * If true, `resizeWidth` and `resizeHeight` must be specified; then an image
   * of size `[resizeWidth, resizeHeight]` is taken from the center of the frame
   * without scaling. If false, the entire image is returned (perhaps scaled to
   * fit in `[resizeWidth, resizeHeight]`, if those are provided).
   */
  // CG305: Camera capture need not request center cropping.
  //@cg-expect-error
  centerCrop?: boolean;
}

/**
 * Interface configuring data from microphone audio stream.
 */
export interface MicrophoneConfig {
  // A number representing Audio sampling rate in Hz. either 44,100 or 48,000.
  // If provided sample rate is not available on the device, it will throw an
  // error. Optional, defaults to the sample rate available on device.
  // CG305: The device's available sample rate is used when unspecified.
  //@cg-expect-error
  sampleRateHz?: 44100|48000;

  // The FFT length of each spectrogram column. A higher value will result in
  // more details in the frequency domain but fewer details in the time domain.
  // Must be a power of 2 between 2 to 4 and 2 to 14, so one of: 16, 32, 64,
  // 128, 256, 512, 1024, 2048, 4096, 8192, and 16384. It will throw an error if
  // it is an invalid number.Defaults to 1024.
  // CG305: MicrophoneIterator defaults FFT size to 1024.
  //@cg-expect-error
  fftSize?: number;

  // Truncate each spectrogram column at how many frequency points. Each audio
  // frame contains fftSize, for example, 1024 samples which covers voice
  // frequency from 0 to 22,500 Hz. However, the frequency content relevant to
  // human speech is generally in the frequency range from 0 to 5000 Hz. So each
  // audio frame only need 232 columns to cover the frequency range of human
  // voice. This will be part of the output spectrogram tensor shape. Optional,
  // defaults to null which means no truncation.
  // CG305: Omitted truncation length retains the full FFT width.
  //@cg-expect-error
  columnTruncateLength?: number;

  // Number of audio frames per spectrogram. The time duration of one
  // spectrogram equals to numFramesPerSpectrogram*fftSize/sampleRateHz second.
  // For example: the device sampling rate is 44,100 Hz, and fftSize is 1024,
  // then each frame duration between two sampling is 0.023 second. If the
  // purpose is for an audio model to recognize speech command that last 1
  // second, each spectrogram should contain 1/0.023, which is 43 frames. This
  // will be part of the output spectrogram tensor shape. Optional, defaults to
  // 43 so that each audio data last 1 second.
  // CG305: MicrophoneIterator defaults to 43 frames per spectrogram.
  //@cg-expect-error
  numFramesPerSpectrogram?: number;

  // A dictionary specifying the requirements of audio to request, such as
  // deviceID, echoCancellation, etc. Optional.
  // CG305: Default microphone capture requires no custom track constraints.
  //@cg-expect-error
  audioTrackConstraints?: MediaTrackConstraints;

  // The averaging constant with the last analysis frame -- basically, it makes
  // the transition between values over time smoother. It is used by
  // AnalyserNode interface during FFT. Optional, has to be between 0 and 1,
  // defaults to 0.
  // CG305: MicrophoneIterator defaults smoothing to zero.
  //@cg-expect-error
  smoothingTimeConstant?: number;

  // Whether to collect the frequency domain audio spectrogram in
  // MicrophoneIterator result. If both includeSpectrogram and includeWaveform
  // are false, it will throw an error. Defaults to true.
  // CG305: Spectrogram collection defaults to enabled.
  //@cg-expect-error
  includeSpectrogram?: boolean;

  // Whether to collect the time domain audio waveform in MicrophoneIterator
  // result. If both includeSpectrogram and includeWaveform are false, it will
  // throw an error. Defaults to false.
  // CG305: Waveform collection defaults to disabled.
  //@cg-expect-error
  includeWaveform?: boolean;
}
