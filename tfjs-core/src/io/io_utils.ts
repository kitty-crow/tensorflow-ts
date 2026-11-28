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
 * =============================================================================
 */

import {complex} from '../ops/complex';
import {tensor} from '../ops/tensor';
import {NamedTensor, NamedTensorMap} from '../tensor_types';
import {TypedArray} from '../types';
import {requiredNumericEntry, sizeFromShape} from '../util';

import {requiredWeightByteSize, ModelArtifacts, ModelArtifactsInfo, ModelJSON, WeightData, WeightGroup, WeightsManifestConfig, WeightsManifestEntry} from './types';
import {CompositeArrayBuffer} from './composite_array_buffer';
import {Tensor} from '../tensor';
import {backend} from '../globals';
import {DataId} from '../tensor_info';
import {env} from '../environment';
import {getBackend} from '../globals';

/** Number of bytes reserved for the length of the string. (32bit integer). */
const NUM_BYTES_STRING_LENGTH = 4;

/**
 * Encode a map from names to weight values as an ArrayBuffer, along with an
 * `Array` of `WeightsManifestEntry` as specification of the encoded weights.
 *
 * This function does not perform sharding.
 *
 * This function is the reverse of `decodeWeights`.
 *
 * @param tensors A map ("dict") from names to tensors.
 * @param group Group to which the weights belong (optional).
 * @returns A `Promise` of
 *   - A flat `ArrayBuffer` with all the binary values of the `Tensor`s
 *     concatenated.
 *   - An `Array` of `WeightManifestEntry`s, carrying information including
 *     tensor names, `dtype`s and shapes.
 * @throws Error: on unsupported tensor `dtype`.
 */
export async function encodeWeights(
    tensors: NamedTensorMap|NamedTensor[], group?: WeightGroup):
    Promise<{data: ArrayBuffer, specs: WeightsManifestEntry[]}> {
  // TODO(adarob, cais): Support quantization.
  const specs: WeightsManifestEntry[] = [];
  const dataPromises: Array<Promise<TypedArray>> = [];

  const entries: NamedTensor[] = Array.isArray(tensors) ?
      tensors : Object.entries(tensors).map(([name, tensor]) => ({name, tensor}));

  for (const {name, tensor: t} of entries) {
    if (t.dtype !== 'float32' && t.dtype !== 'int32' && t.dtype !== 'bool' &&
        t.dtype !== 'string' && t.dtype !== 'complex64') {
      throw new Error(`Unsupported dtype in weight '${name}': ${t.dtype}`);
    }
    const spec: WeightsManifestEntry = {name, shape: t.shape, dtype: t.dtype};
    if (t.dtype === 'string') {
      const utf8bytes = new Promise<TypedArray>(async resolve => {
        const vals = await t.bytes() as Uint8Array[];
        const totalNumBytes = vals.reduce((p, c) => p + c.length, 0) +
            NUM_BYTES_STRING_LENGTH * vals.length;
        const bytes = new Uint8Array(totalNumBytes);
        let offset = 0;
        for (const val of vals) {
          const bytesOfLength =
              new Uint8Array(new Uint32Array([val.length]).buffer);
          bytes.set(bytesOfLength, offset);
          offset += NUM_BYTES_STRING_LENGTH;
          bytes.set(val, offset);
          offset += val.length;
        }
        resolve(bytes);
      });
      dataPromises.push(utf8bytes);
    } else {
      dataPromises.push(t.data());
    }
    if (group != null) {
      spec.group = group;
    }
    specs.push(spec);
  }

  const tensorValues = await Promise.all(dataPromises);
  return {data: concatenateTypedArrays(tensorValues), specs};
}

/**
 * Decode flat ArrayBuffer as weights.
 *
 * This function does not handle sharding.
 *
 * This function is the reverse of `encodeWeights`.
 *
 * @param weightData A flat ArrayBuffer or an array of ArrayBuffers carrying the
 *   binary values of the tensors concatenated in the order specified in
 *   `specs`.
 * @param specs Specifications of the names, dtypes and shapes of the tensors
 *   whose value are encoded by `buffer`.
 * @return A map from tensor name to tensor value, with the names corresponding
 *   to names in `specs`.
 * @throws Error, if any of the tensors has unsupported dtype.
 */
export function decodeWeights(
    weightData: WeightData,
    specs: WeightsManifestEntry[]): NamedTensorMap {
  // TODO(adarob, cais): Support quantization.
  const compositeBuffer = new CompositeArrayBuffer(weightData);
  const out: NamedTensorMap = {};
  let offset = 0;
  for (const spec of specs) {
    const byteLength = getWeightBytelength(spec, (start, end) => {
      return compositeBuffer.slice(offset + start, offset + end);
    });
    out[spec.name] = decodeWeight(spec, compositeBuffer
      .slice(offset, offset + byteLength));
    offset += byteLength;
  }
  return out;
}

function getWeightBytelength(spec: WeightsManifestEntry,
  slice: (start: number, end: number) => ArrayBuffer): number {

  const size = sizeFromShape(spec.shape);
  if (!('quantization' in spec) && spec.dtype === 'string') {
    // Can not statically determine string length.
    let byteLength = 0;
    for (let i = 0; i < size; i++) {
      byteLength += NUM_BYTES_STRING_LENGTH + requiredNumericEntry(new Uint32Array(
        slice(byteLength, byteLength + NUM_BYTES_STRING_LENGTH)), 0);
    }
    return byteLength;
  }
  const dtype = 'quantization' in spec ?
      validateWeightQuantization(spec).dtype : spec.dtype;
  return size * requiredWeightByteSize(dtype, spec.name);
}

async function getWeightBytelengthAsync(
  spec: WeightsManifestEntry,
  slice: (start: number, end: number) => Promise<ArrayBuffer>
): Promise<number> {

  const size = sizeFromShape(spec.shape);
  if (!('quantization' in spec) && spec.dtype === 'string') {
    // Can not statically determine string length.
    let byteLength = 0;
    for (let i = 0; i < size; i++) {
      byteLength += NUM_BYTES_STRING_LENGTH + requiredNumericEntry(new Uint32Array(
        await slice(byteLength, byteLength + NUM_BYTES_STRING_LENGTH)), 0);
    }
    return byteLength;
  }
  const dtype = 'quantization' in spec ?
      validateWeightQuantization(spec).dtype : spec.dtype;
  return size * requiredWeightByteSize(dtype, spec.name);
}


/** Verify quantised metadata before reading weight bytes. */
type ValidWeightQuantization =
    {dtype: 'uint8'|'uint16', min: number, scale: number}|
    {dtype: 'float16'};

function validateWeightQuantization(spec: WeightsManifestEntry):
    ValidWeightQuantization {
  const quantization = spec.quantization;
  if (quantization == null) {
    throw new Error(`Weight ${spec.name} is missing quantization metadata.`);
  }
  const {dtype, min, scale} = quantization;
  const integerQuantization = dtype === 'uint8' || dtype === 'uint16';
  if (integerQuantization &&
      (typeof min !== 'number' || typeof scale !== 'number')) {
    throw new Error(
        `Weight ${spec.name} with quantization ${dtype} ` +
        `doesn't have corresponding metadata min and scale.`);
  }
  if (integerQuantization && typeof min === 'number' &&
      typeof scale === 'number') {
    return {dtype, min, scale};
  }
  if (dtype === 'float16' && spec.dtype !== 'float32') {
    throw new Error(
        `Weight ${spec.name} is quantized with ${dtype} ` +
        `which only supports weights of type float32 not ${spec.dtype}.`);
  }
  if (dtype === 'float16') {
    return {dtype};
  }
  throw new Error(
      `Weight ${spec.name} has unknown ` +
      `quantization dtype ${dtype}. ` +
      `Supported quantization dtypes are: ` +
      `'uint8', 'uint16', and 'float16'.`);
}

/** Decode quantized weights after validating their wire metadata. */
function decodeQuantizedWeight(
    spec: WeightsManifestEntry, byteBuffer: ArrayBuffer): Tensor {
  const quantization = validateWeightQuantization(spec);
  const quantizedArray = quantization.dtype === 'uint8' ?
      new Uint8Array(byteBuffer) : new Uint16Array(byteBuffer);
  if (spec.dtype === 'float32' && quantization.dtype === 'float16') {
    return tensor(getFloat16Decoder()(quantizedArray as Uint16Array),
        spec.shape, spec.dtype);
  }
  if (spec.dtype === 'int32' && quantization.dtype === 'float16') {
    throw new Error(
        `Unsupported quantization type ${quantization.dtype} ` +
        'for weight type int32.');
  }
  if (spec.dtype !== 'float32' && spec.dtype !== 'int32') {
    throw new Error(`Unsupported dtype in weight '${spec.name}': ${spec.dtype}`);
  }
  if (quantization.dtype === 'float16') {
    throw new Error(`Unsupported quantization type ${quantization.dtype}.`);
  }
  const values = spec.dtype === 'float32' ?
      new Float32Array(quantizedArray.length) :
      new Int32Array(quantizedArray.length);
  for (let i = 0; i < quantizedArray.length; i++) {
    const value = requiredNumericEntry(quantizedArray, i) *
        quantization.scale + quantization.min;
    values[i] = spec.dtype === 'int32' ? Math.round(value) : value;
  }
  return tensor(values, spec.shape, spec.dtype);
}

function decodeStringWeight(
    spec: WeightsManifestEntry, byteBuffer: ArrayBuffer): Tensor {
  const size = sizeFromShape(spec.shape);
  const values: Uint8Array[] = [];
  let offset = 0;
  for (let i = 0; i < size; i++) {
    const byteLength = requiredNumericEntry(new Uint32Array(
        byteBuffer.slice(offset, offset + NUM_BYTES_STRING_LENGTH)), 0);
    offset += NUM_BYTES_STRING_LENGTH;
    values.push(new Uint8Array(byteBuffer.slice(offset, offset + byteLength)));
    offset += byteLength;
  }
  return tensor(values, spec.shape, 'string');
}

function decodeWeight(
    spec: WeightsManifestEntry, byteBuffer: ArrayBuffer): Tensor {
  const {name, dtype, shape} = spec;
  if ('quantization' in spec) {
    return decodeQuantizedWeight(spec, byteBuffer);
  }
  if (dtype === 'string') {
    return decodeStringWeight(spec, byteBuffer);
  }
  switch (dtype) {
    case 'float32':
      return tensor(new Float32Array(byteBuffer), shape, dtype);
    case 'int32':
      return tensor(new Int32Array(byteBuffer), shape, dtype);
    case 'bool':
      return tensor(new Uint8Array(byteBuffer), shape, dtype);
    case 'complex64': {
      const values = new Float32Array(byteBuffer);
      const real = new Float32Array(values.length / 2);
      const image = new Float32Array(values.length / 2);
      for (let i = 0; i < real.length; i++) {
        real[i] = requiredNumericEntry(values, i * 2);
        image[i] = requiredNumericEntry(values, i * 2 + 1);
      }
      const realTensor = tensor(real, shape, 'float32');
      const imageTensor = tensor(image, shape, 'float32');
      const complexTensor = complex(realTensor, imageTensor);
      realTensor.dispose();
      imageTensor.dispose();
      return complexTensor;
    }
    default:
      throw new Error(`Unsupported dtype in weight '${name}': ${dtype}`);
  }
}

async function readToLength(reader: ReadableStreamDefaultReader<ArrayBuffer>,
                            initialData: ArrayBuffer,
                            length: number): Promise<ArrayBuffer> {
  let data = new Uint8Array(initialData);

  while (data.byteLength < length) {
    const {done, value} = await reader.read();
    if (done && value == null) {
      const missing  = length - data.byteLength;
      throw new Error(`Reader is done but ${missing} bytes are still expected`);
    }

    if (value == null) {
      throw new Error('Reader returned no data before completion');
    }

    // TODO: Don't create a new array every loop.
    const newData = new Uint8Array(data.length + value.byteLength);
    newData.set(data, 0);
    newData.set(new Uint8Array(value), data.length);
    data = newData;
  }

  return data.buffer;
}

export async function decodeWeightsStream(
  weightStream: ReadableStream<ArrayBuffer>,
  specs: WeightsManifestEntry[]): Promise<NamedTensorMap> {

  const tensors: NamedTensorMap = {};
  const reader = weightStream.getReader();
  let data = new ArrayBuffer(0);

  for (const spec of specs) {
    const byteLength = await getWeightBytelengthAsync(spec,
                                                      async (start, end) => {
      data = await readToLength(reader, data, end);
      return data.slice(start, end);
    });
    data = await readToLength(reader, data, byteLength);

    // Slice the tensor out
    const tensorData = data.slice(0, byteLength);
    data = data.slice(byteLength);

    const weightTensor = decodeWeight(spec, tensorData);
    tensors[spec.name] = weightTensor;

    // TODO(mattsoulanille): Better way to call uploadToGPU.
    // TODO(mattsoulanille): Make this work for webgl too.
    if (!(getBackend() === 'webgpu')) continue;
    {
      const b = backend();

      if ('uploadToGPU' in b &&
        sizeFromShape(weightTensor.shape) >= (env()
          .get('WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD') as number)) {
        (b.uploadToGPU as (dataId: DataId) => void)(weightTensor.dataId);
      }
    }
  }

  return tensors;
}

/**
 * Concatenate TypedArrays into an ArrayBuffer.
 */
export function concatenateTypedArrays(xs: TypedArray[]): ArrayBuffer {
  // TODO(adarob, cais): Support quantization.
  if (xs === null) {
    throw new Error(`Invalid input value: ${JSON.stringify(xs)}`);
  }

  let totalByteLength = 0;

  for (const x of xs) {
    const constructor = x.constructor;
    if (!(x instanceof Float32Array || x instanceof Int32Array ||
          x instanceof Uint8Array)) {
      throw new Error(`Unsupported TypedArray subtype: ${constructor.name}`);
    }
    totalByteLength += x.byteLength;
  }

  // Copy only the bytes belonging to each view, not its entire backing buffer.
  const y = new Uint8Array(totalByteLength);
  let offset = 0;
  for (const x of xs) {
    y.set(new Uint8Array(x.buffer, x.byteOffset, x.byteLength), offset);
    offset += x.byteLength;
  }

  return y.buffer;
}

// Use Buffer on Node.js instead of Blob/atob/btoa
const useNodeBuffer = typeof Buffer !== 'undefined' &&
    (typeof Blob === 'undefined' || typeof atob === 'undefined' ||
     typeof btoa === 'undefined');

/**
 * Calculate the byte length of a JavaScript string.
 *
 * Note that a JavaScript string can contain wide characters, therefore the
 * length of the string is not necessarily equal to the byte length.
 *
 * @param str Input string.
 * @returns Byte length.
 */
export function stringByteLength(str: string): number {
  if (useNodeBuffer) {
    return Buffer.byteLength(str, 'utf8');
  }
  return new Blob([str]).size;
}

/**
 * Encode an ArrayBuffer as a base64 encoded string.
 *
 * @param buffer `ArrayBuffer` to be converted.
 * @returns A string that base64-encodes `buffer`.
 */
export function arrayBufferToBase64String(buffer: ArrayBuffer): string {
  if (useNodeBuffer) {
    return Buffer.from(buffer).toString('base64');
  }
  const buf = new Uint8Array(buffer);
  let s = '';
  for (let i = 0; i < buf.length; i++) {
    s += String.fromCharCode(requiredNumericEntry(buf, i));
  }
  return btoa(s);
}

/**
 * Decode a base64 string as an ArrayBuffer.
 *
 * @param str Base64 string.
 * @returns Decoded `ArrayBuffer`.
 */
export function base64StringToArrayBuffer(str: string): ArrayBuffer {
  if (useNodeBuffer) {
    const buf = Buffer.from(str, 'base64');
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }
  const s = atob(str);
  const buffer = new Uint8Array(s.length);
  for (let i = 0; i < s.length; ++i) {
    buffer.set([s.charCodeAt(i)], i);
  }
  return buffer.buffer;
}

/**
 * Concatenate a number of ArrayBuffers into one.
 *
 * @param buffers An array of ArrayBuffers to concatenate, or a single
 *     ArrayBuffer.
 * @returns Result of concatenating `buffers` in order.
 *
 * @deprecated Use tf.io.CompositeArrayBuffer.join() instead.
 */
export function concatenateArrayBuffers(buffers: ArrayBuffer[]
      | ArrayBuffer): ArrayBuffer {
  return CompositeArrayBuffer.join(buffers);
}

/**
 * Get the basename of a path.
 *
 * Behaves in a way analogous to Linux's basename command.
 *
 * @param path
 */
export function basename(path: string): string {
  const SEPARATOR = '/';
  path = path.trim();
  while (path.endsWith(SEPARATOR)) {
    path = path.slice(0, path.length - 1);
  }
  return path.slice(path.lastIndexOf(SEPARATOR) + 1);
}

/**
 * Create `ModelJSON` from `ModelArtifacts`.
 *
 * @param artifacts Model artifacts, describing the model and its weights.
 * @param manifest Weight manifest, describing where the weights of the
 *     `ModelArtifacts` are stored, and some metadata about them.
 * @returns Object representing the `model.json` file describing the model
 *     artifacts and weights
 */
export function getModelJSONForModelArtifacts(
    artifacts: ModelArtifacts, manifest: WeightsManifestConfig): ModelJSON {
  const result: ModelJSON = {
    modelTopology: artifacts.modelTopology,
    format: artifacts.format,
    generatedBy: artifacts.generatedBy,
    convertedBy: artifacts.convertedBy,
    weightsManifest: manifest
  };
  if (artifacts.signature != null) {
    result.signature = artifacts.signature;
  }
  if (artifacts.userDefinedMetadata != null) {
    result.userDefinedMetadata = artifacts.userDefinedMetadata;
  }
  if (artifacts.modelInitializer != null) {
    result.modelInitializer = artifacts.modelInitializer;
  }
  if (artifacts.initializerSignature != null) {
    result.initializerSignature = artifacts.initializerSignature;
  }
  if (artifacts.trainingConfig != null) {
    result.trainingConfig = artifacts.trainingConfig;
  }
  return result;
}

/**
 * Create `ModelArtifacts` from a JSON file and weights.
 *
 * @param modelJSON Object containing the parsed JSON of `model.json`
 * @param weightSpecs The list of WeightsManifestEntry for the model. Must be
 *     passed if the modelJSON has a weightsManifest.
 * @param weightData An ArrayBuffer or array of ArrayBuffers of weight data for
 *     the model corresponding to the weights in weightSpecs. Must be passed if
 *     the modelJSON has a weightsManifest.
 * @returns A Promise of the `ModelArtifacts`, as described by the JSON file.
 */
export function getModelArtifactsForJSONSync(
    modelJSON: ModelJSON, weightSpecs?: WeightsManifestEntry[],
    weightData?: WeightData): ModelArtifacts {

  const modelArtifacts: ModelArtifacts = {
    modelTopology: modelJSON.modelTopology,
    format: modelJSON.format,
    generatedBy: modelJSON.generatedBy,
    convertedBy: modelJSON.convertedBy
  };

  if (modelJSON.trainingConfig != null) {
    modelArtifacts.trainingConfig = modelJSON.trainingConfig;
  }
  const hasWeightsManifest = modelJSON.weightsManifest != null;
  if (hasWeightsManifest && !weightSpecs) {
    throw new Error('modelJSON has weightsManifest but weightSpecs is null');
  }
  if (hasWeightsManifest && !weightData) {
    throw new Error('modelJSON has weightsManifest but weightData is null');
  }
  if (hasWeightsManifest) {
    modelArtifacts.weightSpecs = weightSpecs;
    modelArtifacts.weightData = weightData;
  }
  if (modelJSON.signature != null) {
    modelArtifacts.signature = modelJSON.signature;
  }
  if (modelJSON.userDefinedMetadata != null) {
    modelArtifacts.userDefinedMetadata = modelJSON.userDefinedMetadata;
  }
  if (modelJSON.modelInitializer != null) {
    modelArtifacts.modelInitializer = modelJSON.modelInitializer;
  }
  if (modelJSON.initializerSignature != null) {
    modelArtifacts.initializerSignature = modelJSON.initializerSignature;
  }

  return modelArtifacts;
}

/**
 * Create `ModelArtifacts` from a JSON file.
 *
 * @param modelJSON Object containing the parsed JSON of `model.json`
 * @param loadWeights Function that takes the JSON file's weights manifest,
 *     reads weights from the listed path(s), and returns a Promise of the
 *     weight manifest entries along with the weights data.
 * @returns A Promise of the `ModelArtifacts`, as described by the JSON file.
 */
export async function getModelArtifactsForJSON(
    modelJSON: ModelJSON,
    loadWeights: (weightsManifest: WeightsManifestConfig) => Promise<[
      /* weightSpecs */ WeightsManifestEntry[], WeightData,
    ]>): Promise<ModelArtifacts> {
  let weightSpecs: WeightsManifestEntry[] | undefined;
  let weightData: WeightData | undefined;

  if (modelJSON.weightsManifest != null) {
    [weightSpecs, weightData] = await loadWeights(modelJSON.weightsManifest);
  }

  return getModelArtifactsForJSONSync(modelJSON, weightSpecs, weightData);
}

/**
 * Populate ModelArtifactsInfo fields for a model with JSON topology.
 * @param modelArtifacts
 * @returns A ModelArtifactsInfo object.
 */
export function getModelArtifactsInfoForJSON(modelArtifacts: ModelArtifacts):
    ModelArtifactsInfo {
  if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
    throw new Error('Expected JSON model topology, received ArrayBuffer.');
  }

  return {
    dateSaved: new Date(),
    modelTopologyType: 'JSON',
    modelTopologyBytes: modelArtifacts.modelTopology == null ?
        0 :
        stringByteLength(JSON.stringify(modelArtifacts.modelTopology)),
    weightSpecsBytes: modelArtifacts.weightSpecs == null ?
        0 :
        stringByteLength(JSON.stringify(modelArtifacts.weightSpecs)),
    weightDataBytes: modelArtifacts.weightData == null ?
        0 :
        new CompositeArrayBuffer(modelArtifacts.weightData).byteLength,
  };
}

/**
 * Concatenate the weights stored in a WeightsManifestConfig into a list of
 * WeightsManifestEntry
 *
 * @param weightsManifest The WeightsManifestConfig to extract weights from.
 * @returns A list of WeightsManifestEntry of the weights in the weightsManifest
 */
export function getWeightSpecs(weightsManifest: WeightsManifestConfig):
    WeightsManifestEntry[] {
  const weightSpecs: WeightsManifestEntry[] = [];
  for (const entry of weightsManifest) {
    weightSpecs.push(...entry.weights);
  }
  return weightSpecs;
}

/**
 * Computes mantisa table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 2048 mantissa lookup values.
 */
function computeFloat16MantisaTable(): Uint32Array {
  const convertMantissa = (i: number): number => {
    let m = i << 13;
    let e = 0;

    while ((m & 0x00800000) === 0) {
      e -= 0x00800000;
      m <<= 1;
    }
    m &= ~0x00800000;
    e += 0x38800000;

    return m | e;
  };

  const mantisaTable = new Uint32Array(2048);

  mantisaTable[0] = 0;
  for (let i = 1; i < 1024; i++) {
    mantisaTable[i] = convertMantissa(i);
  }
  for (let i = 1024; i < 2048; i++) {
    mantisaTable[i] = 0x38000000 + ((i - 1024) << 13);
  }

  return mantisaTable;
}

/**
 * Computes exponent table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 64 exponent lookup values.
 */
function computeFloat16ExponentTable(): Uint32Array {
  const exponentTable = new Uint32Array(64);

  exponentTable[0] = 0;
  exponentTable[31] = 0x47800000;
  exponentTable[32] = 0x80000000;
  exponentTable[63] = 0xc7800000;
  for (let i = 1; i < 31; i++) {
    exponentTable[i] = i << 23;
  }
  for (let i = 33; i < 63; i++) {
    exponentTable[i] = 0x80000000 + ((i - 32) << 23);
  }

  return exponentTable;
}

/**
 * Computes offset table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 6d offset values.
 */
function computeFloat16OffsetTable(): Uint32Array {
  const offsetTable = new Uint32Array(64);

  for (let i = 0; i < 64; i++) {
    offsetTable[i] = 1024;
  }
  offsetTable[0] = offsetTable[32] = 0;

  return offsetTable;
}

/**
 * Retrieve a Float16 decoder which will decode a ByteArray of Float16 values
 * to a Float32Array.
 *
 * @returns Function (buffer: Uint16Array) => Float32Array which decodes
 *          the Uint16Array of Float16 bytes to a Float32Array.
 */
export function getFloat16Decoder(): (buffer: Uint16Array) => Float32Array {
  // Algorithm is based off of
  // http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf

  // Cache lookup tables
  const mantisaTable = computeFloat16MantisaTable();
  const exponentTable = computeFloat16ExponentTable();
  const offsetTable = computeFloat16OffsetTable();

  return (quantizedArray: Uint16Array) => {
    const buffer = new ArrayBuffer(4 * quantizedArray.length);
    const bufferUint32View = new Uint32Array(buffer);
    for (let index = 0; index < quantizedArray.length; index++) {
      const float16Bits = requiredNumericEntry(quantizedArray, index);
      const float32Bits =
          requiredNumericEntry(mantisaTable,
              requiredNumericEntry(offsetTable, float16Bits >> 10) + (float16Bits & 0x3ff)) +
          requiredNumericEntry(exponentTable, float16Bits >> 10);
      bufferUint32View[index] = float32Bits;
    }
    return new Float32Array(buffer);
  };
}
