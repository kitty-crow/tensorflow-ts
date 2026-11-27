/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import type {DataType, MetaGraph, ModelTensorInfo, SignatureDefEntry} from '@tensorflow/tfjs-core';

const SAVED_MODEL_INIT_OP_KEY = '__saved_model_init_op';

// Generated jspb code is a JavaScript boundary. Validate the actual returned
// values instead of asserting a TypeScript type for the entire generated API.
function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function requireRecord(value: unknown, label: string): Record<string, unknown> {
  if (!isRecord(value)) {
    throw new Error(`Expected a protobuf object for ${label}`);
  }
  return value;
}

function readMethod(message: unknown, name: string, ...args: unknown[]): unknown {
  const record = requireRecord(message, name);
  const method = record[name];
  if (typeof method !== 'function') {
    throw new Error(`Missing protobuf method ${name}`);
  }
  return Reflect.apply(method, message, args);
}

function readArray(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`Expected a protobuf list for ${label}`);
  }
  return value;
}

function readString(value: unknown, label: string): string {
  if (typeof value !== 'string') {
    throw new Error(`Expected a string for ${label}`);
  }
  return value;
}

function readNumber(value: unknown, label: string): number {
  if (typeof value !== 'number') {
    throw new Error(`Expected a number for ${label}`);
  }
  return value;
}

function readMap<T>(message: unknown, readValue: (value: unknown) => T,
                    skip = ''): Record<string, T> {
  const iterator = readMethod(message, 'keys');
  const entries: Array<[string, T]> = [];
  // jspb's map keys() has a next() method, but is not necessarily an iterable.
  for (;;) {
    const next = requireRecord(readMethod(iterator, 'next'), 'map iterator');
    if (next.done === true) {
      break;
    }
    if (next.done !== false) {
      throw new Error('Invalid protobuf map iterator state');
    }
    const key = readString(next.value, 'map key');
    if (key !== skip) {
      entries.push([key, readValue(readMethod(message, 'get', key))]);
    }
  }
  return Object.fromEntries(entries);
}

/** Return the enum key matching the protobuf numeric value, if present. */
export function getEnumKeyFromValue(object: object, value: number): string|undefined {
  const entries: Array<[string, unknown]> = Object.entries(object);
  return entries.find(([, entry]) => entry === value)?.[0];
}

function mapTFDtypeToJSDtype(tfDtype: string): DataType {
  switch (tfDtype) {
    case 'DT_FLOAT': return 'float32';
    case 'DT_INT64':
    case 'DT_INT32':
    case 'DT_UINT8': return 'int32';
    case 'DT_BOOL': return 'bool';
    case 'DT_COMPLEX64': return 'complex64';
    case 'DT_STRING': return 'string';
    default:
      throw new Error(
          'Unsupported tensor DataType: ' + tfDtype +
          ', try to modify the model in python to convert the datatype');
  }
}

function readTensorInfo(message: unknown, dataTypes: object): ModelTensorInfo {
  const dtypeCode = readNumber(readMethod(message, 'getDtype'), 'tensor dtype');
  const tfDtype = getEnumKeyFromValue(dataTypes, dtypeCode);
  if (tfDtype == null) {
    throw new Error('Unsupported tensor DataType: undefined, try to modify the model in python to convert the datatype');
  }
  const dtype = mapTFDtypeToJSDtype(tfDtype);
  const name = readString(readMethod(message, 'getName'), 'tensor name');
  const shapeMessage = readMethod(message, 'getTensorShape');
  const dimensions = readArray(readMethod(shapeMessage, 'getDimList'), 'tensor dimensions');
  // getDimList() returns protobuf Dim objects, not numeric dimension sizes.
  const shape = dimensions.map(dim => readNumber(readMethod(dim, 'getSize'), 'dimension size'));
  return {dtype, tfDtype, name, shape};
}

/** Inspect the generated message without depending on the native TF binding. */
export function inspectSavedModelMetadata(message: unknown, rawDataTypes: unknown): MetaGraph[] {
  const dataTypes = requireRecord(rawDataTypes, 'DataType enum');
  const graphs = readArray(readMethod(message, 'getMetaGraphsList'), 'metagraphs');
  const readSignature = (value: unknown): SignatureDefEntry => ({
    inputs: readMap(readMethod(value, 'getInputsMap'), tensor => readTensorInfo(tensor, dataTypes)),
    outputs: readMap(readMethod(value, 'getOutputsMap'), tensor => readTensorInfo(tensor, dataTypes))
  });
  return graphs.map(graph => {
    const metaInfo = readMethod(graph, 'getMetaInfoDef');
    const tags = readArray(readMethod(metaInfo, 'getTagsList'), 'metagraph tags')
        .map(tag => readString(tag, 'metagraph tag'));
    const signatureDefs = readMap(readMethod(graph, 'getSignatureDefMap'),
        readSignature, SAVED_MODEL_INIT_OP_KEY);
    return {tags, signatureDefs};
  });
}
