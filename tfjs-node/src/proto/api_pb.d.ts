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

// Read-side declarations for the generated CommonJS protobuf API used by
// SavedModel. The method names and types match api.proto and api_pb.js.
// The runtime metadata parser still validates data at the boundary.
export interface ProtoMap<T> {
  has(key: string): boolean;
  get(key: string): T|undefined;
  getLength(): number;
  keys(): {next(): {done: false, value: string}|{done: true}};
}

export interface ProtoTensorInfo {
  getName(): string;
  getDtype(): number;
}

export interface ProtoSignatureDef {
  getInputsMap(): ProtoMap<ProtoTensorInfo>;
  getOutputsMap(): ProtoMap<ProtoTensorInfo>;
}

export interface ProtoMetaInfo {
  getTagsList(): string[];
}

export interface ProtoMetaGraph {
  getMetaInfoDef(): ProtoMetaInfo|null|undefined;
  getSignatureDefMap(): ProtoMap<ProtoSignatureDef>;
}

export interface ProtoSavedModel {
  getMetaGraphsList(): ProtoMetaGraph[];
}

export const SavedModel: {deserializeBinary(bytes: Uint8Array): ProtoSavedModel};
export const DataType: Readonly<Record<string, number>>;
