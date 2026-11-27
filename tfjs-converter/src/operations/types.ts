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
import {Tensor, util} from '@tensorflow/tfjs-core';
// tslint:disable-next-line:no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import * as tensorflow from '../data/compiled_api';
import {NamedTensorsMap} from '../data/types';
import {ExecutionContext} from '../executor/execution_context';
import {ResourceManager} from '../executor/resource_manager';

export type ParamType = 'number'|'string'|'string[]'|'number[]'|'bool'|'bool[]'|
    'shape'|'shape[]'|'tensor'|'tensors'|'dtype'|'dtype[]'|'func';
export type Category = 'arithmetic'|'basic_math'|'control'|'convolution'|
    'creation'|'custom'|'dynamic'|'evaluation'|'graph'|'hash_table'|'image'|
    'logical'|'matrices'|'normalization'|'ragged'|'reduction'|'slice_join'|
    'sparse'|'spectral'|'string'|'transformation';

// For mapping input or attributes of NodeDef into TensorFlow.js op param.
export declare interface ParamMapper {
  // tensorflow.js name for the field, it should be in camelcase format.
  name: string;
  type: ParamType;
  // CG305: Some mapped attributes have no default and are supplied by the graph.
  //@cg-expect-error
  defaultValue?: AttributeValue;
  // CG305: Supported parameters omit the not-supported marker.
  //@cg-expect-error
  notSupported?: boolean;
}

// For mapping the input of TensorFlow NodeDef into TensorFlow.js Op param.
export declare interface InputParamMapper extends ParamMapper {
  // The first number is the starting index of the param, the second number is
  // the length of the param. If the length value is positive number, it
  // represents the true length of the param. Otherwise, it represents a
  // variable length, the value is the index go backward from the end of the
  // array.
  // For example `[0, 5]`: this param is the array of input tensors starting at
  // index 0 and with the length of 5.
  // For example `[1, -1]`: this param is the array of input tensors starting at
  // index 1 and with the `inputs.length - 1`.
  // Zero-based index at where in the input array this param starts.
  // A negative index can be used, indicating an offset from the end of the
  // sequence. slice(-2) extracts the last two elements in the sequence.
  start: number;
  // Zero-based index before where in the input array the param ends. The
  // mapping is up to but not including end. For example, start = 1, end = 4
  // includes the second element through the fourth element (elements indexed 1,
  // 2, and 3). A negative index can be used, indicating an offset from the end
  // of the sequence. start = 2, end = -1 includes the third element through the
  // second-to-last element in the sequence. If end is omitted, end is set to
  // start + 1, the mapping only include the single element at start index. If
  // end is set to 0, the mapping is through the end of the input array
  // (arr.length). If end is greater than the length of the inputs, mapping
  // inncludes through to the end of the sequence (arr.length).
  // CG305: The executor distinguishes an omitted end from a ranged input mapping.
  //@cg-expect-error
  end?: number;
}

// For mapping the attributes of TensorFlow NodeDef into TensorFlow.js op param.
export declare interface AttrParamMapper extends ParamMapper {
  // TensorFlow attribute name, this should be set if the tensorflow attribute
  // name is different form the tensorflow.js name.
  // CG305: The mapper permits the same TensorFlow and TensorFlow.js attribute name.
  //@cg-expect-error
  tfName?: string;
  // TensorFlow deprecated attribute name, this is used to support old models.
  // CG305: An attribute need not have a deprecated alternate name.
  //@cg-expect-error
  tfDeprecatedName?: string;
  // Preserve case for case-sensitive string attributes.
  // CG305: Case preservation is opt-in for case-sensitive string attributes.
  //@cg-expect-error
  keepCase?: boolean;
}

export interface InternalOpExecutor {
  (node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
   // CG305: The executor supports its default ops implementation without injection.
   //@cg-expect-error
   ops?: typeof tfOps): Tensor|Tensor[];
}

export interface InternalOpAsyncExecutor {
  (node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
   resourceManager?: ResourceManager, ops?: typeof tfOps): Promise<Tensor[]>;
}

export declare interface OpMapper {
  tfOpName: string;
  // CG305: Custom mappings need not carry the built-in operation category.
  //@cg-expect-error
  category?: Category;
  // CG305: An operation can have no mapped input parameters.
  //@cg-expect-error
  inputs?: InputParamMapper[];
  // CG305: An operation can have no mapped attributes.
  //@cg-expect-error
  attrs?: AttrParamMapper[];
  // CG305: An operation need not expose named outputs.
  //@cg-expect-error
  outputs?: string[];
  // CG305: Built-in operations need no custom executor.
  //@cg-expect-error
  customExecutor?: OpExecutor;
}

export declare interface Node {
  // CG305: Graph nodes not exposed by a signature have no signature key.
  //@cg-expect-error
  signatureKey?: string;
  name: string;
  op: string;
  category: Category;
  inputNames: string[];
  inputs: Node[];
  inputParams: {[key: string]: InputParamValue};
  attrParams: {[key: string]: ParamValue};
  children: Node[];
  // CG305: Only nodes that require raw graph attributes retain them.
  //@cg-expect-error
  rawAttrs?: {[k: string]: tensorflow.IAttrValue};
  // CG305: Nodes without a selected output do not need a default output index.
  //@cg-expect-error
  defaultOutput?: number;
  // CG305: Nodes without named outputs may omit the output names.
  //@cg-expect-error
  outputs?: string[];
}

export declare interface Graph {
  nodes: {[key: string]: Node};
  placeholders: Node[];
  inputs: Node[];
  outputs: Node[];
  weights: Node[];
  // CG305: Graph execution supports graphs without an exported signature.
  //@cg-expect-error
  signature?: tensorflow.ISignatureDef;
  // CG305: Graphs without function definitions omit the function map.
  //@cg-expect-error
  functions?: {[key: string]: Graph};
  // CG305: Graphs without initialisation operations omit init nodes.
  //@cg-expect-error
  initNodes?: Node[];
}

/** Values representable by the graph attribute schema. */
export type AttributeValue =
    string|string[]|number|number[]|number[][]|boolean|boolean[]|Tensor|Tensor[];

/** Resolved op inputs may additionally contain numeric arrays of arbitrary rank. */
export type ValueType = AttributeValue|util.NestedNumericArray;
export declare interface ParamValue {
  // CG305: The mapper represents absent optional attribute values without a value property.
  //@cg-expect-error
  value?: AttributeValue;
  type: ParamType;
}

export declare interface InputParamValue extends ParamValue {
  // CG305: Attribute-only or unmapped input parameters may omit an input start.
  //@cg-expect-error
  inputIndexStart?: number;
  // CG305: Single-input mappings omit the range end.
  //@cg-expect-error
  inputIndexEnd?: number;
}

export interface OpExecutor {
  (node: GraphNode): Tensor|Tensor[]|Promise<Tensor|Tensor[]>;
}

export interface GraphNode {
  inputs: Tensor[];
  attrs: {[key: string]: AttributeValue};
}
