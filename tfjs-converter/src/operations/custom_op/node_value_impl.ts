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

import {DataType, Tensor} from '@tensorflow/tfjs-core';
import * as tensorflow from '../../data/compiled_api';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {getTensor} from '../executors/utils';
import {getBoolArrayParam, getBoolParam, getDtypeArrayParam, getDtypeParam, getNumberParam, getNumericArrayParam, getStringArrayParam, getStringParam, getTensorShapeArrayParam, getTensorShapeParam} from '../operation_mapper';
import {GraphNode, Node, AttributeValue} from '../types';

/**
 * Helper class for lookup inputs and params for nodes in the model graph.
 */
export class NodeValueImpl implements GraphNode {
  public readonly inputs: Tensor[] = [];
  public readonly attrs: {[key: string]: AttributeValue} = {};
  constructor(
      node: Node, private tensorMap: NamedTensorsMap,
      private context: ExecutionContext) {
    this.inputs = node.inputNames.map(name => this.getInput(name));
    const rawAttrs = node.rawAttrs;
    if (rawAttrs != null) {
      this.attrs = Object.keys(rawAttrs)
                       .reduce((attrs: {[key: string]: AttributeValue}, key) => {
                         attrs[key] = this.getAttr(rawAttrs, key);
                         return attrs;
                       }, {});
    }
  }

  /**
   * Return the value of the attribute or input param.
   * @param name String: name of attribute or input param.
   */
  private getInput(name: string): Tensor {
    return getTensor(name, this.tensorMap, this.context);
  }

  /**
   * Return the value of the attribute or input param.
   * @param name String: name of attribute or input param.
   */
  private getAttr(
      rawAttrs: {[key: string]: tensorflow.IAttrValue}, name: string,
      defaultValue?: AttributeValue): AttributeValue {
    const value = rawAttrs[name];
    if (value == null) {
      throw new Error(`Missing raw attribute '${name}'.`);
    }
    if (value.tensor != null) {
      return getTensor(name, this.tensorMap, this.context);
    }
    if (value.i != null || value.f != null) {
      return getNumberParam(rawAttrs, name, defaultValue as number);
    }
    if (value.s != null) {
      return getStringParam(rawAttrs, name, defaultValue as string);
    }
    if (value.b != null) {
      return getBoolParam(rawAttrs, name, defaultValue as boolean);
    }
    if (value.shape != null) {
      return getTensorShapeParam(
          rawAttrs, name, defaultValue as number[]);
    }
    if (value.type != null) {
      return getDtypeParam(rawAttrs, name, defaultValue as DataType);
    }
    // List attributes are the final supported representation.
    if (value.list == null) {
      return defaultValue;
    }
    if (value.list.i != null || value.list.f != null) {
      return getNumericArrayParam(
          rawAttrs, name, defaultValue as number[]);
    }
    if (value.list.s != null) {
      return getStringArrayParam(
          rawAttrs, name, defaultValue as string[]);
    }
    if (value.list.shape != null) {
      return getTensorShapeArrayParam(
          rawAttrs, name, defaultValue as number[][]);
    }
    if (value.list.b != null) {
      return getBoolArrayParam(
          rawAttrs, name, defaultValue as boolean[]);
    }
    if (value.list.type != null) {
      return getDtypeArrayParam(
          rawAttrs, name, defaultValue as DataType[]);
    }

    return defaultValue;
  }
}
