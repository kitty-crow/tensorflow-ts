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

import * as tensorflow from '../../data/compiled_api';
import {ExecutionContext} from '../../executor/execution_context';
import {Graph, Node} from '../types';
import {getExecutionSubgraph, getNodesInTopologicalOrder, getNodeLiveUntilMap} from '../../executor/model_analysis';
import {GraphExecutor} from '../../executor/graph_executor';

import {NodeValueImpl} from './node_value_impl';
import {getPadding, parseNodeName} from '../executors/utils';

describe('custom node attribute conditional dispatch', () => {
  it('preserves scalar/list precedence and all list representations', () => {
    const node: Node = {
      name: 'attributes', op: 'const', category: 'custom',
      inputNames: [], inputs: [], inputParams: {}, attrParams: {}, children: [],
      rawAttrs: {
        scalarFirst: {i: 7, list: {i: [1]}},
        numericFirst: {list: {i: [3, 6, 0], s: ['TkhXQw==']}},
        floats: {list: {f: [1.5, 2.5]}},
        strings: {list: {s: ['TkhXQw==']}},
        bools: {list: {b: [true, false]}},
        shapes: {list: {shape: [{dim: [{size: 2}, {size: 3}]}]}},
        types: {list: {type: [tensorflow.DataType.DT_INT32]}},
        empty: {},
        emptyList: {list: {}}
      }
    };
    const result = new NodeValueImpl(node, {}, new ExecutionContext({}, {}, {}));
    expect(result.attrs['scalarFirst']).toBe(7);
    expect(result.attrs['numericFirst']).toEqual([3, 6, 0]);
    expect(result.attrs['floats']).toEqual([1.5, 2.5]);
    expect(result.attrs['strings']).toEqual(['nhwc']);
    expect(result.attrs['bools']).toEqual([true, false]);
    expect(result.attrs['shapes']).toEqual([[2, 3]]);
    expect(result.attrs['types']).toEqual(['int32']);
    expect(result.attrs['empty']).toBeUndefined();
    expect(result.attrs['emptyList']).toBeUndefined();
  });
});

describe('graph executor map contracts', () => {
  it('retains function signatures and ordered input/output names', () => {
    const input: Node = {name: 'first', op: 'Placeholder', category: 'graph',
      inputNames: [], inputs: [], inputParams: {}, attrParams: {}, children: [],
      signatureKey: 'feature'};
    const output: Node = {...input, name: 'second', signatureKey: 'result',
      defaultOutput: 1};
    const signature = {inputs: {}, outputs: {}};
    const functionGraph: Graph = {nodes: {}, placeholders: [], inputs: [],
      outputs: [], weights: [], signature};
    const graph: Graph = {nodes: {first: input, second: output},
      placeholders: [input], inputs: [input], outputs: [output], weights: [],
      functions: {inner: functionGraph}};
    const executor = new GraphExecutor(graph);
    expect(executor.inputNodes).toEqual(['feature']);
    expect(executor.outputNodes).toEqual(['result:1']);
    expect(executor.functions).toEqual({inner: signature});
    executor.weightMap = {};
    expect(executor.weightIds).toEqual([]);
    expect(() => executor.dispose()).not.toThrow();
  });
});

describe('graph analysis traversal and lifetime contracts', () => {
  it('retains dependency order and releases both nodes at the last consumer', () => {
    const input: Node = {name: 'first', op: 'Placeholder', category: 'graph',
      inputNames: [], inputs: [], inputParams: {}, attrParams: {}, children: []};
    const output: Node = {...input, name: 'second', op: 'Identity',
      inputNames: ['first'], inputs: [input], children: []};
    input.children.push(output);
    const graph: Graph = {nodes: {first: input, second: output},
      placeholders: [], inputs: [], outputs: [output], weights: [input]};
    const info = getExecutionSubgraph({}, [output], {}, [input]);
    expect(info.missingInputs).toEqual([]);
    const ordered = getNodesInTopologicalOrder(graph, info);
    expect(ordered.map(node => node.name)).toEqual(['first', 'second']);
    const release = getNodeLiveUntilMap(ordered);
    expect(release.get('second')).toEqual([input, output]);
  });
});

describe('converter indexed metadata contracts', () => {
  it('preserves node names, optional output labels and numeric conversion', () => {
    expect(parseNodeName('')).toEqual(['', 0, undefined]);
    expect(parseNodeName('op')).toEqual(['op', 0, undefined]);
    expect(parseNodeName('op:2')).toEqual(['op', 2, undefined]);
    expect(parseNodeName('op:output:3')).toEqual(['op', 3, 'output']);
    expect(parseNodeName('op:a:b:4')).toEqual(['op', 4, undefined]);
    expect(parseNodeName(':0')).toEqual(['', 0, undefined]);
    expect(Number.isNaN(parseNodeName('op:invalid')[1])).toBe(true);
  });
  it('preserves explicit padding order and zeroes and rejects missing entries', () => {
    const node: Node = {name: 'padding', op: 'Conv2D', category: 'convolution',
      inputNames: [], inputs: [], inputParams: {}, children: [],
      attrParams: {pad: {value: 'explicit', type: 'string'},
        explicitPaddings: {value: [0, 0, 1, 2, 3, 4, 0, 0], type: 'number[]'}}};
    const context = new ExecutionContext({}, {}, {});
    expect(getPadding(node, {}, context)).toEqual([[0, 0], [1, 2], [3, 4], [0, 0]]);
    node.attrParams.explicitPaddings = {value: [0, 0], type: 'number[]'};
    expect(() => getPadding(node, {}, context)).toThrowError(/explicit padding/);
  });
});
