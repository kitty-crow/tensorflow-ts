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

import {test_util} from '@tensorflow/tfjs-core';
import * as tensorflow from '../../data/compiled_api';
import {ExecutionContext} from '../../executor/execution_context';
import {assertShapesMatchAllowUndefinedSize, mergeElementShape} from '../../executor/tensor_utils';
import {Graph, Node} from '../types';
import {getExecutionSubgraph, getNodesInTopologicalOrder, getNodeLiveUntilMap} from '../../executor/model_analysis';
import {GraphExecutor} from '../../executor/graph_executor';
import {getNumericArrayParam, OperationMapper, parseStringParam} from '../operation_mapper';

import {NodeValueImpl} from './node_value_impl';
import {spyOnAllFunctions} from '../executors/spy_ops';
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
    expect(test_util.requiredTestEntry(result.attrs, 'scalarFirst')).toBe(7);
    expect(test_util.requiredTestEntry(result.attrs, 'numericFirst')).toEqual([3, 6, 0]);
    expect(test_util.requiredTestEntry(result.attrs, 'floats')).toEqual([1.5, 2.5]);
    expect(test_util.requiredTestEntry(result.attrs, 'strings')).toEqual(['nhwc']);
    expect(test_util.requiredTestEntry(result.attrs, 'bools')).toEqual([true, false]);
    expect(test_util.requiredTestEntry(result.attrs, 'shapes')).toEqual([[2, 3]]);
    expect(test_util.requiredTestEntry(result.attrs, 'types')).toEqual(['int32']);
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
    expect(test_util.requiredTestValue(release.get('second'), 'second')).toEqual([input, output]);
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
    const padding = getPadding(node, {}, context);
    if (!Array.isArray(padding)) {
      throw new Error('Expected explicit padding pairs.');
    }
    const pairs: number[][] = [];
    for (const pair of padding) {
      if (!Array.isArray(pair) || pair.length !== 2) {
        throw new Error('Expected a pair of explicit padding values.');
      }
      const [before, after] = pair;
      if (typeof before !== 'number' || typeof after !== 'number') {
        throw new Error('Expected numeric explicit padding values.');
      }
      pairs.push([before, after]);
    }
    expect(pairs).toEqual([[0, 0], [1, 2], [3, 4], [0, 0]]);
    node.attrParams.explicitPaddings = {value: [0, 0], type: 'number[]'};
    expect(() => getPadding(node, {}, context)).toThrowError(/explicit padding/);
  });
});

describe('tensor list shape metadata', () => {
  it('preserves wildcard dimensions, zero dimensions and unknown rank', () => {
    expect(mergeElementShape([-1, 0], [4, -1])).toEqual([4, 0]);
    expect(mergeElementShape(-1, [2, 0])).toEqual([2, 0]);
    expect(() => assertShapesMatchAllowUndefinedSize([-1, 0], [4, 0])).not.toThrow();
    expect(() => mergeElementShape([2], [3])).toThrowError(/Incompatible shape/);
    expect(() => mergeElementShape(new Array<number>(1), [2]))
        .toThrowError(/Missing numeric entry/);
  });
});


describe('checked function signature mapping', () => {
  const dtype = tensorflow.DataType.DT_FLOAT;
  it('keeps argument order and maps output names to their returned tensor', () => {
    const graph = OperationMapper.Instance.transformGraph({
      node: [], library: {function: [{
        signature: {name: 'inner', inputArg: [{name: 'arg', type: dtype}],
          outputArg: [{name: 'result', type: dtype}]},
        nodeDef: [], ret: {result: 'arg:0'}
      }]}
    });
    const inner = test_util.requiredTestEntry(
        test_util.requiredTestValue(graph.functions, 'functions'), 'inner');
    const input = test_util.requiredTestElement(inner.inputs, 0);
    expect(input.signatureKey).toBe('arg');
    expect(test_util.requiredTestElement(inner.outputs, 0)).toBe(input);
    expect(inner.signature).toEqual({
      methodName: 'inner', inputs: {arg: {name: 'arg', dtype}},
      outputs: {result: {name: 'arg:0', dtype}}
    });
  });

  it('rejects missing return names and permits an empty signature', () => {
    const definition: tensorflow.IOpDef = {name: 'inner', inputArg: [],
      outputArg: [{name: 'result', type: dtype}]};
    expect(() => OperationMapper.Instance.transformGraph({
      node: [], library: {function: [{signature: definition, ret: {}}]}
    })).toThrowError(/missing return "result"/);
    const empty = OperationMapper.Instance.transformGraph({
      node: [], library: {function: [{
        signature: {name: 'empty', inputArg: [], outputArg: []}
      }]}
    });
    const fn = test_util.requiredTestEntry(
        test_util.requiredTestValue(empty.functions, 'functions'), 'empty');
    expect(fn.signature).toEqual({methodName: 'empty', inputs: {}, outputs: {}});
  });

  it('retains float precedence, zero, integer strings and empty numeric arrays', () => {
    expect(getNumericArrayParam({a: {list: {f: [0, 1.5], i: [3]}}}, 'a', [9]))
        .toEqual([0, 1.5]);
    expect(getNumericArrayParam({a: {list: {f: [], i: ['2', 0]}}}, 'a', [9]))
        .toEqual([2, 0]);
    expect(getNumericArrayParam({a: {list: {}}}, 'a', [9])).toEqual([]);
    expect(getNumericArrayParam({}, 'a', [9])).toEqual([9]);
    expect(() => getNumericArrayParam({a: {}}, 'a', [])).toThrowError(/missing its list/);
    expect(parseStringParam([], true)).toBe('');
  });
});


describe('converter legacy attribute fallback', () => {
  it('uses legacy squeeze axes only if the modern attribute is absent', () => {
    const graph = OperationMapper.Instance.transformGraph({node: [
      {name: 'input', op: 'Placeholder'},
      {name: 'legacy', op: 'Squeeze', input: ['input'], attr: {
        squeeze_dims: {list: {i: [1, 2]}}
      }},
      {name: 'modern', op: 'Squeeze', input: ['input'], attr: {
        axis: {list: {i: [0]}}, squeeze_dims: {list: {i: [1, 2]}}
      }},
      {name: 'empty', op: 'Squeeze', input: ['input'], attr: {
        axis: {list: {i: []}}, squeeze_dims: {list: {i: [1, 2]}}
      }},
      {name: 'absent', op: 'Squeeze', input: ['input']}
    ]});
    const axis = (name: string) => test_util.requiredTestEntry(
        test_util.requiredTestEntry(graph.nodes, name).attrParams, 'axis').value;
    expect(axis('legacy')).toEqual([1, 2]);
    expect(axis('modern')).toEqual([0]);
    expect(axis('empty')).toEqual([]);
    expect(axis('absent')).toBeUndefined();
  });
});


describe('converter spy module entries', () => {
  it('retains the exempt constructor and calls through nested spies', () => {
    class CompositeArrayBuffer {
      constructor(readonly size: number) {}
    }
    const source = {CompositeArrayBuffer, nested: {add: (a: number, b: number) => a + b}};
    const spies = spyOnAllFunctions(source);
    expect(spies.CompositeArrayBuffer).toBe(CompositeArrayBuffer);
    expect(new spies.CompositeArrayBuffer(3).size).toBe(3);
    expect(spies.nested.add(2, 3)).toBe(5);
    expect(spies.nested.add).toHaveBeenCalledWith(2, 3);
  });
});
