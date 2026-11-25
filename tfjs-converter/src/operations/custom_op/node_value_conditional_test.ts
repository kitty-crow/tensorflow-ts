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
import {Node} from '../types';

import {NodeValueImpl} from './node_value_impl';

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
