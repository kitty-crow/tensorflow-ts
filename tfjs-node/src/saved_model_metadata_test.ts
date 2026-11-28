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

import {getEnumKeyFromValue, inspectSavedModelMetadata} from './saved_model_metadata';

describe('SavedModel metadata', () => {
  const types = {DT_FLOAT: 1, DT_INT32: 3};

  function tensor(name: string, dimensions: number[], dtype = 1) {
    return {
      getDtype: () => dtype,
      getName: () => name,
      getTensorShape: () => ({getDimList: () => dimensions.map(size => ({getSize: () => size}))})
    };
  }

  function graph(inputs: Map<string, ReturnType<typeof tensor>>,
                 outputs: Map<string, ReturnType<typeof tensor>>) {
    return {
      getMetaGraphsList: () => [{
        getMetaInfoDef: () => ({getTagsList: () => ['serve']}),
        getSignatureDefMap: () => new Map([
          ['__saved_model_init_op', {}],
          ['serving_default', {getInputsMap: () => inputs, getOutputsMap: () => outputs}]
        ])
      }]
    };
  }

  it('projects numeric dimensions, scalar shapes, dtypes and tags', () => {
    const model = graph(new Map([['input', tensor('x:0', [-1, 3])]]),
        new Map([['output', tensor('y:0', [], 3)]]));
    expect(inspectSavedModelMetadata(model, types)).toEqual([{
      tags: ['serve'],
      signatureDefs: {serving_default: {
        inputs: {input: {name: 'x:0', shape: [-1, 3], tfDtype: 'DT_FLOAT', dtype: 'float32'}},
        outputs: {output: {name: 'y:0', shape: [], tfDtype: 'DT_INT32', dtype: 'int32'}}
      }}
    }]);
  });

  it('rejects unsupported dtypes and malformed generated messages', () => {
    expect(() => inspectSavedModelMetadata(
        graph(new Map([['input', tensor('x:0', [], 42)]]), new Map<string, ReturnType<typeof tensor>>()), types))
        .toThrowError(/Unsupported tensor DataType/);
    expect(() => inspectSavedModelMetadata({}, types))
        .toThrowError('Missing protobuf method getMetaGraphsList');
  });

  it('preserves enum lookup for known and unknown numeric values', () => {
    expect(getEnumKeyFromValue(types, 3)).toBe('DT_INT32');
    expect(getEnumKeyFromValue(types, 42)).toBeUndefined();
  });
});
