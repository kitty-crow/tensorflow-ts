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
import {Max, MaxAttrs} from './kernel_names';
import {KernelAttributes, KernelConfig, KernelFunc} from './kernel_registry';
import {maximumGradConfig} from './gradients/Maximum_grad';
import {Tensor} from './tensor';
import {TensorInfo} from './tensor_info';
import {Rank} from './types';

describe('kernel signature and saved-tensor contracts', () => {
  it('registers a structurally typed attribute contract without erasure', () => {
    const input: TensorInfo = {dataId: {}, shape: [2], dtype: 'float32'};
    const attrs: KernelAttributes<MaxAttrs> = {reductionIndices: [0], keepDims: false};
    const kernel = (params: {
      inputs: Record<'x', TensorInfo>, backend: {},
      attrs: KernelAttributes<MaxAttrs>
    }): TensorInfo => {
      expect(params.inputs.x).toBe(input);
      expect(params.attrs).toBe(attrs);
      return params.inputs.x;
    };
    const config: KernelConfig = {
      kernelName: Max, backendName: 'test', kernelFunc: kernel
    };
    const registered: KernelFunc = config.kernelFunc;
    expect(registered({inputs: {x: input}, backend: {}, attrs}))
        .toBe(input);
    expect(config.kernelFunc).toBe(kernel);
  });

  it('keeps optional attributes and multiple outputs without a wrapper', () => {
    interface DispatchAttrs {
      axis: number;
      keepDims?: boolean;
    }
    const input: TensorInfo = {dataId: {}, shape: [2], dtype: 'float32'};
    const outputs: TensorInfo[] = [input, input];
    const attrs: KernelAttributes<DispatchAttrs> = {axis: 0};
    const kernel = (params: {
      inputs: Record<'x', TensorInfo>, backend: {},
      attrs: KernelAttributes<DispatchAttrs>
    }): TensorInfo[] => {
      expect(params.inputs.x).toBe(input);
      expect(params.attrs).toBe(attrs);
      expect('keepDims' in params.attrs).toBeFalse();
      return outputs;
    };
    const config: KernelConfig = {
      kernelName: 'OptionalDispatchTest', backendName: 'test',
      kernelFunc: kernel
    };
    expect(config.kernelFunc).toBe(kernel);
    expect(config.kernelFunc({inputs: {x: input}, backend: {}, attrs}))
        .toBe(outputs);
  });

  it('rejects missing or surplus saved tensors before gradient dispatch', () => {
    const tensor = new Tensor<Rank.R1>([1], 'float32', {}, 0);
    expect(() => maximumGradConfig.gradFunc(tensor, [tensor], {}))
        .toThrowError('Expected an array with exactly 2 elements.');
    expect(() => maximumGradConfig.gradFunc(tensor, [tensor, tensor, tensor], {}))
        .toThrowError('Expected an array with exactly 2 elements.');
    const result = maximumGradConfig.gradFunc(tensor, [tensor, tensor], {});
    expect(Object.keys(result)).toEqual(['a', 'b']);
  });
});
