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
import {ArgMax, ArgMin, Max, MaxAttrs, Reshape, Sum} from './kernel_names';
import {ENGINE} from './engine';
import {reduceBroadcastGradient, sumBroadcastGradient} from './gradients/broadcast_grad_util';
import {GradConfig, KernelAttributes, KernelConfig, KernelFunc} from './kernel_registry';
import {argMaxGradConfig} from './gradients/ArgMax_grad';
import {argMinGradConfig} from './gradients/ArgMin_grad';
import {getShapeForBatchMatMul} from './ops/conv_util';
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

  it('preserves nested attributes and optional input identity at dispatch', () => {
    interface NestedAttrs {
      paddings: number[][];
      axis?: number;
    }
    const input: TensorInfo = {dataId: {}, shape: [2], dtype: 'float32'};
    const inputs = {x: input};
    const attrs: KernelAttributes<NestedAttrs> = {paddings: [[1, 2]]};
    const backend = {result: input};
    const config: KernelConfig = {
      kernelName: 'NestedDispatchTest', backendName: 'test',
      kernelFunc: (args: {
        inputs: {x: TensorInfo, bias?: TensorInfo},
        backend: {result: TensorInfo},
        attrs: KernelAttributes<NestedAttrs>
      }) => {
        expect(args.inputs).toBe(inputs);
        expect('bias' in args.inputs).toBeFalse();
        expect(args.backend).toBe(backend);
        expect(args.attrs).toBe(attrs);
        expect(args.attrs.paddings).toEqual([[1, 2]]);
        expect('axis' in args.attrs).toBeFalse();
        return args.backend.result;
      }
    };
    expect(config.kernelFunc({inputs, backend, attrs})).toBe(input);
  });

  it('retains typed gradient attributes at the registry boundary', () => {
    interface GradientAttrs {
      axis: number;
      keepDims?: boolean;
    }
    const x = new Tensor<Rank.R1>([2], 'float32', {}, 14);
    const attrs: KernelAttributes<GradientAttrs> = {axis: 0};
    const config: GradConfig = {
      kernelName: 'TypedGradientTest',
      gradFunc: (dy: Tensor, saved: Tensor[], params: KernelAttributes<GradientAttrs>) => {
        expect(params).toBe(attrs);
        expect(params.axis).toBe(0);
        expect('keepDims' in params).toBeFalse();
        expect(saved).toEqual([x]);
        return {x: () => dy};
      }
    };
    expect(Object.keys(config.gradFunc(x, [x], attrs))).toEqual(['x']);
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

describe('shared backend and extrema contracts', () => {
  it('fuses only spatial dimensions of convolution broadcast operands', () => {
    expect(getShapeForBatchMatMul([2, 3, 4, 5], true)).toEqual([2, 12, 5]);
    expect(getShapeForBatchMatMul([2, 5, 3, 4], false)).toEqual([2, 5, 12]);
    expect(getShapeForBatchMatMul([5], false)).toEqual([5, 1]);
    expect(getShapeForBatchMatMul([5], true)).toBeNull();
    expect(getShapeForBatchMatMul([1], false)).toBeNull();
    expect(getShapeForBatchMatMul([3, 4], false)).toBeNull();
  });

  it('preserves independent extrema registrations and saved-input checks', () => {
    const x = new Tensor<Rank.R1>([2], 'float32', {}, 1);
    expect(argMinGradConfig.kernelName).toBe(ArgMin);
    expect(argMaxGradConfig.kernelName).toBe(ArgMax);
    expect(argMinGradConfig.inputsToSave).toEqual(['x']);
    expect(argMaxGradConfig.inputsToSave).toEqual(['x']);
    expect(argMinGradConfig.gradFunc).toBe(argMaxGradConfig.gradFunc);
    expect(() => argMinGradConfig.gradFunc(x, [], {})).toThrowError();
    expect(Object.keys(argMaxGradConfig.gradFunc(x, [x], {}))).toEqual(['x']);
  });
});

describe('shared broadcast gradient reduction', () => {
  it('retains identity without a broadcast reduction', () => {
    const gradient = new Tensor<Rank.R1>([2], 'float32', {}, 10);
    const dispatch = spyOn(ENGINE, 'runKernel');
    expect(sumBroadcastGradient(gradient, [2], [2])).toBe(gradient);
    expect(reduceBroadcastGradient(gradient, [2], [2])).toBe(gradient);
    expect(dispatch).not.toHaveBeenCalled();
  });

  it('sums broadcast axes and restores only the requested shape', () => {
    const gradient = new Tensor<Rank.R2>([3, 2], 'float32', {}, 11);
    const reduced = new Tensor<Rank.R1>([2], 'float32', {}, 12);
    const restored = new Tensor<Rank.R2>([1, 2], 'float32', {}, 13);
    spyOn(ENGINE, 'startScope');
    spyOn(ENGINE, 'endScope');
    const dispatch = spyOn(ENGINE, 'runKernel').and.returnValues(reduced, reduced, restored);
    expect(sumBroadcastGradient(gradient, [1, 2], [3, 2])).toBe(reduced);
    expect(reduceBroadcastGradient(gradient, [1, 2], [3, 2])).toBe(restored);
    expect(dispatch.calls.allArgs()).toEqual([
      [Sum, {x: gradient}, {axis: [0], keepDims: false}],
      [Sum, {x: gradient}, {axis: [0], keepDims: false}],
      [Reshape, {x: reduced}, {shape: [1, 2]}]
    ]);
  });
});
