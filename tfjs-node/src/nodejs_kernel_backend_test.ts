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

import * as tf from '@tensorflow/tfjs';
// tslint:disable-next-line: no-imports-from-dist
import {TestKernelBackend} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {absConfig} from './kernels/Abs';
import {argMaxConfig} from './kernels/ArgMax';
import {argMinConfig} from './kernels/ArgMin';
import {resizeBilinearConfig} from './kernels/ResizeBilinear';
import {resizeNearestNeighborConfig} from './kernels/ResizeNearestNeighbor';
import {cosConfig} from './kernels/Cos';
import {FFTConfig} from './kernels/FFT';
import {IFFTConfig} from './kernels/IFFT';
import {sinConfig} from './kernels/Sin';
import {executeWithInt32Vector, indexedTensorAttrs, localResponseAttrs, pool2DAttrs, pool3DAttrs} from './kernels/native_dispatch';
import {addConfig} from './kernels/Add';
import {atan2Config} from './kernels/Atan2';
import {modConfig} from './kernels/Mod';
import {multiplyConfig} from './kernels/Multiply';
import {createTensorsTypeOpAttr, ensureTensorflowBackend, getTFDType, nodeBackend, NodeJSKernelBackend} from './nodejs_kernel_backend';

describe('delayed upload', () => {
  it('should handle data before op execution', async () => {
    const t = tf.tensor1d([1, 2, 3]);
    tf.test_util.expectArraysClose(await t.data(), [1, 2, 3]);

    const r = t.add(tf.tensor1d([4, 5, 6]));
    tf.test_util.expectArraysClose(await r.data(), [5, 7, 9]);
  });

  it('Should not cache tensors in the tensor map for device support. ', () => {
    const logits = tf.tensor1d([1, 2, 3]);
    const softmaxLogits = tf.softmax(logits);
    const data = softmaxLogits.dataSync();
    expect(softmaxLogits.dataSync()[0]).toEqual(data[0]);
    expect(softmaxLogits.dataSync()[1]).toEqual(data[1]);
    expect(softmaxLogits.dataSync()[2]).toEqual(data[2]);
  });
});

describe('type casting', () => {
  it('exp support int32', () => {
    tf.exp(tf.scalar(2, 'int32'));
  });
});

describe('conv3d dilations', () => {
  it('CPU should throw error on dilations >1', () => {
    const input: tf.Tensor5D = tf.ones([1, 2, 2, 2, 1]);
    const filter: tf.Tensor5D = tf.ones([1, 1, 1, 1, 1]);
    expect(() => {
      tf.conv3d(input, filter, 1, 'same', 'NDHWC', [2, 2, 2]);
    }).toThrowError();
  });
  it('GPU should handle dilations >1', () => {
    // This test can only run locally with CUDA bindings and GPU package
    // installed.
    if ((tf.backend() as NodeJSKernelBackend).isUsingGpuDevice) {
      const input: tf.Tensor5D = tf.ones([1, 2, 2, 2, 1]);
      const filter: tf.Tensor5D = tf.ones([1, 1, 1, 1, 1]);
      tf.conv3d(input, filter, 1, 'same', 'NDHWC', [2, 2, 2]);
    }
  });
});

describe('Exposes Backend for internal Op execution.', () => {
  it('Provides the Node backend over a function', () => {
    const backend = nodeBackend();
    expect(backend instanceof NodeJSKernelBackend).toBeTruthy();
  });

  it('Provides internal access to the binding', () => {
    expect(nodeBackend().binding).toBeDefined();
  });

  it('throw error if backend is not tensorflow', async () => {
    const testBackend = new TestKernelBackend();
    tf.registerBackend('fake', () => testBackend);
    tf.setBackend('fake');
    try {
      expect(() => ensureTensorflowBackend()).toThrowError(
        'Expect the current backend to be "tensorflow", but got "fake"');
    } finally {
      tf.setBackend('tensorflow');
    }
  });
});

describe('getTFDType()', () => {
  const binding = nodeBackend().binding;

  it('handles float32', () => {
    expect(getTFDType('float32')).toBe(binding.TF_FLOAT);
  });
  it('handles int32', () => {
    expect(getTFDType('int32')).toBe(binding.TF_INT32);
  });
  it('handles bool', () => {
    expect(getTFDType('bool')).toBe(binding.TF_BOOL);
  });
  it('handles unknown types', () => {
    expect(() => getTFDType(null)).toThrowError();
  });
});

describe('createTypeOpAttr()', () => {
  const binding = nodeBackend().binding;

  it('Creates a valid type attribute', () => {
    const attr = createTensorsTypeOpAttr('foo', 'float32');
    expect(attr.name).toBe('foo');
    expect(attr.type).toBe(binding.TF_ATTR_TYPE);
    expect(attr.value).toBe(binding.TF_FLOAT);
  });

  it('handles unknown dtypes', () => {
    expect(() => createTensorsTypeOpAttr('foo', null)).toThrowError();
  });
});

describe('Returns TFEOpAttr for a Tensor or list of Tensors', () => {
  const binding = nodeBackend().binding;

  it('handles a single Tensor', () => {
    const result = createTensorsTypeOpAttr('T', tf.scalar(13, 'float32'));
    expect(result.name).toBe('T');
    expect(result.type).toBe(binding.TF_ATTR_TYPE);
    expect(result.value).toBe(binding.TF_FLOAT);
  });
  it('handles a list of Tensors', () => {
    const tensors = [tf.scalar(1, 'int32'), tf.scalar(20.1, 'float32')];
    const result = createTensorsTypeOpAttr('T', tensors);
    expect(result.name).toBe('T');
    expect(result.type).toBe(binding.TF_ATTR_TYPE);
    expect(result.value).toBe(binding.TF_INT32);
  });
  it('handles null', () => {
    expect(() => createTensorsTypeOpAttr('T', null)).toThrowError();
  });
  it('handles list of null', () => {
    const inputs = [null, null] as tf.Tensor[];
    expect(() => createTensorsTypeOpAttr('T', inputs)).toThrowError();
  });
});

describe('binary native kernel dispatch', () => {
  it('preserves upcast, native operation names and operand order', () => {
    const backend = nodeBackend();
    const a = tf.scalar(2, 'int32');
    const b = tf.scalar(3, 'float32');
    const output = tf.scalar(5);
    try {
      const dispatch = spyOn(backend, 'executeSingleOutput').and.returnValue(output);
      expect(addConfig.kernelFunc({backend, inputs: {a, b}})).toBe(output);
      expect(dispatch).toHaveBeenCalledWith('Add',
          [createTensorsTypeOpAttr('T', 'float32')], [a, b]);
      expect(multiplyConfig.kernelFunc({backend, inputs: {a, b}})).toBe(output);
      expect(dispatch).toHaveBeenCalledWith('Mul',
          [createTensorsTypeOpAttr('T', 'float32')], [a, b]);
    } finally {
      a.dispose();
      b.dispose();
      output.dispose();
    }
  });

  it('preserves left-operand dtype and native aliases', () => {
    const backend = nodeBackend();
    const a = tf.scalar(2, 'int32');
    const b = tf.scalar(3, 'float32');
    const output = tf.scalar(2);
    try {
      const dispatch = spyOn(backend, 'executeSingleOutput').and.returnValue(output);
      expect(atan2Config.kernelFunc({backend, inputs: {a, b}})).toBe(output);
      expect(dispatch).toHaveBeenCalledWith('Atan2',
          [createTensorsTypeOpAttr('T', 'int32')], [a, b]);
      expect(modConfig.kernelFunc({backend, inputs: {a, b}})).toBe(output);
      expect(dispatch).toHaveBeenCalledWith('FloorMod',
          [createTensorsTypeOpAttr('T', 'int32')], [a, b]);
    } finally {
      a.dispose();
      b.dispose();
      output.dispose();
    }
  });
});

describe('shared native dispatch contracts', () => {
  it('retains separate unary kernel names and native dtype attributes', () => {
    const backend = nodeBackend();
    const x = tf.scalar(2, 'float32');
    const result = tf.scalar(3);
    try {
      const dispatch = spyOn(backend, 'executeSingleOutput').and.returnValue(result);
      expect(sinConfig.kernelFunc({backend, inputs: {x}})).toBe(result);
      expect(dispatch).toHaveBeenCalledWith('Sin', [createTensorsTypeOpAttr('T', 'float32')], [x]);
      expect(cosConfig.kernelFunc({backend, inputs: {x}})).toBe(result);
      expect(dispatch).toHaveBeenCalledWith('Cos', [createTensorsTypeOpAttr('T', 'float32')], [x]);
      expect(absConfig.kernelFunc({backend, inputs: {x}})).toBe(result);
      expect(dispatch).toHaveBeenCalledWith('Abs', [createTensorsTypeOpAttr('T', 'float32')], [x]);
      expect(FFTConfig.kernelFunc({backend, inputs: {input: x}})).toBe(result);
      expect(dispatch).toHaveBeenCalledWith('FFT', [createTensorsTypeOpAttr('Tcomplex', 'float32')], [x]);
      expect(IFFTConfig.kernelFunc({backend, inputs: {input: x}})).toBe(result);
      expect(dispatch).toHaveBeenCalledWith('IFFT', [createTensorsTypeOpAttr('Tcomplex', 'float32')], [x]);
    } finally {
      x.dispose();
      result.dispose();
    }
  });

  it('disposes the temporary vector on success and native failure', () => {
    const backend = nodeBackend();
    const x = tf.scalar(2);
    const result = tf.scalar(3);
    const captured: tf.Tensor[] = [];
    try {
      const dispatch = spyOn(backend, 'executeSingleOutput').and.callFake(
          (name, attrs, inputs) => {
            expect(name).toBe('Tile');
            expect(attrs).toEqual([]);
            expect(inputs[0]).toBe(x);
            const vector = inputs[1] as tf.Tensor;
            captured.push(vector);
            expect(vector.dtype).toBe('int32');
            expect(Array.from(vector.dataSync())).toEqual([2, 3]);
            return result;
          });
      expect(executeWithInt32Vector(backend, 'Tile', [], [x], [2, 3])).toBe(result);
      expect(captured[0].isDisposed).toBeTrue();
      dispatch.and.callFake((name, attrs, inputs) => {
        captured.push(inputs[1] as tf.Tensor);
        throw new Error('native failure');
      });
      expect(() => executeWithInt32Vector(backend, 'Tile', [], [x], [2, 3]))
          .toThrowError('native failure');
      expect(captured[1].isDisposed).toBeTrue();
      expect(x.isDisposed).toBeFalse();
    } finally {
      x.dispose();
      result.dispose();
    }
  });

  it('preserves LRN attribute order and binding types', () => {
    const backend = nodeBackend();
    const attrs = localResponseAttrs(backend, 'float32', {
      depthRadius: 2, bias: 1, alpha: 0.5, beta: 0.75
    });
    expect(attrs).toEqual([
      createTensorsTypeOpAttr('T', 'float32'),
      {name: 'depth_radius', type: backend.binding.TF_ATTR_INT, value: 2},
      {name: 'bias', type: backend.binding.TF_ATTR_FLOAT, value: 1},
      {name: 'alpha', type: backend.binding.TF_ATTR_FLOAT, value: 0.5},
      {name: 'beta', type: backend.binding.TF_ATTR_FLOAT, value: 0.75}
    ]);
  });
});

describe('native reduction, pooling and resize contracts', () => {
  it('preserves pooling rank, layout and padding diagnostics', () => {
    const backend = nodeBackend();
    const info2D = tf.backend_util.computePool2DInfo(
        [1, 4, 6, 2], [2, 3], [1, 2], 1, 'same');
    const info3D = tf.backend_util.computePool3DInfo(
        [1, 3, 4, 6, 2], [2, 2, 3], [1, 1, 2], 1, 'valid');
    const attrs2D = pool2DAttrs(backend, 'float32', info2D, 'padding');
    const attrs3D = pool3DAttrs(backend, 'float32', info3D, 'padding type');
    expect(attrs2D.map(a => a.value)).toEqual([
      backend.binding.TF_FLOAT, [1, 2, 3, 1], [1, 1, 2, 1], 'SAME', 'NHWC'
    ]);
    expect(attrs3D.map(a => a.value)).toEqual([
      backend.binding.TF_FLOAT, [1, 2, 2, 3, 1], [1, 1, 1, 2, 1], 'VALID', 'NDHWC'
    ]);
    expect(() => pool2DAttrs(backend, 'float32', {
      ...info2D, padInfo: {...info2D.padInfo, type: 'NUMBER'}
    }, 'padding type')).toThrowError(
        "TF Backend supports only 'valid' and 'same' padding while padding type was NUMBER");
  });

  it('preserves scalar reduction axes and distinct native extrema identities', () => {
    const backend = nodeBackend();
    const x = tf.tensor1d([3, 1], 'int32');
    const result = tf.scalar(0, 'int32');
    const axes: tf.Tensor[] = [];
    try {
      const dispatch = spyOn(backend, 'executeSingleOutput').and.callFake(
          (name, attrs, inputs) => {
            expect(['ArgMin', 'ArgMax']).toContain(name);
            expect(inputs[0]).toBe(x);
            expect(attrs.map(a => a.name)).toEqual(['T', 'Tidx', 'output_type']);
            const axis = inputs[1] as tf.Tensor;
            axes.push(axis);
            expect(axis.shape).toEqual([]);
            expect(axis.dtype).toBe('int32');
            expect(Array.from(axis.dataSync())).toEqual([0]);
            return result;
          });
      expect(argMinConfig.kernelFunc({backend, inputs: {x}, attrs: {axis: 0}})).toBe(result);
      expect(argMaxConfig.kernelFunc({backend, inputs: {x}, attrs: {axis: 0}})).toBe(result);
      expect(dispatch.calls.allArgs().map(args => args[0])).toEqual(['ArgMin', 'ArgMax']);
      expect(axes.every(axis => axis.isDisposed)).toBeTrue();
    } finally {
      x.dispose();
      result.dispose();
    }
  });

  it('keeps resize flags and native names independent of the shared adapter', () => {
    const backend = nodeBackend();
    const images = tf.tensor4d([1], [1, 1, 1, 1]);
    const result = tf.scalar(1);
    try {
      const dispatch = spyOn(backend, 'executeSingleOutput').and.callFake(
          (name, attrs, inputs) => {
            expect(inputs[0]).toBe(images);
            expect(attrs.map(a => a.value)).toEqual([backend.binding.TF_FLOAT, false, true]);
            expect(Array.from((inputs[1] as tf.Tensor).dataSync())).toEqual([2, 3]);
            return result;
          });
      const attrs = {alignCorners: false, halfPixelCenters: true, size: [2, 3]};
      expect(resizeBilinearConfig.kernelFunc({backend, inputs: {images}, attrs})).toBe(result);
      expect(resizeNearestNeighborConfig.kernelFunc({backend, inputs: {images}, attrs})).toBe(result);
      expect(dispatch.calls.allArgs().map(args => args[0])).toEqual(
          ['ResizeBilinear', 'ResizeNearestNeighbor']);
    } finally {
      images.dispose();
      result.dispose();
    }
  });
});

describe('native indexed tensor attributes', () => {
  it('preserves gather and scatter value attribute names and index dtype', () => {
    expect(indexedTensorAttrs('Tparams', 'float32')).toEqual([
      createTensorsTypeOpAttr('Tparams', 'float32'),
      createTensorsTypeOpAttr('Tindices', 'int32')
    ]);
    expect(indexedTensorAttrs('T', 'bool')).toEqual([
      createTensorsTypeOpAttr('T', 'bool'),
      createTensorsTypeOpAttr('Tindices', 'int32')
    ]);
  });
});
