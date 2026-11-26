/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {requireComplexTensorInfos, requireTensorDataValues} from './tensor_data_values';
import {MathBackendCPU} from './backend_cpu';
import {argMin} from './kernels/ArgMin';
import {argMax} from './kernels/ArgMax';
import {complex} from './kernels/Complex';
import {fft} from './kernels/FFT';
import {ifft} from './kernels/IFFT';
import {avgPool} from './kernels/AvgPool';
import {maxPool} from './kernels/MaxPool';
import {avgPool3D} from './kernels/AvgPool3D';
import {maxPool3D} from './kernels/MaxPool3D';
import {resizeBilinear} from './kernels/ResizeBilinear';
import {resizeNearestNeighbor} from './kernels/ResizeNearestNeighbor';
import {resizeBilinearGrad} from './kernels/ResizeBilinearGrad';
import {resizeNearestNeighborGrad} from './kernels/ResizeNearestNeighborGrad';
import {nonMaxSuppressionV3} from './kernels/NonMaxSuppressionV3';
import {nonMaxSuppressionV4} from './kernels/NonMaxSuppressionV4';
import {nonMaxSuppressionV5} from './kernels/NonMaxSuppressionV5';

describe('CPU tensor storage payload', () => {
  it('preserves the numeric array identity, including zero and NaN', () => {
    const values = new Float32Array([0, NaN, -2]);
    expect(requireTensorDataValues({values, dtype: 'float32', refCount: 1})).toBe(values);
  });

  it('accepts empty numeric payloads', () => {
    const values = new Int32Array(0);
    expect(requireTensorDataValues({values, dtype: 'int32', refCount: 1})).toBe(values);
  });

  it('preserves encoded string arrays without conversion', () => {
    const values = [new Uint8Array([65]), new Uint8Array(0)];
    expect(requireTensorDataValues({values, dtype: 'string', refCount: 1})).toBe(values);
  });

  it('rejects a complex container without a direct payload', () => {
    expect(() => requireTensorDataValues({dtype: 'complex64', refCount: 1}))
        .toThrowError('CPU tensor data has no stored values.');
  });
});

describe('CPU complex tensor storage', () => {
  it('preserves the owned component records', () => {
    const real = {dataId: {}, shape: [1], dtype: 'float32' as const};
    const imag = {dataId: {}, shape: [1], dtype: 'float32' as const};
    const parts = {real, imag};
    expect(requireComplexTensorInfos({dtype: 'complex64', refCount: 1,
      complexTensorInfos: parts})).toBe(parts);
  });

  it('rejects missing component ownership', () => {
    expect(() => requireComplexTensorInfos({dtype: 'complex64', refCount: 1}))
        .toThrowError('CPU tensor data has no complex components.');
  });
});

describe('shared CPU pooling dispatch', () => {
  it('retains distinct 2D reductions and the unit-filter identity path', () => {
    const backend = new MathBackendCPU();
    const x = backend.makeTensorInfo([1, 2, 2, 1], 'float32',
        new Float32Array([1, 2, 3, 4]));
    const args = {inputs: {x}, backend, attrs: {
      filterSize: 2, strides: 1, pad: 'valid' as const
    }};
    const avg = avgPool(args);
    const max = maxPool(args);
    const identity = maxPool({...args, attrs: {...args.attrs, filterSize: 1}});
    expect(avg.shape).toEqual([1, 1, 1, 1]);
    expect(max.shape).toEqual(avg.shape);
    expect(avg.dtype).toBe('float32');
    expect(max.dtype).toBe('float32');
    expect(backend.readSync(avg.dataId)).toEqual(new Float32Array([2.5]));
    expect(backend.readSync(max.dataId)).toEqual(new Float32Array([4]));
    expect(identity.dataId).toBe(x.dataId);
    expect(backend.refCount(x.dataId)).toBe(2);
    backend.disposeData(identity.dataId);
    backend.disposeData(x.dataId);
    backend.disposeData(avg.dataId);
    backend.disposeData(max.dataId);
  });

  it('keeps the independent 3D reduction and output shape', () => {
    const backend = new MathBackendCPU();
    const x = backend.makeTensorInfo([1, 2, 2, 2, 1], 'float32',
        new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]));
    const args = {inputs: {x}, backend, attrs: {filterSize: 2, strides: 1,
      pad: 'valid' as const, dataFormat: 'NDHWC' as const}};
    const avg = avgPool3D(args);
    const max = maxPool3D(args);
    expect(avg.shape).toEqual([1, 1, 1, 1, 1]);
    expect(max.shape).toEqual(avg.shape);
    expect(avg.dtype).toBe('float32');
    expect(max.dtype).toBe('float32');
    expect(backend.readSync(avg.dataId)).toEqual(new Float32Array([4.5]));
    expect(backend.readSync(max.dataId)).toEqual(new Float32Array([8]));
    backend.disposeData(x.dataId);
    backend.disposeData(avg.dataId);
    backend.disposeData(max.dataId);
  });
});

describe('shared CPU extremum index and Fourier dispatch', () => {
  it('retains first ties, NaN behaviour and non-innermost axis reduction', () => {
    const backend = new MathBackendCPU();
    const x = backend.makeTensorInfo([3, 2], 'float32',
        new Float32Array([2, NaN, 2, 1, 0, 4]));
    const args = {inputs: {x}, backend, attrs: {axis: 0}};
    const min = argMin(args);
    const max = argMax(args);
    expect(min.shape).toEqual([2]);
    expect(max.dtype).toBe('int32');
    expect(backend.readSync(min.dataId)).toEqual(new Int32Array([2, 0]));
    expect(backend.readSync(max.dataId)).toEqual(new Int32Array([0, 0]));
    expect(backend.refCount(x.dataId)).toBe(1);
    backend.disposeData(min.dataId);
    backend.disposeData(max.dataId);
    backend.disposeData(x.dataId);
    expect(backend.numDataIds()).toBe(0);
  });

  it('restores batched shape and distinguishes forward from inverse FFT', () => {
    const backend = new MathBackendCPU();
    const shape = [2, 1, 2];
    const real = backend.makeTensorInfo(shape, 'float32',
        new Float32Array([1, 2, 3, 4]));
    const imag = backend.makeTensorInfo(shape, 'float32',
        new Float32Array(4));
    const input = complex({inputs: {real, imag}, backend});
    const forward = fft({inputs: {input}, backend});
    const inverse = ifft({inputs: {input: forward}, backend});
    expect(forward.shape).toEqual(shape);
    expect(inverse.shape).toEqual(shape);
    const expectedForward = [3, 0, -1, 0, 7, 0, -1, 0];
    const expectedInverse = [1, 0, 2, 0, 3, 0, 4, 0];
    const forwardValues = backend.readSync(forward.dataId);
    const inverseValues = backend.readSync(inverse.dataId);
    expect(forwardValues.length).toBe(expectedForward.length);
    expect(inverseValues.length).toBe(expectedInverse.length);
    expectedForward.forEach((value, index) => {
      expect(Number(forwardValues[index])).toBeCloseTo(value, 5);
    });
    expectedInverse.forEach((value, index) => {
      expect(Number(inverseValues[index])).toBeCloseTo(value, 5);
    });
    expect(backend.refCount(input.dataId)).toBe(1);
    [inverse, forward, input, real, imag].forEach(t => backend.disposeData(t.dataId));
    expect(backend.numDataIds()).toBe(0);
  });
});

describe('shared CPU resize geometry', () => {
  it('preserves distinct aligned interpolation and gradient accumulation', () => {
    const backend = new MathBackendCPU();
    const images = backend.makeTensorInfo([1, 2, 2, 1], 'float32',
        new Float32Array([1, 2, 3, 4]));
    const dy = backend.makeTensorInfo([1, 3, 3, 1], 'float32',
        new Float32Array(9).fill(1));
    const size: [number, number] = [3, 3];
    const attrs = {size, alignCorners: true, halfPixelCenters: false};
    const args = {inputs: {images}, backend, attrs};
    const bilinear = resizeBilinear(args);
    const nearest = resizeNearestNeighbor(args);
    const gradArgs = {inputs: {images, dy}, backend, attrs};
    const bilinearGrad = resizeBilinearGrad(gradArgs);
    const nearestGrad = resizeNearestNeighborGrad(gradArgs);
    expect(bilinear.shape).toEqual([1, 3, 3, 1]);
    expect(nearest.shape).toEqual(bilinear.shape);
    expect(backend.readSync(bilinear.dataId)).toEqual(
        new Float32Array([1, 1.5, 2, 2, 2.5, 3, 3, 3.5, 4]));
    expect(backend.readSync(nearest.dataId)).toEqual(
        new Float32Array([1, 2, 2, 3, 4, 4, 3, 4, 4]));
    expect(backend.readSync(bilinearGrad.dataId)).toEqual(
        new Float32Array([2.25, 2.25, 2.25, 2.25]));
    expect(backend.readSync(nearestGrad.dataId)).toEqual(
        new Float32Array([1, 2, 2, 4]));
    [images, dy, bilinear, nearest, bilinearGrad, nearestGrad]
        .forEach(t => backend.disposeData(t.dataId));
    expect(backend.numDataIds()).toBe(0);
  });

  it('preserves singleton output geometry and half-pixel differences', () => {
    const backend = new MathBackendCPU();
    const images = backend.makeTensorInfo([1, 2, 2, 1], 'float32',
        new Float32Array([1, 2, 3, 4]));
    const size: [number, number] = [1, 1];
    const args = {inputs: {images}, backend,
      attrs: {size, alignCorners: false, halfPixelCenters: true}};
    const bilinear = resizeBilinear(args);
    const nearest = resizeNearestNeighbor(args);
    const aligned = resizeBilinear({...args,
      attrs: {size, alignCorners: true, halfPixelCenters: false}});
    expect(backend.readSync(bilinear.dataId)).toEqual(new Float32Array([2.5]));
    expect(backend.readSync(nearest.dataId)).toEqual(new Float32Array([4]));
    expect(backend.readSync(aligned.dataId)).toEqual(new Float32Array([1]));
    [images, bilinear, nearest, aligned].forEach(t => backend.disposeData(t.dataId));
    expect(backend.numDataIds()).toBe(0);
  });
});

describe('shared CPU non-maximum suppression preparation', () => {
  it('preserves indices, padded count, scores and input ownership', () => {
    const backend = new MathBackendCPU();
    const boxes = backend.makeTensorInfo([3, 4], 'float32',
        new Float32Array([0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3]));
    const scores = backend.makeTensorInfo([3], 'float32',
        new Float32Array([0.9, 0.8, 0.7]));
    const attrs = {maxOutputSize: 3, iouThreshold: 0.5, scoreThreshold: 0};
    const args = {inputs: {boxes, scores}, backend, attrs};
    const indices = nonMaxSuppressionV3(args);
    const [padded, count] = nonMaxSuppressionV4({...args,
      attrs: {...attrs, padToMaxOutputSize: true}});
    const [scored, selectedScores] = nonMaxSuppressionV5({...args,
      attrs: {...attrs, softNmsSigma: 0}});
    expect(indices.dtype).toBe('int32');
    expect(indices.shape).toEqual([2]);
    expect(padded.shape).toEqual([3]);
    expect(count.shape).toEqual([]);
    expect(selectedScores.dtype).toBe('float32');
    expect(backend.readSync(indices.dataId)).toEqual(new Int32Array([0, 2]));
    expect(backend.readSync(padded.dataId)).toEqual(new Int32Array([0, 2, 0]));
    expect(backend.readSync(count.dataId)).toEqual(new Int32Array([2]));
    expect(backend.readSync(scored.dataId)).toEqual(new Int32Array([0, 2]));
    expect(backend.readSync(selectedScores.dataId)).toEqual(
        new Float32Array([0.9, 0.7]));
    expect(backend.refCount(boxes.dataId)).toBe(1);
    expect(backend.refCount(scores.dataId)).toBe(1);
    [boxes, scores, indices, padded, count, scored, selectedScores]
        .forEach(t => backend.disposeData(t.dataId));
    expect(backend.numDataIds()).toBe(0);
  });
});
