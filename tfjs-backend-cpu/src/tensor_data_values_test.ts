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
import {transpose} from './kernels/Transpose';
import {unpack} from './kernels/Unpack';
import {splitV} from './kernels/SplitV';
import {leakyRelu} from './kernels/LeakyRelu';
import {multinomial} from './kernels/Multinomial';
import {MathBackendCPU} from './backend_cpu';
import {all} from './kernels/All';
import {any as anyReduction} from './kernels/Any';
import {min} from './kernels/Min';
import {max} from './kernels/Max';
import {batchNorm} from './kernels/BatchNorm';
import {select} from './kernels/Select';
import {concatImpl} from './kernels/Concat_impl';
import {createSimpleBinaryKernelImpl} from './utils/binary_impl';
import {cumsum} from './kernels/Cumsum';
import {cumprod} from './kernels/Cumprod';
import {mirrorPad} from './kernels/MirrorPad';
import {depthToSpace} from './kernels/DepthToSpace';
import {concat} from './kernels/Concat';
import {flipLeftRightConfig} from './kernels/FlipLeftRight';
import {imageDimensions, matrixBatchStrides, matrixDimensions, rankFiveStrides, spatialStrides2D} from './cpu_util';
import {batchMatMul} from './kernels/BatchMatMul';
import {sum} from './kernels/Sum';
import {conv2D} from './kernels/Conv2D';
import {conv2DBackpropInput} from './kernels/Conv2DBackpropInput';
import {conv3D} from './kernels/Conv3D';
import {conv3DBackpropFilterV2} from './kernels/Conv3DBackpropFilterV2';
import {conv3DBackpropInputV2} from './kernels/Conv3DBackpropInputV2';
import {depthwiseConv2dNative} from './kernels/DepthwiseConv2dNative';
import {depthwiseConv2dNativeBackpropInput} from './kernels/DepthwiseConv2dNativeBackpropInput';
import {argMin} from './kernels/ArgMin';
import {argMax} from './kernels/ArgMax';
import {complex} from './kernels/Complex';
import {fft} from './kernels/FFT';
import {ifft} from './kernels/IFFT';
import {avgPool} from './kernels/AvgPool';
import {maxPool} from './kernels/MaxPool';
import {avgPool3D} from './kernels/AvgPool3D';
import {maxPool3D} from './kernels/MaxPool3D';
import {avgPool3DGrad} from './kernels/AvgPool3DGrad';
import {maxPool3DGrad} from './kernels/MaxPool3DGrad';
import {resizeBilinear} from './kernels/ResizeBilinear';
import {resizeNearestNeighbor} from './kernels/ResizeNearestNeighbor';
import {resizeBilinearGrad} from './kernels/ResizeBilinearGrad';
import {resizeNearestNeighborGrad} from './kernels/ResizeNearestNeighborGrad';
import {nonMaxSuppressionV3} from './kernels/NonMaxSuppressionV3';
import {nonMaxSuppressionV4} from './kernels/NonMaxSuppressionV4';
import {nonMaxSuppressionV5} from './kernels/NonMaxSuppressionV5';
import {dilation2DConfig} from './kernels/Dilation2D';
import {dilation2DBackpropInputConfig} from './kernels/Dilation2DBackpropInput';
import {dilation2DBackpropFilterConfig} from './kernels/Dilation2DBackpropFilter';

import {stringNGramsImpl} from './kernels/StringNGrams_impl';
import {stringSplitImpl} from './kernels/StringSplit_impl';
import {sparseReshapeImpl} from './kernels/SparseReshape_impl';
import {sparseSegmentReductionImpl} from './kernels/SparseSegmentReduction_impl';
import {uniqueImpl} from './kernels/Unique_impl';
import {topKImpl} from './kernels/TopK_impl';
import {raggedRangeImpl} from './kernels/RaggedRange_impl';
import {raggedGatherImpl} from './kernels/RaggedGather_impl';
import {raggedTensorToTensorImpl} from './kernels/RaggedTensorToTensor_impl';
import {sparseFillEmptyRowsImpl} from './kernels/SparseFillEmptyRows_impl';

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

describe('shared CPU 3D pooling gradient traversal', () => {
  it('accumulates overlapping average and maximum windows differently', () => {
    const backend = new MathBackendCPU();
    const input = backend.makeTensorInfo([1, 1, 1, 3, 1], 'float32',
        new Float32Array([1, 3, 2]));
    const dy = backend.makeTensorInfo([1, 1, 1, 2, 1], 'float32',
        new Float32Array([2, 4]));
    const filterSize: [number, number, number] = [1, 1, 2];
    const args = {inputs: {input, dy}, backend,
      attrs: {filterSize, strides: 1, pad: 'valid' as const}};
    const avg = avgPool3DGrad(args);
    const output = maxPool3D({inputs: {x: input}, backend,
      attrs: {...args.attrs, dataFormat: 'NDHWC'}});
    expect(output.shape).toEqual(dy.shape);
    expect(backend.readSync(output.dataId)).toEqual(new Float32Array([3, 3]));
    const max = maxPool3DGrad({...args, inputs: {input, dy, output}});
    expect(avg.shape).toEqual(input.shape);
    expect(max.shape).toEqual(input.shape);
    expect(backend.readSync(avg.dataId)).toEqual(new Float32Array([1, 3, 2]));
    expect(backend.readSync(max.dataId)).toEqual(new Float32Array([0, 6, 0]));
    const nanDy = backend.makeTensorInfo(dy.shape, 'float32',
        new Float32Array([NaN, 4]));
    const nanMax = maxPool3DGrad({...args, inputs: {input, dy: nanDy, output}});
    expect(backend.readSync(nanMax.dataId)).toEqual(
        new Float32Array([0, NaN, 0]));
    expect(backend.refCount(output.dataId)).toBe(1);
    [input, dy, output, avg, max, nanDy, nanMax]
        .forEach(t => backend.disposeData(t.dataId));
    expect(backend.numDataIds()).toBe(0);
  });
});

describe('checked CPU matrix and convolution geometry', () => {
  it('retains zero strides and dimensions and rejects missing coordinates', () => {
    expect(matrixDimensions([4, 0, 3])).toEqual([0, 3]);
    expect(matrixBatchStrides([0, 3])).toEqual([0, 3]);
    expect(spatialStrides2D([0, 6, 2], true)).toEqual([0, 6, 2, 1]);
    expect(spatialStrides2D([0, 6, 2], false)).toEqual([0, 2, 1, 6]);
    expect(() => matrixDimensions([3])).toThrowError();
    expect(() => matrixBatchStrides([3])).toThrowError();
    expect(() => spatialStrides2D([3, 2], true)).toThrowError();
  });

  it('multiplies rectangular matrices with each transpose combination', () => {
    const backend = new MathBackendCPU();
    const a = backend.makeTensorInfo([2, 3], 'float32',
        new Float32Array([1, 2, 3, 4, 5, 6]));
    const b = backend.makeTensorInfo([3, 2], 'float32',
        new Float32Array([7, 8, 9, 10, 11, 12]));
    const at = backend.makeTensorInfo([3, 2], 'float32',
        new Float32Array([1, 4, 2, 5, 3, 6]));
    const bt = backend.makeTensorInfo([2, 3], 'float32',
        new Float32Array([7, 9, 11, 8, 10, 12]));
    for (const transposeA of [false, true]) {
      for (const transposeB of [false, true]) {
        const out = batchMatMul({backend, inputs: {
          a: transposeA ? at : a, b: transposeB ? bt : b
        }, attrs: {transposeA, transposeB}});
        expect(out.shape).toEqual([2, 2]);
        expect(backend.readSync(out.dataId)).toEqual(
            new Float32Array([58, 64, 139, 154]));
        backend.disposeData(out.dataId);
      }
    }
    [a, b, at, bt].forEach(t => {
      expect(backend.refCount(t.dataId)).toBe(1);
      backend.disposeData(t.dataId);
    });
    expect(backend.numDataIds()).toBe(0);
  });

  it('preserves forward and input-gradient convolution in both layouts', () => {
    const backend = new MathBackendCPU();
    const inputShape: [number, number, number, number] = [1, 2, 2, 2];
    const filter = backend.makeTensorInfo([1, 1, 2, 1], 'float32',
        new Float32Array([2, 3]));
    const layouts: Array<'NHWC'|'NCHW'> = ['NHWC', 'NCHW'];
    for (const dataFormat of layouts) {
      const channelsLast = dataFormat === 'NHWC';
      const x = backend.makeTensorInfo(inputShape, 'float32', new Float32Array(
          channelsLast ? [1, 10, 2, 20, 3, 30, 4, 40] :
                         [1, 2, 3, 4, 10, 20, 30, 40]));
      const dy = backend.makeTensorInfo(
          channelsLast ? [1, 2, 2, 1] : [1, 1, 2, 2], 'float32',
          new Float32Array([1, 2, 3, 4]));
      const attrs = {strides: 1, pad: 'valid' as const, dataFormat};
      const y = conv2D({inputs: {x, filter}, backend,
        attrs: {...attrs, dilations: 1}});
      const dx = conv2DBackpropInput({inputs: {dy, filter}, backend,
        attrs: {...attrs, inputShape}});
      expect(y.shape).toEqual(dy.shape);
      expect(backend.readSync(y.dataId)).toEqual(
          new Float32Array([32, 64, 96, 128]));
      expect(dx.shape).toEqual(inputShape);
      expect(backend.readSync(dx.dataId)).toEqual(new Float32Array(
          channelsLast ? [2, 3, 4, 6, 6, 9, 8, 12] :
                         [2, 4, 6, 8, 3, 6, 9, 12]));
      [x, dy, y, dx].forEach(t => backend.disposeData(t.dataId));
    }
    expect(backend.refCount(filter.dataId)).toBe(1);
    backend.disposeData(filter.dataId);
    expect(backend.numDataIds()).toBe(0);
  });
});

describe('checked CPU volumetric and depthwise geometry', () => {
  it('retains volumetric and image dimensions including zero', () => {
    expect(rankFiveStrides([0, 6, 3, 1])).toEqual([0, 6, 3, 1]);
    expect(imageDimensions([2, 0, 3, 1])).toEqual([2, 0, 3, 1]);
    expect(() => rankFiveStrides([6, 3, 1])).toThrowError();
    expect(() => imageDimensions([2, 3, 1])).toThrowError();
  });

  it('preserves volumetric forward, input gradient and filter gradient', () => {
    const backend = new MathBackendCPU();
    const inputShape: [number, number, number, number, number] =
        [1, 2, 1, 2, 1];
    const filterShape: [number, number, number, number, number] =
        [1, 1, 1, 1, 1];
    const x = backend.makeTensorInfo(inputShape, 'float32',
        new Float32Array([1, 2, 3, 4]));
    const filter = backend.makeTensorInfo(filterShape, 'float32',
        new Float32Array([2]));
    const dy = backend.makeTensorInfo(inputShape, 'float32',
        new Float32Array([1, 2, 3, 4]));
    const attrs = {strides: 1, pad: 'valid' as const};
    const y = conv3D({inputs: {x, filter}, backend,
      attrs: {...attrs, dilations: 1, dataFormat: 'NDHWC'}});
    const dx = conv3DBackpropInputV2({inputs: {dy, filter}, backend,
      attrs: {...attrs, inputShape}});
    const dw = conv3DBackpropFilterV2({inputs: {x, dy}, backend,
      attrs: {...attrs, filterShape}});
    expect(y.shape).toEqual(inputShape);
    expect(dx.shape).toEqual(inputShape);
    expect(dw.shape).toEqual(filterShape);
    expect(backend.readSync(y.dataId)).toEqual(new Float32Array([2, 4, 6, 8]));
    expect(backend.readSync(dx.dataId)).toEqual(new Float32Array([2, 4, 6, 8]));
    expect(backend.readSync(dw.dataId)).toEqual(new Float32Array([30]));
    [x, filter, dy, y, dx, dw].forEach(t => backend.disposeData(t.dataId));
    expect(backend.numDataIds()).toBe(0);
  });

  it('keeps separate depthwise channels and multipliers', () => {
    const backend = new MathBackendCPU();
    const inputShape: [number, number, number, number] = [1, 1, 2, 2];
    const x = backend.makeTensorInfo(inputShape, 'float32',
        new Float32Array([1, 2, 3, 4]));
    const filter = backend.makeTensorInfo([1, 1, 2, 2], 'float32',
        new Float32Array([2, 3, 4, 5]));
    const dy = backend.makeTensorInfo([1, 1, 2, 4], 'float32',
        new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]));
    const attrs = {strides: 1, dilations: 1, pad: 'valid' as const};
    const y = depthwiseConv2dNative({inputs: {x, filter}, backend,
      attrs: {...attrs, dataFormat: 'NHWC'}});
    const dx = depthwiseConv2dNativeBackpropInput({inputs: {dy, filter},
      backend, attrs: {...attrs, inputShape}});
    expect(y.shape).toEqual(dy.shape);
    expect(backend.readSync(y.dataId)).toEqual(
        new Float32Array([2, 3, 8, 10, 6, 9, 16, 20]));
    expect(dx.shape).toEqual(inputShape);
    expect(backend.readSync(dx.dataId)).toEqual(
        new Float32Array([8, 32, 28, 68]));
    [x, filter, dy, y, dx].forEach(t => backend.disposeData(t.dataId));
    expect(backend.numDataIds()).toBe(0);
  });
});

describe('checked CPU ragged and sparse metadata', () => {
  it('retains scalar broadcast, descending ranges and empty rows', () => {
    const scalar = raggedRangeImpl(new Int32Array([0]), [], 'int32',
        new Int32Array([3]), [], new Int32Array([1]), []);
    expect(scalar).toEqual([new Int32Array([0, 3]), new Int32Array([0, 1, 2])]);
    const rows = raggedRangeImpl(new Int32Array([4, 1]), [2], 'int32',
        new Int32Array([0, 0]), [2], new Int32Array([-2, 1]), [2]);
    expect(rows).toEqual([new Int32Array([0, 2, 2]), new Int32Array([4, 2])]);
    expect(() => raggedRangeImpl(new Int32Array(0), [], 'int32',
        new Int32Array([3]), [], new Int32Array([1]), [])).toThrowError();
  });

  it('gathers ragged rows in requested order, including empty rows', () => {
    const output = raggedGatherImpl([new Int32Array([0, 2, 2, 3])], [[4]],
        new Int32Array([10, 20, 30]), [3], 'int32',
        new Int32Array([2, 0, 1]), [3], 1);
    expect(output).toEqual([[new Int32Array([0, 1, 3, 3])],
      new Int32Array([30, 10, 20]), [3]]);
  });

  it('densifies an inferred shape while preserving empty rows and padding', () => {
    const output = raggedTensorToTensorImpl(new Int32Array([-1]), [],
        new Int32Array([1, 2, 3]), [3], 'int32', new Int32Array([0]), [],
        [new Int32Array([0, 2, 2, 3])], [[4]], ['ROW_SPLITS']);
    expect(output).toEqual([[3, 2], new Int32Array([1, 2, 0, 0, 3, 0])]);
  });

  it('infers dense trailing dimensions and checks explicit shape constraints', () => {
    const values = new Int32Array([1, 2, 3, 4, 5, 6]);
    const partitions = [new Int32Array([0, 2, 2, 3])];
    const densify = (shape: Int32Array, shapeShape: number[]) =>
        raggedTensorToTensorImpl(shape, shapeShape, values, [3, 2],
            'int32', new Int32Array([-9]), [], partitions, [[4]],
            ['ROW_SPLITS']);
    const expected = [[3, 2, 2],
      new Int32Array([1, 2, 3, 4, -9, -9, -9, -9, 5, 6, -9, -9])];
    expect(densify(new Int32Array([-1]), [])).toEqual(expected);
    expect(densify(new Int32Array([-1, -1, -1]), [3])).toEqual(expected);
    expect(densify(new Int32Array([3, 2, 2]), [3])).toEqual(expected);
    expect(() => densify(new Int32Array([3, 2, 4]), [3]))
        .toThrowError(/are incompatible/);
    expect(() => densify(new Int32Array([0]), []))
        .toThrowError(/only valid scalar shape tensor/);
    expect(() => densify(new Int32Array([3, 2]), [2]))
        .toThrowError(/are incompatible/);
  });

  it('fills sparse rows, restores row order and retains reverse indices', () => {
    const output = sparseFillEmptyRowsImpl(new Int32Array([2, 0, 0, 1]),
        [2, 2], 'int32', new Int32Array([5, 7]), 'int32',
        new Int32Array([3, 2]), -1);
    expect(output).toEqual([new Int32Array([0, 1, 1, 0, 2, 0]), [3, 2],
      new Int32Array([7, -1, 5]), [false, true, false], [2, 0]]);
  });
});

describe('checked CPU string and index kernels', () => {
  it('preserves empty string tokens, short sequences and ragged splits', () => {
    const a = new Uint8Array([97]);
    const b = new Uint8Array([98]);
    expect(stringNGramsImpl([a, b], new Int32Array([0, 2, 2]),
        ' ', [2], '', '', 0, true)).toEqual([
      [new Uint8Array([97, 32, 98])], new Int32Array([0, 1, 1])]);
    expect(stringSplitImpl([new Uint8Array([97, 44, 44, 98])],
        new Uint8Array([44]), false)).toEqual([
      new Int32Array([0, 0, 0, 1, 0, 2]), [a, new Uint8Array(0), b], [1, 3]]);
  });
  it('preserves inferred sparse dimensions, segment gaps and means', () => {
    expect(sparseReshapeImpl(new Int32Array([1, 2]), [1, 2], 'int32',
        [2, 3], [3, -1])).toEqual([new Int32Array([2, 1]), [1, 2], [3, 2]]);
    expect(sparseSegmentReductionImpl(new Float32Array([2, 4, 8]), [3, 1],
        'float32', new Int32Array([0, 1, 2]), new Int32Array([0, 0, 2]),
        true, -1)).toEqual([new Float32Array([3, -1, 8]), [3, 1]]);
  });
  it('preserves first-occurrence unique order and stable TopK ties', () => {
    const unique = uniqueImpl(new Int32Array([4, 2, 4]), 0, [3], 'int32');
    expect(unique).toEqual({outputValues: new Int32Array([4, 2]),
      outputShape: [2], indices: new Int32Array([0, 1, 0])});
    const [values, indices] = topKImpl(new Float32Array([4, 2, 4]),
        [3], 'float32', 2, true);
    expect(values.values).toEqual(new Float32Array([4, 4]));
    expect(indices.values).toEqual(new Int32Array([0, 2]));
  });
});


describe('checked CPU cumulative and image geometry', () => {
  it('preserves exclusive reverse accumulation and numeric identities', () => {
    const backend = new MathBackendCPU();
    const x = backend.makeTensorInfo([2, 3], 'float32',
        new Float32Array([1, 2, 3, 4, 0, 2]));
    const args = {inputs: {x}, backend, attrs: {axis: 1, reverse: true, exclusive: true}};
    const sum = cumsum(args);
    const product = cumprod(args);
    expect(backend.readSync(sum.dataId)).toEqual(new Float32Array([5, 3, 0, 2, 2, 0]));
    expect(backend.readSync(product.dataId)).toEqual(new Float32Array([6, 3, 1, 0, 2, 1]));
    [x, sum, product].forEach(t => backend.disposeData(t.dataId));
  });

  it('preserves reflected and symmetric edges, including zeros', () => {
    const backend = new MathBackendCPU();
    const x = backend.makeTensorInfo([3], 'float32', new Float32Array([0, 2, 4]));
    const paddings: Array<[number, number]> = [[1, 1]];
    const reflect = mirrorPad({inputs: {x}, backend, attrs: {paddings, mode: 'reflect'}});
    const symmetric = mirrorPad({inputs: {x}, backend, attrs: {paddings, mode: 'symmetric'}});
    expect(backend.readSync(reflect.dataId)).toEqual(new Float32Array([2, 0, 2, 4, 2]));
    expect(backend.readSync(symmetric.dataId)).toEqual(new Float32Array([0, 0, 2, 4, 4]));
    [x, reflect, symmetric].forEach(t => backend.disposeData(t.dataId));
  });

  it('maps NHWC depth and flips each row without changing channel order', () => {
    const backend = new MathBackendCPU();
    const x = backend.makeTensorInfo([1, 1, 1, 4], 'float32', new Float32Array([0, 1, 2, 3]));
    const expanded = depthToSpace({inputs: {x}, backend, attrs: {blockSize: 2, dataFormat: 'NHWC'}});
    expect(expanded.shape).toEqual([1, 2, 2, 1]);
    expect(backend.readSync(expanded.dataId)).toEqual(new Float32Array([0, 1, 2, 3]));
    const flipped = flipLeftRightConfig.kernelFunc({inputs: {image: expanded}, backend});
    if (Array.isArray(flipped)) {
      throw new Error('Expected one flipped image.');
    }
    expect(backend.readSync(flipped.dataId)).toEqual(new Float32Array([1, 0, 3, 2]));
    [x, expanded, flipped].forEach(t => backend.disposeData(t.dataId));
  });

  it('concatenates nonempty matrices by rows while ignoring empty inputs', () => {
    const backend = new MathBackendCPU();
    const a = backend.makeTensorInfo([2, 1], 'float32', new Float32Array([0, 2]));
    const b = backend.makeTensorInfo([2, 2], 'float32', new Float32Array([3, 4, 5, 6]));
    const empty = backend.makeTensorInfo([2, 0], 'float32', new Float32Array());
    const result = concat({inputs: [a, empty, b], backend, attrs: {axis: -1}});
    expect(result.shape).toEqual([2, 3]);
    expect(backend.readSync(result.dataId)).toEqual(new Float32Array([0, 3, 4, 2, 5, 6]));
    [a, b, empty, result].forEach(t => backend.disposeData(t.dataId));
  });
});

describe('checked CPU kernel reads', () => {
  it('transposes, splits and unpacks zero-valued entries in order', () => {
    const backend = new MathBackendCPU();
    const x = backend.makeTensorInfo([2, 3], 'float32',
        new Float32Array([0, 1, 2, 3, 4, 5]));
    const transposed = transpose({inputs: {x}, backend, attrs: {perm: [1, 0]}});
    expect(transposed.shape).toEqual([3, 2]);
    expect(backend.readSync(transposed.dataId)).toEqual(new Float32Array([0, 3, 1, 4, 2, 5]));
    const splits = splitV({inputs: {x}, backend, attrs: {axis: 1, numOrSizeSplits: [1, 2]}});
    expect(splits.map(t => t.shape)).toEqual([[2, 1], [2, 2]]);
    expect(splits.map(t => backend.readSync(t.dataId))).toEqual([
      new Float32Array([0, 3]), new Float32Array([1, 2, 4, 5])]);
    const slices = unpack({inputs: {value: x}, backend, attrs: {axis: -1}});
    expect(slices.map(t => backend.readSync(t.dataId))).toEqual([
      new Float32Array([0, 3]), new Float32Array([1, 4]), new Float32Array([2, 5])]);
    [x, transposed, ...splits, ...slices].forEach(t => backend.disposeData(t.dataId));
  });

  it('preserves zero, negative and NaN activation values', () => {
    const backend = new MathBackendCPU();
    const x = backend.makeTensorInfo([4], 'float32', new Float32Array([0, -2, NaN, 3]));
    const result = leakyRelu({inputs: {x}, backend, attrs: {alpha: 0.5}});
    expect(backend.readSync(result.dataId)).toEqual(new Float32Array([0, -1, NaN, 3]));
    backend.disposeData(x.dataId);
    backend.disposeData(result.dataId);
  });

  it('samples the only available event without a CDF entry', () => {
    const backend = new MathBackendCPU();
    const logits = backend.makeTensorInfo([2, 1], 'float32', new Float32Array([1, 1]));
    const result = multinomial({inputs: {logits}, backend,
      attrs: {numSamples: 3, seed: 42, normalized: true}});
    expect(result.shape).toEqual([2, 3]);
    expect(backend.readSync(result.dataId)).toEqual(new Int32Array(6));
    backend.disposeData(logits.dataId);
    backend.disposeData(result.dataId);
  });
});

describe('typed CPU reduction result paths', () => {
  it('retains reduction shape, values and input ownership with keepDims', () => {
    const backend = new MathBackendCPU();
    const x = backend.makeTensorInfo([2, 2], 'float32', new Float32Array([1, 2, 3, 4]));
    for (const keepDims of [false, true]) {
      const out = sum({inputs: {x}, backend, attrs: {axis: [1], keepDims}});
      expect(out.shape).toEqual(keepDims ? [2, 1] : [2]);
      expect(backend.readSync(out.dataId)).toEqual(new Float32Array([3, 7]));
      backend.disposeData(out.dataId);
    }
    expect(backend.refCount(x.dataId)).toBe(1);
    backend.disposeData(x.dataId);
    expect(backend.numDataIds()).toBe(0);
  });
});

describe('shared CPU dilation maximum and gradient dispatch', () => {
  const cases = [
    {
      name: 'retains the first tied maximum',
      inputShape: [1, 2, 2, 1], input: [2, 2, 1, 0],
      filterShape: [2, 2, 1], filter: [0, 0, 0, 0],
      outputShape: [1, 1, 1, 1], dy: [3], pad: 'valid',
      output: [2], dx: [3, 0, 0, 0], df: [3, 0, 0, 0]
    },
    {
      name: 'accumulates repeated maxima with asymmetric padding',
      inputShape: [1, 1, 3, 1], input: [1, 3, 2],
      filterShape: [1, 2, 1], filter: [0, 0],
      outputShape: [1, 1, 3, 1], dy: [1, 2, 3], pad: 'same',
      output: [3, 3, 2], dx: [0, 3, 3], df: [5, 1]
    }
  ];
  for (const testCase of cases) {
    it(testCase.name, () => {
      const backend = new MathBackendCPU();
      const x = backend.makeTensorInfo(testCase.inputShape, 'float32',
          new Float32Array(testCase.input));
      const filter = backend.makeTensorInfo(testCase.filterShape, 'float32',
          new Float32Array(testCase.filter));
      const dy = backend.makeTensorInfo(testCase.outputShape, 'float32',
          new Float32Array(testCase.dy));
      const attrs = {strides: 1, pad: testCase.pad, dilations: 1};
      const forward = dilation2DConfig.kernelFunc({inputs: {x, filter}, backend, attrs});
      const inputs = {x, filter, dy: {...dy, rank: 4}};
      const dx = dilation2DBackpropInputConfig.kernelFunc({inputs, backend, attrs});
      const df = dilation2DBackpropFilterConfig.kernelFunc({inputs, backend, attrs});
      if (Array.isArray(forward) || Array.isArray(dx) || Array.isArray(df)) {
        throw new Error('Dilation kernels must return a single tensor.');
      }
      expect(forward.shape).toEqual(testCase.outputShape);
      expect(dx.shape).toEqual(x.shape);
      expect(df.shape).toEqual(filter.shape);
      expect(backend.readSync(forward.dataId)).toEqual(new Float32Array(testCase.output));
      expect(backend.readSync(dx.dataId)).toEqual(new Float32Array(testCase.dx));
      expect(backend.readSync(df.dataId)).toEqual(new Float32Array(testCase.df));
      [x, filter, dy, forward, dx, df].forEach(t => backend.disposeData(t.dataId));
      expect(backend.numDataIds()).toBe(0);
    });
  }
});

describe('CPU reduction and checked numeric traversal', () => {
  it('preserves transposed boolean reductions, retained axes and ownership', () => {
    const backend = new MathBackendCPU();
    const x = backend.makeTensorInfo([2, 2], 'bool', new Uint8Array([1, 0, 1, 1]));
    const args = {inputs: {x}, backend, attrs: {axis: [0], keepDims: true}};
    const a = all(args);
    const b = anyReduction(args);
    expect(a.shape).toEqual([1, 2]);
    expect(b.shape).toEqual([1, 2]);
    expect(backend.readSync(a.dataId)).toEqual(new Uint8Array([1, 0]));
    expect(backend.readSync(b.dataId)).toEqual(new Uint8Array([1, 1]));
    [a, b, x].forEach(tensor => backend.disposeData(tensor.dataId));
    expect(backend.numDataIds()).toBe(0);
  });

  it('propagates NaN and retains the first signed zero in min and max', () => {
    const backend = new MathBackendCPU();
    const x = backend.makeTensorInfo([2, 3], 'float32',
        new Float32Array([3, NaN, 1, -0, 0, -0]));
    const a = min({inputs: {x}, backend, attrs: {axis: [1], keepDims: false}});
    const b = max({inputs: {x}, backend, attrs: {reductionIndices: [1], keepDims: false}});
    expect(backend.readSync(a.dataId)).toEqual(new Float32Array([NaN, -0]));
    expect(backend.readSync(b.dataId)).toEqual(new Float32Array([NaN, -0]));
    [a, b, x].forEach(tensor => backend.disposeData(tensor.dataId));
    expect(backend.numDataIds()).toBe(0);
  });

  it('preserves historical empty-block conversions', () => {
    const backend = new MathBackendCPU();
    const x = backend.makeTensorInfo([2, 0], 'bool', new Uint8Array(0));
    const y = backend.makeTensorInfo([2, 0], 'float32', new Float32Array(0));
    const args = {inputs: {x}, backend, attrs: {axis: [1], keepDims: false}};
    const a = all(args);
    const b = anyReduction(args);
    const c = min({inputs: {x: y}, backend, attrs: {axis: [1], keepDims: false}});
    const d = max({inputs: {x: y}, backend, attrs: {reductionIndices: [1], keepDims: false}});
    expect(backend.readSync(a.dataId)).toEqual(new Uint8Array([0, 0]));
    expect(backend.readSync(b.dataId)).toEqual(new Uint8Array([0, 0]));
    expect(backend.readSync(c.dataId)).toEqual(new Float32Array([NaN, NaN]));
    expect(backend.readSync(d.dataId)).toEqual(new Float32Array([NaN, NaN]));
    [a, b, c, d, x, y].forEach(tensor => backend.disposeData(tensor.dataId));
    expect(backend.numDataIds()).toBe(0);
  });

  it('retains batch normalisation defaults and cyclic parameter indexing', () => {
    const backend = new MathBackendCPU();
    const x = backend.makeTensorInfo([2, 2], 'float32', new Float32Array([1, 2, 3, 4]));
    const mean = backend.makeTensorInfo([2], 'float32', new Float32Array([1, 2]));
    const variance = backend.makeTensorInfo([2], 'float32', new Float32Array([1, 4]));
    const scale = backend.makeTensorInfo([2], 'float32', new Float32Array([2, 4]));
    const offset = backend.makeTensorInfo([2], 'float32', new Float32Array([1, 2]));
    const attrs = {varianceEpsilon: 0};
    const a = batchNorm({inputs: {x, mean, variance, scale: null, offset: null}, backend, attrs});
    const b = batchNorm({inputs: {x, mean, variance, scale, offset}, backend, attrs});
    expect(backend.readSync(a.dataId)).toEqual(new Float32Array([0, 0, 2, 1]));
    expect(backend.readSync(b.dataId)).toEqual(new Float32Array([1, 2, 5, 6]));
    [a, b, x, mean, variance, scale, offset].forEach(tensor => backend.disposeData(tensor.dataId));
    expect(backend.numDataIds()).toBe(0);
  });

  it('selects only the chosen values and retains zero', () => {
    const backend = new MathBackendCPU();
    const condition = backend.makeTensorInfo([3], 'bool', new Uint8Array([1, 0, 1]));
    const t = backend.makeTensorInfo([3], 'float32', new Float32Array([0, NaN, 3]));
    const e = backend.makeTensorInfo([3], 'float32', new Float32Array([NaN, 2, NaN]));
    const result = select({inputs: {condition, t, e}, backend});
    expect(backend.readSync(result.dataId)).toEqual(new Float32Array([0, 2, 3]));
    [condition, t, e, result].forEach(tensor => backend.disposeData(tensor.dataId));
    expect(backend.numDataIds()).toBe(0);
  });

  it('checks concatenation storage without changing interleaving', () => {
    const inputs = [
      {vals: new Float32Array([0, 2]), shape: [2, 1]},
      {vals: new Float32Array([1, 3]), shape: [2, 1]}
    ];
    expect(concatImpl(inputs, [2, 2], 'float32', false))
        .toEqual(new Float32Array([0, 1, 2, 3]));
    expect(() => concatImpl([{vals: new Float32Array(0), shape: [1, 1]}],
        [1, 1], 'float32', false)).toThrowError();
  });

  it('preserves binary broadcasting, string payloads and empty output', () => {
    const subtract = createSimpleBinaryKernelImpl((a, b) => Number(a) - Number(b));
    expect(subtract([2, 1], [1, 2], new Float32Array([0, 3]),
        new Float32Array([1, 2]), 'float32'))
        .toEqual([new Float32Array([-1, -2, 2, 1]), [2, 2]]);
    expect(subtract([2], [1], ['', '2'], ['1'], 'float32'))
        .toEqual([new Float32Array([-1, 1]), [2]]);
    expect(subtract([0], [1], new Float32Array(0), new Float32Array([1]), 'float32'))
        .toEqual([new Float32Array(0), [0]]);
  });
});
