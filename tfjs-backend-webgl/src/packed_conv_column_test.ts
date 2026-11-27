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

import {ConcatPackedProgram} from './concat_packed_gpu';
import {ReversePackedProgram} from './reverse_packed_gpu';
import {shaderEntry} from './packing_util';
import {getLogicalCoordinatesFromFlatIndexByUniform} from './shader_compiler_util';
import {getRowsCols, getShapeAs3D, isReshapeFree} from './webgl_util';
import {TensorInfo} from '@tensorflow/tfjs-core';
import {makeSelectedIndices, readNonMaxSuppressionInputs} from './kernels/non_max_suppression_util';
import {selectActivationSource} from './activation_source';
import {packedConvColumnSource} from './packed_conv_column';
import {resizeAxisRatio} from './resize_geometry';
import {ResizeBilinearProgram} from './resize_bilinear_gpu';
import {ResizeBilinearPackedProgram} from './resize_bilinear_packed_gpu';
import {ResizeNearestNeighborProgram} from './resize_nearest_neighbor_gpu';
import {ResizeNearestNeighborPackedProgram} from './resize_nearest_neighbor_packed_gpu';
import {avgPool, avgPoolConfig} from './kernels/AvgPool';
import {maxPool, maxPoolConfig} from './kernels/MaxPool';
import {batchNorm, batchNormConfig} from './kernels/BatchNorm';
import {relu, reluConfig} from './kernels/Relu';
import {fill, fillConfig} from './kernels/Fill';
import {linSpace, linSpaceConfig} from './kernels/LinSpace';
import {range, rangeConfig} from './kernels/Range';
import {resizeBilinearGrad, resizeBilinearGradConfig} from './kernels/ResizeBilinearGrad';
import {resizeNearestNeighborGrad, resizeNearestNeighborGradConfig} from './kernels/ResizeNearestNeighborGrad';

describe('packed convolution sampling source', () => {
  it('does not emit a sample beyond the filter boundary', () => {
    expect(packedConvColumnSource(4, 4, 1, 0, 1)).toBe('');
    expect(packedConvColumnSource(4, 4, 2, 1, 2)).toBe('');
  });

  it('reuses the adjacent texel for odd padding and unit dilation', () => {
    const code = packedConvColumnSource(2, 4, 1, 1, 1);
    expect(code).toContain('xC2 = vec4(xTexelC0.zw, xTexelC2.xy)');
    expect(code).toContain('xC3 = xTexelC2;');
    expect(code).not.toContain('previous = getX');
  });

  it('samples the previous texel for the first odd-padded column', () => {
    const code = packedConvColumnSource(0, 1, 1, 1, 1);
    expect(code).toContain('xCOffset = xC + 1 - 2;');
    expect(code).toContain('previous = getX');
    expect(code).not.toContain('xC1 =');
  });

  it('uses separate texel reads when dilation prevents sharing', () => {
    const code = packedConvColumnSource(0, 2, 1, 1, 2);
    expect(code).toContain('xCOffset = xC + imod(pads[1], 2) + 2;');
    expect(code).toContain('xCOffset -= 2;');
    expect(code).toContain('xC1 = vec4(previous.zw, xTexelC1.xy)');
  });

  it('retains stride-two odd and even padding paths', () => {
    const odd = packedConvColumnSource(0, 2, 2, 1, 1);
    expect(odd).toContain('xCOffset = xC + 1 - strides[1];');
    expect(odd).toContain('final = getX');
    expect(odd).toContain('xC1 = vec4(xTexelC1.xy, final.xy);');
    const even = packedConvColumnSource(0, 2, 2, 0, 1);
    expect(even).toContain('xCOffset = xC + strides[1];');
    expect(even).toContain('xC1 = vec4(xTexelC0.zw, xTexelC1.zw);');
    expect(even).not.toContain('final = getX');
  });

  it('does not emit an unused second output column for odd filter width', () => {
    expect(packedConvColumnSource(2, 3, 1, 0, 1)).not.toContain('xC3 =');
    expect(packedConvColumnSource(2, 3, 2, 1, 1)).not.toContain('xC3 =');
    expect(packedConvColumnSource(2, 3, 2, 0, 1)).not.toContain('xC3 =');
  });
});

describe('ordered fused activation source', () => {
  it('only evaluates the selected fragment, retaining PReLU precedence', () => {
    const prelu = jasmine.createSpy('prelu').and.returnValue('prelu');
    const leaky = jasmine.createSpy('leaky').and.returnValue('leaky');
    const plain = jasmine.createSpy('plain').and.returnValue('plain');
    expect(selectActivationSource('', true, true, prelu, leaky, plain)).toBe('');
    expect(prelu).not.toHaveBeenCalled();
    expect(leaky).not.toHaveBeenCalled();
    expect(plain).not.toHaveBeenCalled();
    expect(selectActivationSource('a', true, true, prelu, leaky, plain))
        .toBe('prelu');
    expect(prelu).toHaveBeenCalledTimes(1);
    expect(leaky).not.toHaveBeenCalled();
    expect(plain).not.toHaveBeenCalled();
    expect(selectActivationSource('a', false, true, prelu, leaky, plain))
        .toBe('leaky');
    expect(selectActivationSource('a', false, false, prelu, leaky, plain))
        .toBe('plain');
    expect(leaky).toHaveBeenCalledTimes(1);
    expect(plain).toHaveBeenCalledTimes(1);
  });
});

describe('WebGL kernel registration contracts', () => {
  it('retains the distinct pooling and resize registration entry points', () => {
    expect(avgPoolConfig.kernelName).toBe('AvgPool');
    expect(maxPoolConfig.kernelName).toBe('MaxPool');
    expect(avgPoolConfig.kernelFunc).toBe(avgPool);
    expect(maxPoolConfig.kernelFunc).toBe(maxPool);
    expect(resizeBilinearGradConfig.kernelName).toBe('ResizeBilinearGrad');
    expect(resizeNearestNeighborGradConfig.kernelName).toBe('ResizeNearestNeighborGrad');
    expect(resizeBilinearGradConfig.kernelFunc).toBe(resizeBilinearGrad);
    expect(resizeNearestNeighborGradConfig.kernelFunc).toBe(resizeNearestNeighborGrad);
  });

  it('registers attribute-only creation kernels without wrappers', () => {
    expect(fillConfig.kernelName).toBe('Fill');
    expect(fillConfig.backendName).toBe('webgl');
    expect(fillConfig.kernelFunc).toBe(fill);
    expect(linSpaceConfig.kernelName).toBe('LinSpace');
    expect(linSpaceConfig.backendName).toBe('webgl');
    expect(linSpaceConfig.kernelFunc).toBe(linSpace);
    expect(rangeConfig.kernelName).toBe('Range');
    expect(rangeConfig.backendName).toBe('webgl');
    expect(rangeConfig.kernelFunc).toBe(range);
  });

  it('registers named-input and factory kernels without a wrapper', () => {
    expect(batchNormConfig.backendName).toBe('webgl');
    expect(batchNormConfig.kernelName).toBe('FusedBatchNorm');
    expect(batchNormConfig.kernelFunc).toBe(batchNorm);
    expect(reluConfig.backendName).toBe('webgl');
    expect(reluConfig.kernelFunc).toBe(relu);
  });
});

describe('WebGL forward resize geometry', () => {
  it('retains singleton, aligned and unaligned axis ratios', () => {
    expect(resizeAxisRatio(7, 1, true)).toBe(7);
    expect(resizeAxisRatio(7, 4, true)).toBe(2);
    expect(resizeAxisRatio(7, 4, false)).toBe(1.75);
  });

  it('emits matching geometry for packed and unpacked interpolation programs', () => {
    const shape: [number, number, number, number] = [1, 7, 4, 3];
    const bilinear = new ResizeBilinearProgram(shape, 4, 1, true, false);
    const packedBilinear = new ResizeBilinearPackedProgram(shape, 4, 1, true, false);
    const nearest = new ResizeNearestNeighborProgram(shape, 4, 1, true, false);
    const packedNearest = new ResizeNearestNeighborPackedProgram(shape, 4, 1, true, false);
    [bilinear, packedBilinear, nearest, packedNearest].forEach(program => {
      expect(program.outputShape).toEqual([1, 4, 1, 3]);
      expect(program.userCode).toMatch(
          /effectiveInputOverOutputRatioRC = vec[23]\(\s*2,\s*4(?:,\s*4)?\);/);
    });
    expect(bilinear.userCode).toContain('sourceFloorRC');
    expect(nearest.userCode).toContain('sourceNearestRC');
    expect(packedBilinear.packedOutput).toBe(true);
    expect(packedNearest.packedOutput).toBe(true);
  });
});

describe('WebGL NMS readback and output contract', () => {
  it('reads boxes before scores and retains the selected index vector', () => {
    const boxes: TensorInfo = {dataId: {}, shape: [1, 4], dtype: 'float32'};
    const scores: TensorInfo = {dataId: {}, shape: [1], dtype: 'float32'};
    const boxesVals = new Float32Array([0, 0, 1, 1]);
    const scoresVals = new Float32Array([0.9]);
    const reads: TensorInfo['dataId'][] = [];
    const values = readNonMaxSuppressionInputs(boxes, scores, {
      readSync(dataId) {
        reads.push(dataId);
        return dataId === boxes.dataId ? boxesVals : scoresVals;
      }
    });
    expect(reads).toEqual([boxes.dataId, scores.dataId]);
    expect(values[0]).toBe(boxesVals);
    expect(values[1]).toBe(scoresVals);
    const result = makeSelectedIndices({
      makeTensorInfo(shape, dtype, data) {
        expect(data).toEqual(new Int32Array([2, 0]));
        return {dataId: {}, shape, dtype};
      }
    }, [2, 0]);
    expect(result.shape).toEqual([2]);
    expect(result.dtype).toBe('int32');
  });
});

describe('checked WebGL texture dimensions', () => {
  it('preserves vector, matrix, scalar and zero dimensions', () => {
    expect(getRowsCols([0])).toEqual([1, 0]);
    expect(getRowsCols([2, 3, 5])).toEqual([3, 5]);
    expect(getShapeAs3D([])).toEqual([1, 1, 1]);
    expect(getShapeAs3D([2, 3, 5])).toEqual([2, 3, 5]);
    expect(() => getRowsCols([])).toThrowError();
  });
  it('preserves packed reshape decisions for vectors and matrices', () => {
    expect(isReshapeFree([6], [2, 3])).toBe(false);
    expect(isReshapeFree([4], [1, 4])).toBe(true);
    expect(isReshapeFree([4], [6])).toBe(true);
    expect(isReshapeFree([3], [5])).toBe(false);
    expect(isReshapeFree([2, 3], [4, 3])).toBe(true);
    expect(isReshapeFree([2, 3], [2, 4])).toBe(false);
    expect(isReshapeFree([0, 3], [1, 2])).toBe(true);
  });
});

describe('checked packed shader metadata', () => {
  it('preserves rank-one and rank-two reverse coordinates', () => {
    const vector = new ReversePackedProgram([3], [0]);
    expect(vector.outputShape).toEqual([3]);
    expect(vector.userCode).toContain('if(rc + 1 < 3)');
    expect(vector.userCode).not.toContain('undefined');
    const matrix = new ReversePackedProgram([2, 3], [1]);
    expect(matrix.userCode).toContain('rc.y + 1 < 3');
    expect(matrix.userCode).toContain('rc.x + 1 < 2');
    expect(matrix.userCode).not.toContain('undefined');
  });
  it('preserves concatenation offsets and dimensions', () => {
    const program = new ConcatPackedProgram([[2, 3], [2, 4]], 1);
    expect(program.outputShape).toEqual([2, 7]);
    expect(program.userCode).toContain('if (y < 3)');
    expect(program.userCode).toContain('coords.y < 7');
    expect(program.userCode).not.toContain('undefined');
  });
  it('retains symbolic strides and checks missing entries', () => {
    expect(getLogicalCoordinatesFromFlatIndexByUniform(['x'], 'shape')).toBe('');
    expect(getLogicalCoordinatesFromFlatIndexByUniform(['x', 'y'], 'shape'))
        .toBe('int x = index / shape[1]; int y = index - x * shape[1];');
    expect(shaderEntry([0], 0)).toBe(0);
    expect(shaderEntry([''], 0)).toBe('');
    expect(() => shaderEntry([], 0)).toThrowError();
  });
});
