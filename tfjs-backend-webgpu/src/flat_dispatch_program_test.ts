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

import {shaderEntry, symbolicallyComputeStrides} from './shader_util';
import {concatOffsetUniforms} from './kernels/Concat_impl';
import {MatMulPackedProgram} from './matmul_packed_webgpu';
import {MatMulSplitKProgram} from './matmul_splitK_webgpu';
import {computeWorkgroupInfoForMatMul, MatMulProgramType, requiredWebGPUFeatures, selectMatMulProgramType} from './webgpu_util';
import {TileProgram} from './tile_webgpu';
import {TransposeSharedProgram} from './transpose_shared_webgpu';
import {TransposeProgram} from './transpose_webgpu';
import {getCoordsFromIndexSnippet, shouldPrintShader} from './webgpu_program';
import {binaryVariableComponents} from './binary_op_webgpu';
import {Conv2DMMProgram} from './conv2d_mm_webgpu';
import {SoftmaxProgram} from './softmax_webgpu';
import {CropAndResizeProgram} from './crop_and_resize_webgpu';
import {backend_util, TensorInfo} from '@tensorflow/tfjs-core';
import {pool2DBackpropUniforms, pool3DBackpropUniforms} from './kernels/pool_backprop_uniforms';
import {makeSelectedIndices, readNonMaxSuppressionInputs} from './kernels/non_max_suppression_util';
import {ClipVec4Program} from './clip_vec4_webgpu';
import {FillProgram} from './fill_webgpu';
import {OneHotProgram} from './onehot_webgpu';
import {ReduceProgram} from './reduce_webgpu';
import {ResizeBilinearProgram} from './resize_bilinear_webgpu';
import {MergeProgram, SwapProgram} from './top_k_webgpu';
import {resizeBackpropGeometry} from './kernels/resize_backprop_geometry';

describe('concat prefix width uniforms', () => {
  it('preserves zero widths and omits the final matrix width', () => {
    expect(concatOffsetUniforms([[1, 2]])).toEqual([]);
    expect(concatOffsetUniforms([[1, 2], [1, 0], [1, 3], [1, 7]]))
        .toEqual([{type: 'int32', data: [2]}, {type: 'int32', data: [2]},
          {type: 'int32', data: [5]}]);
  });
});

describe('flat WebGPU program dispatch', () => {
  it('retains shape identity and scalar and empty dispatches', () => {
    const shape: number[] = [];
    const scalar = new FillProgram(shape);
    expect(scalar.outputShape).toBe(shape);
    expect(scalar.dispatchLayout).toEqual({x: []});
    expect(scalar.dispatch).toEqual([1, 1, 1]);
    expect(new FillProgram([0, 3]).dispatch).toEqual([0, 1, 1]);
    expect(scalar.variableNames).toEqual([]);
    expect(scalar.size).toBe(true);
  });

  it('uses the supplied workgroup and vector width for dispatch', () => {
    const program = new ClipVec4Program([257]);
    expect(program.dispatch).toEqual([2, 1, 1]);
    expect(program.workgroupSize).toEqual([64, 1, 1]);
    expect(program.workPerThread).toBe(4);
    expect(program.outputComponent).toBe(4);
    expect(program.variableNames).toEqual(['A']);
    expect(program.shaderKey).toBe('clipVec4');
  });

  it('keeps mutable metadata separate per instance', () => {
    const first = new FillProgram([65]);
    const second = new FillProgram([65]);
    first.variableNames.push('test');
    first.workgroupSize[0] = 1;
    first.dispatchLayout.x.push(1);
    first.dispatch[0] = 7;
    expect(second.variableNames).toEqual([]);
    expect(second.workgroupSize).toEqual([64, 1, 1]);
    expect(second.dispatchLayout).toEqual({x: [0]});
    expect(second.dispatch).toEqual([2, 1, 1]);
  });

  it('preserves operation-specific metadata and shader text', () => {
    const oneHot = new OneHotProgram(5, 13);
    expect(oneHot.outputShape).toEqual([5, 13]);
    expect(oneHot.dispatch).toEqual([2, 1, 1]);
    expect(oneHot.shaderKey).toBe('onehot');
    expect(oneHot.getUserCode()).toContain('uniforms.offValue');
    const resize = new ResizeBilinearProgram([1, 2, 3, 4], 5, 6);
    expect(resize.outputShape).toEqual([1, 5, 6, 4]);
    expect(resize.dispatch).toEqual([2, 1, 1]);
    expect(resize.shaderKey).toBe('resizeBilinear');
  });

  it('preserves distinct top-k passes and their workgroup geometry', () => {
    const shape = [1, 257];
    const swap = new SwapProgram(shape);
    const merge = new MergeProgram(shape);
    expect(swap.outputShape).toBe(shape);
    expect(merge.outputShape).toBe(shape);
    expect(swap.dispatch).toEqual([2, 1, 1]);
    expect(merge.dispatch).toEqual([2, 1, 1]);
    expect(swap.shaderKey).toBe('swap');
    expect(merge.shaderKey).toBe('merge');
    expect(swap.getUserCode()).not.toBe(merge.getUserCode());
  });
});

describe('reduction dispatch selection', () => {
  it('preserves workgroup boundaries and the device limit', () => {
    expect(new ReduceProgram({batchSize: 1, inSize: 4095, outSize: 1, windowSize: 4095}, 'sum', 512)
        .workgroupSize).toEqual([64, 1, 1]);
    expect(new ReduceProgram({batchSize: 1, inSize: 4096, outSize: 1, windowSize: 4096}, 'sum', 512)
        .workgroupSize).toEqual([256, 1, 1]);
    expect(new ReduceProgram({batchSize: 1, inSize: 32768, outSize: 1, windowSize: 32768}, 'sum', 256)
        .workgroupSize).toEqual([256, 1, 1]);
    expect(new ReduceProgram({batchSize: 1, inSize: 32768, outSize: 1, windowSize: 32768}, 'sum', 512)
        .workgroupSize).toEqual([512, 1, 1]);
  });

  it('preserves operation-specific reduction code and initial values', () => {
    const info = {batchSize: 2, inSize: 8, outSize: 1, windowSize: 8};
    expect(new ReduceProgram(info, 'min', 256).getUserCode())
        .toContain('candidate < bestValue');
    expect(new ReduceProgram(info, 'max', 256).getUserCode())
        .toContain('candidate > bestValue');
    expect(new ReduceProgram(info, 'sum', 256).getUserCode())
        .toContain('bestValue = bestValue + candidate');
    expect(new ReduceProgram(info, 'mean', 256).getUserCode())
        .toContain('bestValue / f32(uniforms.reduceSize)');
    expect(new ReduceProgram(info, 'prod', 256).getUserCode())
        .toContain('bestValue = bestValue * candidate');
    expect(new ReduceProgram(info, 'all', 256).getUserCode())
        .toContain('bestValue >= 1.0 && candidate >= 1.0');
    expect(new ReduceProgram(info, 'any', 256).getUserCode())
        .toContain('bestValue >= 1.0 || candidate >= 1.0');
  });
});

describe('shared resize gradient geometry', () => {
  it('preserves the ordinary scale and gradient search window', () => {
    expect(resizeBackpropGeometry([1, 3, 5, 1], [1, 6, 10, 1], false))
        .toEqual({effectiveXSize: [3, 5], effectiveYSize: [6, 10],
          heightScale: 0.5, widthScale: 0.5, invHeightScale: 2,
          invWidthScale: 2, winHeight: 6, winWidth: 6});
  });

  it('only subtracts endpoints when the corresponding output exceeds one', () => {
    expect(resizeBackpropGeometry([1, 3, 5, 1], [1, 5, 1, 1], true))
        .toEqual({effectiveXSize: [2, 5], effectiveYSize: [4, 1],
          heightScale: 0.5, widthScale: 5, invHeightScale: 2,
          invWidthScale: 0.2, winHeight: 6, winWidth: 4});
  });
});

describe('WebGPU NMS synchronous preparation', () => {
  it('warns once and retains empty numeric input and output arrays', () => {
    const boxes: TensorInfo = {dataId: {}, shape: [0, 4], dtype: 'float32'};
    const scores: TensorInfo = {dataId: {}, shape: [0], dtype: 'float32'};
    const emptyBoxes = new Float32Array(0);
    const emptyScores = new Float32Array(0);
    const warn = spyOn(console, 'warn');
    const reads: TensorInfo['dataId'][] = [];
    const [boxData, scoreData] = readNonMaxSuppressionInputs(boxes, scores, {
      readSync(dataId) {
        reads.push(dataId);
        return dataId === boxes.dataId ? emptyBoxes : emptyScores;
      }
    });
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn).toHaveBeenCalledWith(
        'tf.nonMaxSuppression() in webgpu locks the UI thread. ' +
        'Call tf.nonMaxSuppressionAsync() instead');
    expect(reads).toEqual([boxes.dataId, scores.dataId]);
    expect(boxData).toBe(emptyBoxes);
    expect(scoreData).toBe(emptyScores);
    const output = makeSelectedIndices({
      makeTensorInfo(shape, dtype, data) {
        expect(data).toEqual(new Int32Array(0));
        return {dataId: {}, shape, dtype};
      }
    }, []);
    expect(output.shape).toEqual([0]);
    expect(output.dtype).toBe('int32');
  });
});

describe('WebGPU pooling backprop uniform layout', () => {
  it('retains 2D dilation and reversed padding in shader order', () => {
    const info = backend_util.computePool2DInfo(
        [1, 5, 6, 2], [3, 2], [2, 3], 1, 'valid');
    expect(pool2DBackpropUniforms(info)).toEqual([
      {type: 'int32', data: [2, 3]},
      {type: 'int32', data: [2, 1]},
      {type: 'int32', data: [1, 1]},
      {type: 'int32', data: [3, 2]},
      {type: 'int32', data: [2]},
      {type: 'int32', data: [2]}
    ]);
  });

  it('retains the distinct 3D uniform order and asymmetric padding', () => {
    const info = backend_util.computePool3DInfo(
        [1, 5, 6, 7, 2], [2, 3, 4], [2, 3, 2], 1, 'same');
    expect(pool3DBackpropUniforms(info)).toEqual([
      {type: 'int32', data: [2, 3, 2]},
      {type: 'int32', data: [1, 2, 2]},
      {type: 'int32', data: [2, 3, 4]},
      {type: 'int32', data: [3]},
      {type: 'int32', data: [2]},
      {type: 'int32', data: [4]}
    ]);
  });

  it('does not share mutable arrays between average and maximum dispatches', () => {
    const info = backend_util.computePool3DInfo(
        [1, 4, 4, 4, 1], 2, 1, 1, 'valid');
    const average = pool3DBackpropUniforms(info);
    average[0].data[0] = 99;
    average.push({type: 'float32', data: [0.125]});
    const maximum = pool3DBackpropUniforms(info);
    expect(maximum.length).toBe(6);
    expect(maximum[0]).toEqual({type: 'int32', data: [1, 1, 1]});
  });
});

describe('checked WebGPU program dimensions', () => {
  it('retains softmax dispatch and the workgroup threshold', () => {
    const small = new SoftmaxProgram([0, 4095]);
    expect(small.dispatch).toEqual([0, 1, 1]);
    expect(small.workgroupSize).toEqual([64, 1, 1]);
    expect(new SoftmaxProgram([3, 4096]).workgroupSize).toEqual([256, 1, 1]);
    expect(() => new SoftmaxProgram([3])).toThrowError(/Missing dimension/);
  });
  it('retains crop geometry and singleton sampling flags', () => {
    const program = new CropAndResizeProgram(3, [2, 4], [1, 5], 'bilinear');
    expect(program.outputShape).toEqual([2, 1, 5, 3]);
    expect(program.cropHeightBiggerThan1).toBe(false);
    expect(program.cropWidthBiggerThan1).toBe(true);
    expect(program.shaderKey).toBe('cropAndResize_1_false_true');
  });
});

describe('checked WebGPU shader strides', () => {
  it('preserves symbolic order and singleton dispatch axes', () => {
    expect(symbolicallyComputeStrides([2], 'shape')).toEqual([]);
    expect(symbolicallyComputeStrides([2, 0, 1], 'shape'))
        .toEqual(['(shape.y * shape.x)', 'shape.y']);
    expect(() => symbolicallyComputeStrides([0, 6], 'shape')).toThrowError();
  });
  it('preserves coordinate names and uniform strides for rank three', () => {
    const source = getCoordsFromIndexSnippet([2, 3, 4]);
    expect(source).toContain('let d0 = index2 / uniforms.outShapeStrides.x');
    expect(source).toContain('let d2 = index2 - d1 * uniforms.outShapeStrides.y');
    expect(source).toContain('return vec3<i32>(d0,d1,d2)');
    expect(source).not.toContain('undefined');
  });
});

describe('checked WebGPU permutation and tile metadata', () => {
  it('preserves transpose order and zero-sized dimensions', () => {
    const shape = [2, 0, 3];
    const order = [2, 0, 1];
    const transpose = new TransposeProgram(shape, order);
    expect(transpose.outputShape).toEqual([3, 2, 0]);
    expect(transpose.getUserCode()).toContain('coords.y,coords.z,coords.x');
    expect(new TransposeSharedProgram([2, 3], [1, 0]).outputShape)
        .toEqual([3, 2]);
    expect(shape).toEqual([2, 0, 3]);
    expect(order).toEqual([2, 0, 1]);
    expect(() => new TransposeProgram([2, 3], [1, 2])).toThrowError();
  });
  it('retains scalar and zero-sized tiling and checks missing repeats', () => {
    expect(new TileProgram([], []).outputShape).toEqual([]);
    expect(new TileProgram([0, 3], [2, 4]).outputShape).toEqual([0, 12]);
    expect(() => new TileProgram([2, 3], [2])).toThrowError();
    expect(shaderEntry([0], 0)).toBe(0);
    expect(() => shaderEntry([], 0)).toThrowError();
  });
  it('retains a complete three-value matrix shape-fit result', () => {
    const program = new MatMulPackedProgram([1, 8, 4], [1, 8, 8]);
    expect(program.getShapeFit(0, 0, 0)).toEqual([true, true, true]);
    expect(program.getUserCode()).not.toContain('undefined');
  });
});

describe('matmul dispatch branch regression', () => {
  it('preserves independent workgroup and thread reductions', () => {
    expect(computeWorkgroupInfoForMatMul(8, 16, 16, false)).toEqual({
      workgroupSize: [4, 8, 1], elementsPerThread: [4, 1, 1]
    });
    expect(computeWorkgroupInfoForMatMul(8, 16, 16, true)).toEqual({
      workgroupSize: [8, 8, 1], elementsPerThread: [4, 4, 1]
    });
    expect(computeWorkgroupInfoForMatMul(9, 16, 16, false)).toEqual({
      workgroupSize: [4, 8, 1], elementsPerThread: [4, 4, 1]
    });
    expect(computeWorkgroupInfoForMatMul(8, 17, 16, false)).toEqual({
      workgroupSize: [8, 8, 1], elementsPerThread: [4, 1, 1]
    });
  });

  it('keeps vector and scalar split-K dispatch choices distinct', () => {
    const scalar = new MatMulSplitKProgram([1, 3, 5], 7);
    expect(scalar.outputComponent).toBe(1);
    expect(scalar.elementsPerThread).toEqual([1, 1, 128]);
    const vector = new MatMulSplitKProgram([1, 4, 8], 8);
    expect(vector.outputComponent).toBe(4);
    expect(vector.elementsPerThread).toEqual([4, 4, 128]);
  });
});

describe('optional WebGPU device features', () => {
  it('requests only supported feature strings in stable order', () => {
    expect(requiredWebGPUFeatures(new Set<GPUFeatureName>())).toEqual([]);
    expect(requiredWebGPUFeatures(new Set<GPUFeatureName>(['timestamp-query'])))
        .toEqual(['timestamp-query']);
    expect(requiredWebGPUFeatures(new Set<GPUFeatureName>(['bgra8unorm-storage'])))
        .toEqual(['bgra8unorm-storage']);
    expect(requiredWebGPUFeatures(new Set<GPUFeatureName>([
      'bgra8unorm-storage', 'timestamp-query'
    ]))).toEqual(['timestamp-query', 'bgra8unorm-storage']);
  });
});

describe('matrix multiplication program selection', () => {
  it('retains reduction, split-K, small-output and packed thresholds', () => {
    expect(selectMatMulProgramType([1, 8, 16], 2000, 1))
        .toBe(MatMulProgramType.MatMulReduceProgram);
    expect(selectMatMulProgramType([1, 8, 32], 2000, 1))
        .toBe(MatMulProgramType.MatMulSplitKProgram);
    expect(selectMatMulProgramType([1, 8, 32], 1999, 1))
        .toBe(MatMulProgramType.MatMulSmallOutputSizeProgram);
    expect(selectMatMulProgramType([2, 8, 32], 2000, 1))
        .toBe(MatMulProgramType.MatMulSmallOutputSizeProgram);
    expect(selectMatMulProgramType([1, 64, 64], 2000, 1))
        .toBe(MatMulProgramType.MatMulPackedProgram);
  });
});

describe('shader selection contracts', () => {
  it('retains aligned vector widths and scalar broadcasting', () => {
    expect(binaryVariableComponents([2, 4], [2, 4])).toEqual([4, 4]);
    expect(binaryVariableComponents([2, 4], [])).toEqual([4, 1]);
    expect(binaryVariableComponents([], [2, 4])).toEqual([1, 4]);
    expect(binaryVariableComponents([2, 3], [2, 3])).toEqual([1, 1]);
    expect(binaryVariableComponents([2, 4], [2, 1])).toEqual([4, 1]);
    expect(binaryVariableComponents([0], [0])).toEqual([4, 4]);
  });
  it('preserves case, comma and empty-filter matching semantics', () => {
    expect(shouldPrintShader('', 'MatMul')).toBe(false);
    expect(shouldPrintShader('ALL', 'MatMul')).toBe(true);
    expect(shouldPrintShader('MATMUL,conv', 'MatMulPacked')).toBe(true);
    expect(shouldPrintShader('conv', 'MatMulPacked')).toBe(false);
    expect(shouldPrintShader('conv,', 'MatMulPacked')).toBe(true);
    expect(shouldPrintShader(' matmul', 'MatMulPacked')).toBe(false);
  });
  it('keeps fused convolution input names and component widths aligned', () => {
    const info = backend_util.computeConv2DInfo(
        [1, 4, 4, 4], [1, 1, 4, 4], 1, 1, 'same');
    const packed = new Conv2DMMProgram(info, 16, 4, 4, true, 'prelu', true);
    expect(packed.variableNames).toEqual(['x', 'W', 'bias', 'preluActivationWeights']);
    expect(packed.variableComponents).toEqual([4, 4, 4, 4]);
    const plain = new Conv2DMMProgram(info, 16, 4, 4);
    expect(plain.variableNames).toEqual(['x', 'W']);
    expect(plain.variableComponents).toEqual([4, 4]);
  });
});
