/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

import * as axis from './axis_util';
import {combineRaggedTensorToTensorShapes, getRaggedRank, RowPartitionType, validateDefaultValueShape} from './ragged_to_dense_util';
import {pixelBytes, pixelGeometry} from './pixel_util';
import * as complexUtil from '../backends/complex_util';
import {nonMaxSuppressionV3Impl, nonMaxSuppressionV5Impl} from '../backends/non_max_suppression_impl';
import {fingerPrint64, hexToLong} from '../hash_util';
import * as broadcast from './broadcast_util';
import {decodeEinsumEquation, getEinsumComputePath, getEinsumPermutation} from '../backends/einsum_util';
import {TensorInfo} from '../tensor_info';
import * as arrayOps from './array_ops_util';
import {calculateShapes} from './scatter_nd_util';
import {collectGatherOpShapeInfo, computeOutShape} from './segment_util';
import * as concat from './concat_util';
import * as slice from './slice_util';
import {depthToSpaceOutputShape, imageDimensions, matrixDimensions, permutedShape, unpackShape, requiredKernelEntry, unbatchedShape, batchedImageShape} from './shape_dimension';

describe('checked shape utility regressions', () => {
  it('interleaves coordinates including zero', () => {
    expect(axis.combineLocations([0, 2], [3, 0], [1, 3]))
        .toEqual([0, 3, 2, 0]);
  });
  it('preserves reduction axis order and zero dimensions', () => {
    expect(axis.computeOutAndReduceShapes([0, 2, 3], [2, 0]))
        .toEqual([[2], [3, 0]]);
    expect(() => axis.computeOutAndReduceShapes([2], [1])).toThrowError();
  });
  it('inverts a permutation without mutating it', () => {
    const permutation = [2, 0, 1, 3];
    expect(axis.getUndoAxesPermutation(permutation)).toEqual([1, 2, 0, 3]);
    expect(permutation).toEqual([2, 0, 1, 3]);
  });
  it('concatenates zero-length dimensions without mutating shapes', () => {
    const shapes = [[0, 3], [2, 3], [4, 3]];
    expect(() => concat.assertParamsConsistent(shapes, 0)).not.toThrow();
    expect(concat.computeOutShape(shapes, 0)).toEqual([6, 3]);
    expect(shapes).toEqual([[0, 3], [2, 3], [4, 3]]);
  });
  it('rejects missing concatenation shapes or dimensions', () => {
    expect(() => concat.computeOutShape([], 0)).toThrowError();
    expect(() => concat.computeOutShape([[2], []], 0)).toThrowError();
    expect(() => concat.assertParamsConsistent([[2, 3], [2, 4]], 0))
        .toThrowError();
  });
  it('preserves the historical broadcast dimension fallback', () => {
    expect(broadcast.getBroadcastDims([0, 1], [3, 4])).toEqual([0, 1]);
    expect(broadcast.getBroadcastDims([1], [0])).toEqual([]);
    expect(broadcast.getBroadcastDims([1, 3], [3])).toEqual([]);
  });
  it('calculates strided shapes and flat offsets', () => {
    expect(slice.computeOutShape([0, 1], [5, 7], [2, 3])).toEqual([3, 2]);
    expect(slice.computeOutShape([4], [0], [-2])).toEqual([2]);
    expect(slice.computeFlatOffset([1, 2, 3], [12, 4])).toBe(23);
    expect(slice.computeFlatOffset([], [])).toBe(1);
  });
  it('preserves contiguous slice detection', () => {
    expect(slice.isSliceContinous([5, 3], [1, 0], [2, 3])).toBe(true);
    expect(slice.isSliceContinous([5, 3], [0, 1], [5, 2])).toBe(false);
    expect(slice.isSliceContinous([], [], [])).toBe(true);
  });
  it('preserves masked reverse slicing and ellipsis expansion', () => {
    const reversed = slice.sliceInfo([4], [0], [0], [-1], 1, 1, 0, 0, 0);
    expect(reversed.finalShape).toEqual([4]);
    expect(reversed.begin).toEqual([3]);
    expect(reversed.end).toEqual([-1]);
    expect(reversed.strides).toEqual([-1]);
    const ellipsis = slice.sliceInfo([2, 3], [0], [0], [1], 0, 0, 1, 0, 0);
    expect(ellipsis.finalShape).toEqual([2, 3]);
    expect(ellipsis.isIdentity).toBe(true);
  });
  it('preserves shrink axes and rejects zero strides', () => {
    const shrink = slice.sliceInfo([2, 3], [1, 0], [0, 3], [1, 1],
        0, 0, 0, 0, 1);
    expect(shrink.finalShape).toEqual([3]);
    expect(shrink.begin).toEqual([1, 0]);
    expect(shrink.end).toEqual([2, 3]);
    expect(() => slice.sliceInfo([4], [0], [4], [0], 0, 0, 0, 0, 0))
        .toThrowError('strides[0] must be non-zero');
  });
  it('rejects missing required slice coordinates', () => {
    expect(() => slice.computeOutShape([0], [], [1])).toThrowError();
    expect(() => slice.computeFlatOffset([1, 2], [])).toThrowError();
  });
});

describe('checked matrix and image dimensions', () => {
  it('reads trailing dimensions, including zero and rectangular matrices', () => {
    expect(matrixDimensions([3, 2, 0])).toEqual([2, 0]);
    expect(matrixDimensions([2, 5])).toEqual([2, 5]);
    expect(() => matrixDimensions([2])).toThrowError();
  });
  it('preserves image dimensions and rejects missing channels', () => {
    expect(imageDimensions([1, 0, 3, 4])).toEqual([1, 0, 3, 4]);
    expect(() => imageDimensions([1, 2, 3])).toThrowError();
  });
  it('computes depth to space for each layout without changing input', () => {
    const nhwc = [2, 3, 5, 12];
    expect(depthToSpaceOutputShape(nhwc, 2, 'NHWC')).toEqual([2, 6, 10, 3]);
    expect(depthToSpaceOutputShape([2, 12, 3, 5], 2, 'NCHW'))
        .toEqual([2, 3, 6, 10]);
    expect(depthToSpaceOutputShape([0, 3, 5, 12], 2, 'NHWC'))
        .toEqual([0, 6, 10, 3]);
    expect(nhwc).toEqual([2, 3, 5, 12]);
  });
});

describe('checked space, gather and scatter dimensions', () => {
  it('preserves reshape order and zero-sized dimensions', () => {
    expect(arrayOps.getReshaped([4, 0, 3], [2], 2)).toEqual([2, 2, 0, 3]);
    expect(arrayOps.getReshaped([2, 4, 3], [2], 2, false))
        .toEqual([2, 2, 2, 3]);
    expect(arrayOps.getReshapedPermuted([4, 0, 3], [2], 2))
        .toEqual([2, 0, 3]);
    expect(arrayOps.getSliceBeginCoords([[0, 2]], 1)).toEqual([0, 0]);
    expect(arrayOps.getSliceSize([2, 5], [[1, 2]], 1)).toEqual([2, 2]);
    expect(() => arrayOps.getSliceSize([2, 5], [[1]], 1)).toThrowError();
  });
  it('retains zero dimensions in gather and scatter shape metadata', () => {
    const x: TensorInfo = {dataId: {}, shape: [2, 3, 0], dtype: 'float32'};
    const indices: TensorInfo = {dataId: {}, shape: [2, 1], dtype: 'int32'};
    expect(collectGatherOpShapeInfo(x, indices, 1, 1)).toEqual({
      batchSize: 2, sliceSize: 0, outerSize: 1, dimSize: 3,
      outputShape: [2, 1, 0]
    });
    expect(computeOutShape([2, 0, 3], 0, 5)).toEqual([5, 0, 3]);
    expect(calculateShapes(x, indices, [3, 0])).toEqual({
      sliceRank: 1, numUpdates: 2, sliceSize: 0, strides: [1], outputSize: 0
    });
    expect(() => collectGatherOpShapeInfo(x, indices, 3, 0)).toThrowError();
  });
});

describe('checked einsum metadata', () => {
  it('preserves label order, contraction and scalar equations', () => {
    const decoded = decodeEinsumEquation('ij,jk -> ik', 2);
    expect(decoded).toEqual({
      allDims: ['i', 'k', 'j'], summedDims: [2], idDims: [[0, 2], [2, 1]]
    });
    expect(getEinsumPermutation(3, [2, 1])).toEqual({
      permutationIndices: [1, 0], expandDims: [0]
    });
    expect(getEinsumComputePath(decoded.summedDims, decoded.idDims))
        .toEqual({path: [2], steps: [[0, 1], []]});
    expect(decodeEinsumEquation('->', 1)).toEqual({
      allDims: [], summedDims: [], idDims: [[]]
    });
    expect(() => decodeEinsumEquation('ii->i', 1)).toThrowError();
  });
});

describe('fixed-width fingerprint state', () => {
  it('preserves existing reference hashes across multiple 64-byte blocks', () => {
    const bytes = new Uint8Array(132);
    let seed = 10;
    for (let i = 0; i < bytes.length; i++) {
      seed = (seed * 16807) % 2147483647;
      bytes[i] = (seed >> 1) & 255;
    }
    expect(fingerPrint64(bytes, 0)).toEqual(hexToLong('9ae16a3b2f90404f'));
    expect(fingerPrint64(bytes, 67)).toEqual(hexToLong('0ced753b45740875'));
    expect(fingerPrint64(bytes, 132)).toEqual(hexToLong('f2e4c5b6db6c154a'));
  });
});


describe('checked numeric kernel metadata', () => {
  it('interleaves and separates complex components including zero and NaN', () => {
    const real = new Float32Array([0, -2, NaN]);
    const imag = new Float32Array([3, 0, 4]);
    const combined = complexUtil.mergeRealAndImagArrays(real, imag);
    const split = complexUtil.splitRealAndImagArrays(combined);
    expect(split.real).toEqual(real);
    expect(split.imag).toEqual(imag);
    expect(complexUtil.getComplexWithIndex(combined, 1)).toEqual({real: -2, imag: 0});
    expect(complexUtil.complexWithEvenIndex(combined).imag).toEqual(new Float32Array([3, 4]));
    expect(complexUtil.complexWithOddIndex(combined).real).toEqual(new Float32Array([-2]));
    expect(() => complexUtil.getComplexWithIndex(combined, 3)).toThrowError(/Missing numeric entry/);
  });
  it('preserves legacy incomplete-pair extraction semantics', () => {
    const incomplete = new Float32Array([1, 2, 3]);
    expect(complexUtil.splitRealAndImagArrays(incomplete)).toEqual({real: new Float32Array([1]), imag: new Float32Array([2])});
    expect(complexUtil.complexWithOddIndex(incomplete).real).toEqual(new Float32Array());
    expect(complexUtil.complexWithEvenIndex(new Float32Array([1])).imag).toEqual(new Float32Array([NaN]));
  });
  it('keeps tied non-max suppression candidates in original index order', () => {
    const boxes = new Float32Array([0, 0, 1, 1, 0, 0, 1, 1, 3, 3, 4, 4]);
    const scores = new Float32Array([1, 1, 0.5]);
    expect(nonMaxSuppressionV3Impl(boxes, scores, 3, 0.5, 0).selectedIndices).toEqual([0, 2]);
    expect(nonMaxSuppressionV5Impl(boxes, scores, 3, 0.5, 0, 0).selectedIndices).toEqual([0, 2]);
  });
});

describe('shared backend shape transformations', () => {
  it('permutes zero and scalar shapes without mutating the input', () => {
    const shape = [2, 0, 3];
    expect(permutedShape(shape, [2, 0, 1])).toEqual([3, 2, 0]);
    expect(permutedShape([], [])).toEqual([]);
    expect(shape).toEqual([2, 0, 3]);
    expect(() => permutedShape(shape, [0, 1])).toThrowError();
    expect(() => permutedShape(shape, [0, 1, 3])).toThrowError();
  });
  it('retains a zero slice count and rejects an absent unpack axis', () => {
    expect(unpackShape([2, 0, 3], 1)).toEqual({num: 0, outShape: [2, 3]});
    expect(unpackShape([2], 0)).toEqual({num: 2, outShape: []});
    expect(() => unpackShape([2], 1)).toThrowError();
    expect(requiredKernelEntry([0, NaN], 0)).toBe(0);
    expect(Number.isNaN(requiredKernelEntry([0, NaN], 1))).toBe(true);
    expect(() => requiredKernelEntry([], 0)).toThrowError();
  });
});

describe('shared tensor pixel conversion', () => {
  it('preserves grayscale, RGB, RGBA, default alpha and byte rounding', () => {
    expect(pixelGeometry([1, 2])).toEqual([1, 2, 1]);
    expect(pixelGeometry([2, 3, 4])).toEqual([2, 3, 4]);
    expect(pixelBytes(new Float32Array([0, 0.5]), 1, 2, 1, 'float32', 255))
        .toEqual(new Uint8ClampedArray([0, 0, 0, 255, 128, 128, 128, 255]));
    expect(pixelBytes(new Int32Array([0, 12, 255]), 1, 1, 3, 'int32', 127.5))
        .toEqual(new Uint8ClampedArray([0, 12, 255, 128]));
    expect(pixelBytes(new Int32Array([0, 12, 255, 0]), 1, 1, 4, 'int32', 255))
        .toEqual(new Uint8ClampedArray([0, 12, 255, 0]));
  });
  it('preserves empty images and NaN coercion and rejects invalid storage', () => {
    expect(pixelGeometry([0, 2, 3])).toEqual([0, 2, 3]);
    expect(pixelBytes(new Float32Array(0), 0, 2, 3, 'float32', 255))
        .toEqual(new Uint8ClampedArray(0));
    expect(pixelBytes(new Float32Array([NaN]), 1, 1, 1, 'float32', 255))
        .toEqual(new Uint8ClampedArray([0, 0, 0, 255]));
    expect(() => pixelBytes(new Int32Array([256]), 1, 1, 1, 'int32', 255))
        .toThrowError(/range \[0 - 255\]/);
    expect(() => pixelBytes(new Float32Array([-0.1]), 1, 1, 1, 'float32', 255))
        .toThrowError(/range \[0 - 1\]/);
    expect(() => pixelBytes(new Float32Array(0), 1, 1, 1, 'float32', 255))
        .toThrowError(/Missing kernel entry/);
  });
});

describe('inserted batch-axis kernel result geometry', () => {
  it('retains zero dimensions and does not mutate either spatial rank', () => {
    const image = [1, 0, 3, 4];
    const volume = [1, 2, 0, 4, 5];
    const imageShape: [number, number, number] = unbatchedShape(image, 4);
    const volumeShape: [number, number, number, number] =
        unbatchedShape(volume, 5);
    expect(imageShape).toEqual([0, 3, 4]);
    expect(volumeShape).toEqual([2, 0, 4, 5]);
    expect(image).toEqual([1, 0, 3, 4]);
    expect(volume).toEqual([1, 2, 0, 4, 5]);
  });
  it('rejects an unexpected kernel result rank', () => {
    expect(() => unbatchedShape([1, 2, 3], 4)).toThrowError();
    expect(() => unbatchedShape([1, 2, 3, 4, 5], 4)).toThrowError();
    expect(() => unbatchedShape([1, 2, 3, 4], 5)).toThrowError();
  });
  it('rejects missing required spatial entries at either rank', () => {
    const image = [1, 2, 3, 4];
    delete image[2];
    const volume = [1, 2, 3, 4, 5];
    delete volume[4];
    expect(() => unbatchedShape(image, 4)).toThrowError(/Missing dimension at index 2/);
    expect(() => unbatchedShape(volume, 5)).toThrowError(/Missing dimension at index 4/);
  });
});


describe('checked ragged metadata', () => {
  it('retains zero dimensions, unknown dimensions and partition rank', () => {
    expect(combineRaggedTensorToTensorShapes(1, [-1, -1, -1], [4, 0]))
        .toEqual([-1, -1, 0]);
    expect(getRaggedRank([])).toBe(0);
    expect(getRaggedRank([RowPartitionType.FIRST_DIM_SIZE, RowPartitionType.ROW_SPLITS]))
        .toBe(1);
    expect(() => validateDefaultValueShape([1], [4, 0])).not.toThrow();
    expect(() => validateDefaultValueShape([2], [4, 0])).toThrowError();
  });
  it('rejects a missing required ragged dimension', () => {
    const missing = [4, 2];
    delete missing[1];
    expect(() => combineRaggedTensorToTensorShapes(1, [-1, -1, -1], missing))
        .toThrowError(/Missing dimension/);
  });
});

describe('checked image batching dimensions', () => {
  it('inserts a batch axis without mutating the original spatial shape', () => {
    const shape = [0, 3, 4];
    expect(batchedImageShape(shape, 4)).toEqual([1, 0, 3, 4]);
    expect(shape).toEqual([0, 3, 4]);
    expect(batchedImageShape([2, 0, 3, 4], 5)).toEqual([1, 2, 0, 3, 4]);
  });
  it('rejects invalid rank and missing spatial coordinates', () => {
    expect(() => batchedImageShape([2, 3], 4)).toThrowError();
    expect(() => batchedImageShape(new Array<number>(4), 5)).toThrowError();
  });
});

describe('checked elided slice bounds', () => {
  it('inserts full bounds and clamps negative explicit ends', () => {
    expect(slice.startIndicesWithElidedDims(0, 0, 1, [0, 1], [2, 3]))
        .toEqual([0, 1]);
    expect(slice.stopIndicesWithElidedDims(0, 0, 1, [0, -1], [2, 3]))
        .toEqual([2, 2]);
  });
  it('does not require masked bounds but rejects missing explicit bounds', () => {
    expect(slice.startIndicesWithElidedDims(2, 0, 1, [], [2, 3]))
        .toEqual([0, 0]);
    expect(slice.stopIndicesWithElidedDims(2, 0, 1, [], [2, 3]))
        .toEqual([2, 3]);
    expect(() => slice.startIndicesWithElidedDims(0, 0, 1, [], [2, 3]))
        .toThrowError(/Missing dimension/);
    expect(() => slice.stopIndicesWithElidedDims(0, 0, 1, [], [2, 3]))
        .toThrowError(/Missing dimension/);
  });
});

describe('dense axis bound contracts', () => {
  it('retains masked reverse bounds and negative shrink coordinates', () => {
    const reversed = slice.sliceInfo([4], [0], [0], [-1], 1, 1, 0, 0, 0);
    expect(reversed.begin).toEqual([3]);
    expect(reversed.end).toEqual([-1]);
    expect(reversed.finalShape).toEqual([4]);
    const shrunk = slice.sliceInfo([4], [-1], [0], [1], 0, 0, 0, 0, 1);
    expect(shrunk.begin).toEqual([3]);
    expect(shrunk.end).toEqual([4]);
    expect(shrunk.finalShape).toEqual([]);
  });
});

import {spatialDimensionDivides} from './space_to_batch_nd';

describe('spatial block divisibility', () => {
  it('checks the selected axis and permits zero padding and dimensions', () => {
    expect(spatialDimensionDivides(3, [[0, 1]], [2], 0)).toBe(true);
    expect(spatialDimensionDivides(3, [[0, 0]], [2], 0)).toBe(false);
    expect(spatialDimensionDivides(0, [[0, 0]], [2], 0)).toBe(true);
    expect(() => spatialDimensionDivides(3, [[]], [2], 0)).toThrow();
    expect(() => spatialDimensionDivides(3, [[0, 1]], [], 0)).toThrow();
  });
});
