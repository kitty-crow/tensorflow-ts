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

import { TensorInfo } from '../tensor_info';
import * as util from '../util';

import {requiredDimension} from './shape_dimension';

const NEW_AXIS = -2;
const SHRINK_AXIS = -1;

// Sparse slicing specification
// if one does foo[3:5, ..., -3], the begin, end and strides will have length
// of 3.
interface StridedSliceSparseSpec {
  dims: number;
  numAddAxisAfterEllipsis: number;
  begin: number[];
  end: number[];
  strides: number[];
  beginMask: number;
  endMask: number;
  ellipsisMask: number;
  newAxisMask: number;
  shrinkAxisMask: number;
}

// Dense slicing specification
// all ellipses and newaxis are expanded out. So if foo[3:5, ..., -3] where foo
// is 10 dimensional, each array of begin, end, strides will have 10 entries
// where as the sparse can have length less than the rank of foo.
interface StridedSliceDenseSpec {
  dims: number;
  beginMask: number;
  endMask: number;
  beginValid: boolean;
  endValid: boolean;
  begin: number[];
  end: number[];
  strides: number[];
  // This array helps construct the final shape of the slice.
  // The final tensor is reduced in rank whenever a single index e.g. foo[3]
  // is called for. The final tensor increases in rank with newAxis entries.
  // If an index in this array is positive, the size of the dimension is
  // obtained from canonical end-begin.  Otherwise, if it is a NEW_AXIS, it will
  // be 1. A shrunk dimension is skipped.
  finalShapeGatherIndices: number[];
  // This array has the same size as finalShapeGatherIndices, but it remembers
  // the sparse index that a dimension comes from, instead of dense index.
  // A -1 in this vector means the index is not from the sparse input.
  finalShapeGatherIndicesSparse: number[];
  inputShapeGatherIndicesSparse: number[];
  // The dense indexed shrink mask is which processing dimensions should be
  // shrunk. For example, if foo.shape = [10, 10, 10, 10], foo[3, ..., 5] has
  // sparseShrinkAxisMask of 5 (0101) and denseShrinkAxisMask of 9 (1001),
  // yielding a final shape [10, 10].
  shrinkAxisMask: number;
}

export type SliceInfo = {
  finalShapeSparse: number[],
  finalShape: number[],
  isIdentity: boolean,
  sliceDim0: boolean,
  isSimpleSlice: boolean,
  begin: number[],
  end: number[],
  strides: number[]
};

export function assertParamsValid(
    input: TensorInfo, begin: number[], size: number[]): void {
  const inputRank = input.shape.length;
  util.assert(
      inputRank === begin.length,
      () => `Error in slice${inputRank}D: Length of begin ${begin} must ` +
          `match the rank of the array (${inputRank}).`);
  util.assert(
      inputRank === size.length,
      () => `Error in slice${inputRank}D: Length of size ${size} must ` +
          `match the rank of the array (${inputRank}).`);

  for (let i = 0; i < inputRank; ++i) {
    const extent = requiredDimension(begin, i) + requiredDimension(size, i);
    const dimension = requiredDimension(input.shape, i);
    util.assert(
        extent <= dimension,
        () => `Error in slice${inputRank}D: begin[${i}] + size[${i}] ` +
            `(${extent}) would overflow input.shape[${i}] (${dimension})`);
  }
}

/** Converts a binary mask to an array of axes. Used in stridedSlice(). */
export function maskToAxes(mask: number): number[] {
  const axes: number[] = [];
  let axis = 0;
  while (mask > 0) {
    if (mask & 1) {
      axes.push(axis);
    }
    mask /= 2;
    axis++;
  }
  return axes;
}

/** Computes the output shape given the strided slice params. */
export function computeOutShape(
    begin: number[], end: number[], strides: number[]): number[] {
  const size: number[] = [];
  for (let axis = 0; axis < begin.length; axis++) {
    size[axis] = Math.ceil(
        (requiredDimension(end, axis) - requiredDimension(begin, axis)) /
        requiredDimension(strides, axis));
  }
  return size;
}

// Creates full selection at the elided dimensions. If the dimension matches
// the ellipsis mask, override the current stride value. Otherwise, insert.
export function stridesWithElidedDims(
    strides: number[], ellipsisInsertionIndex: number, numElidedAxes: number,
    inputShape: number[]): number[] {
  const newStrides = [...strides];
  for (let i = newStrides.length; i < inputShape.length; i++) {
    newStrides.push(1);
  }
  for (let i = 0; i < numElidedAxes; i++) {
    if (i === 0) {
      newStrides[ellipsisInsertionIndex] = 1;
    } else {
      newStrides.splice(
          ellipsisInsertionIndex, 0 /* num elements to delete */,
          1 /* element to add */);
      newStrides.pop();
    }
  }
  return newStrides;
}

function unnormalizeAxis(
    ellipsisInsertionIndex: number, numElidedAxes: number,
    normalizedAxis: number): number {
  if (normalizedAxis <= ellipsisInsertionIndex) {
    return normalizedAxis;
  }

  return normalizedAxis - (numElidedAxes - 1);
}

function getElidedAxes(numElidedAxes: number, ellipsisInsertionIndex: number) {
  const elidedAxes: number[] = [];
  for (let i = 0; i < numElidedAxes; i++) {
    elidedAxes.push(ellipsisInsertionIndex + i);
  }
  return elidedAxes;
}

// Normalize the start, end and strides.
export function getNormalizedAxes(
    inputShape: number[], ellipsisAxes: number[], numInterpolatedAxes: number,
    begin: number[], end: number[], strides: number[], beginMask: number,
    endMask: number,
    ellipsisMask: number): {begin: number[], end: number[], strides: number[]} {
  const inputRank = inputShape.length;
  let normalizedBegin = new Array<number>(inputRank),
      normalizedEnd = new Array<number>(inputRank),
      normalizedStrides = new Array<number>(inputRank);
  if (ellipsisAxes.length && numInterpolatedAxes > 0) {
    const fullIndex = requiredDimension(ellipsisAxes, 0);

    // The ellipsis applies to the masked index as well as any dimensions
    // that are interpolated.
    const numElidedAxes = numInterpolatedAxes + 1;
    normalizedBegin = startIndicesWithElidedDims(
        beginMask, fullIndex, numElidedAxes, begin, inputShape);
    normalizedEnd = stopIndicesWithElidedDims(
        endMask, fullIndex, numElidedAxes, end, inputShape);
    normalizedStrides =
        stridesWithElidedDims(strides, fullIndex, numElidedAxes, inputShape);
  } else {
    for (let axis = 0; axis < inputRank; axis++) {
      normalizedBegin[axis] = startForAxis(
          beginMask, begin, strides, inputShape, axis, ellipsisMask);
      normalizedEnd[axis] =
          stopForAxis(endMask, end, strides, inputShape, axis, ellipsisMask);
      normalizedStrides[axis] = stridesForAxis(strides, axis, ellipsisMask);
    }
  }

  return {
    begin: normalizedBegin,
    end: normalizedEnd,
    strides: normalizedStrides
  };
}

// Creates full selection at the elided dimensions. If the dimension matches
// the ellipsis mask, override the current start value. Otherwise, insert.
export function startIndicesWithElidedDims(
    beginMask: number, ellipsisInsertionIndex: number, numElidedAxes: number,
    originalBegin: number[], inputShape: number[]): number[] {
  const newIndices = [...inputShape];
  const elidedAxes = getElidedAxes(numElidedAxes, ellipsisInsertionIndex);

  for (let axis = 0; axis < newIndices.length; axis++) {
    if (elidedAxes.indexOf(axis) > -1) {
      newIndices[axis] = 0;
      continue;
    }
    const originalAxis =
        unnormalizeAxis(ellipsisInsertionIndex, numElidedAxes, axis);
    newIndices[axis] = beginMask & 1 << originalAxis ?
        0 : originalBegin[originalAxis];
  }
  return newIndices;
}

// Creates full selection at the elided dimensions. If the dimension matches
// the ellipsis mask, override the current stop value. Otherwise, insert.
export function stopIndicesWithElidedDims(
    endMask: number, ellipsisInsertionIndex: number, numElidedAxes: number,
    originalEnd: number[], inputShape: number[]): number[] {
  const newIndices = [...inputShape];
  const elidedAxes = getElidedAxes(numElidedAxes, ellipsisInsertionIndex);

  for (let axis = 0; axis < newIndices.length; axis++) {
    if (elidedAxes.indexOf(axis) > -1) {
      newIndices[axis] = Number.MAX_SAFE_INTEGER;
      continue;
    }
    const originalAxis =
        unnormalizeAxis(ellipsisInsertionIndex, numElidedAxes, axis);
    newIndices[axis] = endMask & 1 << originalAxis ?
        Number.MAX_SAFE_INTEGER : originalEnd[originalAxis];
  }

  for (let i = 0; i < newIndices.length; i++) {
    // Handle negative indices
    const axisSize = inputShape[i];
    if (newIndices[i] < 0) {
      newIndices[i] += axisSize;
    }
    newIndices[i] = util.clamp(0, newIndices[i], inputShape[i]);
  }
  return newIndices;
}

export function stridesForAxis(
    strides: number[], axis: number, ellipsisMask: number): number {
  let stride = strides[axis];
  if (ellipsisMask & (1 << axis) || stride == null) {
    stride = 1;
  }

  return stride;
}

export function startForAxis(
    beginMask: number, startIndices: number[], strides: number[],
    inputShape: number[], axis: number, ellipsisMask: number): number {
  // Begin with the specified index
  let start = startIndices[axis];
  const stride = strides[axis] || 1;

  // Check the axis bit from right of masked axes, or the begin index is not set
  // for the axis.
  if (beginMask & 1 << axis || ellipsisMask & 1 << axis || start == null) {
    // Forward iteration starts at the first element; backward at the last.
    // Extreme values are clamped below, matching StopForAxis().
    start = stride > 0 ? Number.MIN_SAFE_INTEGER : Number.MAX_SAFE_INTEGER;
  }

  // Handle negative indices
  const axisSize = requiredDimension(inputShape, axis);
  if (start < 0) {
    start += axisSize;
  }

  // Clamping
  start = util.clamp(0, start, axisSize - 1);

  return start;
}

export function stopForAxis(
    endMask: number, stopIndices: number[], strides: number[],
    inputShape: number[], axis: number, ellipsisMask: number): number {
  // Begin with the specified index
  let stop = stopIndices[axis];
  const stride = strides[axis] || 1;

  // Check the axis bit from right of masked axes, or if the stop index is not
  // set for this axis.
  if (endMask & (1 << axis) || ellipsisMask & (1 << axis) || stop == null) {
    // Use extreme bounds before clamping to the axis dimensions.
    stop = stride > 0 ? Number.MAX_SAFE_INTEGER : Number.MIN_SAFE_INTEGER;
  }

  // Handle negative indices
  const axisSize = requiredDimension(inputShape, axis);
  if (stop < 0) {
    stop += axisSize;
  }

  // Clamping
  // Because the end index points one past the last element, we need slightly
  // different clamping ranges depending on the direction.
  if (stride > 0) {
    // Forward iteration
    stop = util.clamp(0, stop, axisSize);
  } else {
    // Backward iteration
    stop = util.clamp(-1, stop, axisSize - 1);
  }

  return stop;
}

/**
 * Returns true if the slice occupies a continous set of elements in the
 * 'flat' space.
 */
export function isSliceContinous(
    shape: number[], begin: number[], size: number[]) {
  // Index of the first axis that has size > 1.
  let firstNonOneAxis = size.length;
  for (let i = 0; i < size.length; i++) {
    if (requiredDimension(size, i) > 1) {
      firstNonOneAxis = i;
      break;
    }
  }

  for (let i = firstNonOneAxis + 1; i < size.length; i++) {
    if (requiredDimension(begin, i) > 0 ||
        requiredDimension(size, i) !== requiredDimension(shape, i)) {
      return false;
    }
  }
  return true;
}

export function computeFlatOffset(begin: number[], strides: number[]): number {
  let flatOffset = begin.length > 0 ?
      requiredDimension(begin, begin.length - 1) : 1;
  for (let i = 0; i < begin.length - 1; i++) {
    flatOffset += requiredDimension(begin, i) * requiredDimension(strides, i);
  }
  return flatOffset;
}

export function parseSliceParams(
    x: TensorInfo, begin: number|number[], size?: number|number[]):
    [number[], number[]] {
  // The following logic allows for more ergonomic calls.
  const xRank = x.shape.length;
  const begin_: number[] = typeof begin === 'number' ?
      [begin, ...new Array<number>(xRank - 1).fill(0)] :
      begin.length < xRank ?
      begin.concat(new Array<number>(xRank - begin.length).fill(0)) :
      begin.slice();
  begin_.forEach(d => {
    util.assert(
        d !== -1, () => 'slice() does not support negative begin indexing.');
  });
  const initialSize: number[] = size == null ?
      new Array<number>(xRank).fill(-1) :
      typeof size === 'number' ?
      [size, ...new Array<number>(xRank - 1).fill(-1)] :
      size.length < xRank ?
      size.concat(new Array<number>(xRank - size.length).fill(-1)) : size;
  const size_ = initialSize.map((d, i) => {
    if (d >= 0) {
      return d;
    } else {
      util.assert(
          d === -1,
          () => `Negative size values should be exactly -1 but got ` +
              `${d} for the slice() size at index ${i}.`);
      return requiredDimension(x.shape, i) - requiredDimension(begin_, i);
    }
  });
  return [begin_, size_];
}

// Convert the slicing specification from a sparse representation to a dense
// representation. This means that all ellipses and newaxis are expanded out.
export function sliceInfo(
    xShape: number[], begin: number[], end: number[], strides: number[],
    beginMask: number, endMask: number, ellipsisMask: number,
    newAxisMask: number, shrinkAxisMask: number): SliceInfo {
  let stridesNonNull: number[];
  if (strides == null) {
    stridesNonNull = new Array<number>(begin.length);
    stridesNonNull.fill(1);
  } else {
    stridesNonNull = strides;
  }

  // Only one non-zero bit is allowed in ellipsisMask, which means ellipsisMask
  // is a power of 2. Use bit compares to ensure ellipsisMask is 0 or a power
  // of 2. When i is a power of 2, i & (i - 1) is always 0.
  // Also ref:
  // https://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
  if (ellipsisMask != null && (ellipsisMask & (ellipsisMask - 1)) !== 0) {
    throw new Error('Multiple ellipses in slice is not allowed.');
  }

  // Step 1: Account for ellipsis and new axis.
  // Check for ellipsis and count how many non-newaxis there are after.
  let ellipsisSeen = false;

  const sparseSpec: StridedSliceSparseSpec = {
    dims: stridesNonNull.length,
    numAddAxisAfterEllipsis: 0,
    begin: begin.slice(),
    end: end.slice(),
    strides: stridesNonNull.slice(),
    beginMask,
    endMask,
    ellipsisMask,
    newAxisMask,
    shrinkAxisMask
  };

  for (let i = 0; i < sparseSpec.dims; i++) {
    if (ellipsisSeen && ((1 << i) & newAxisMask) !== 0) {
      sparseSpec.numAddAxisAfterEllipsis++;
    }
    if ((1 << i) & ellipsisMask) {
      ellipsisSeen = true;
    }
  }
  // If no ellipsis insert one at the end.
  if (!ellipsisSeen) {
    sparseSpec.ellipsisMask |= (1 << sparseSpec.dims);
    sparseSpec.dims++;  // this effects loop iteration below
  }

  // Step 2: Make a sparse spec into a full index spec.
  //
  // The sparse spec deos not correspond to the number of dimensions.
  // Make a dense spec that cooresponds to the number of dimensions.
  //
  // For example suppose foo[...,3:] on foo.shape = [2, 2, 3] then we need to
  // produce the missing beginMask for the first two dimensions i.e. from
  // beginMaskSpec = 0, endMaskSpec = 2, we achieve beginMask = 6 (110),
  // endMask = 7 (111).
  const denseSpec = buildDenseSpec(sparseSpec, xShape.length);

  // Step 3: Make implicit ranges (non-zero beginMasks and endMasks) explicit
  // and bounds check.
  let isIdentity = true;
  let sliceDim0 = true;
  let isSimpleSlice = true;
  const processingShape: number[] = [];
  const finalShape: number[] = [];

  for (let i = 0; i < xShape.length; ++i) {
    const stride = requiredDimension(denseSpec.strides, i);
    if (stride === 0) {
      throw Error(`strides[${i}] must be non-zero`);
    }
    const shrinkI = !!(denseSpec.shrinkAxisMask & (1 << i));
    const dimI = requiredDimension(xShape, i);
    if (dimI === -1) {
      processingShape.push(shrinkI ? 1 : -1);
      continue;
    }

    const masks =
        [denseSpec.beginMask & (1 << i), denseSpec.endMask & (1 << i)];
    const validRange = [
      stride > 0 ? 0 : -1,
      stride > 0 ? dimI : dimI - 1
    ];

    if (shrinkI && stride <= 0) {
      throw Error('only stride 1 allowed on non-range indexing.');
    }

    isSimpleSlice = isSimpleSlice && (stride === 1);

    const beginAndEndMasked =
        !!((denseSpec.beginMask & (1 << i)) && (denseSpec.endMask & (1 << i)));

    if (denseSpec.beginValid && denseSpec.endValid) {
      if (shrinkI) {
        // If we are shrinking, the end index is now possibly incorrect. In
        // particular foo[-1] produces sparseBegin = -1, sparseEnd = 0.
        // and canonical puts these to n-1 and 0, which implies a degenerate
        // interval. Fortunately, it is now safe to re-create end as begin + 1.
        const beginIndex = requiredDimension(denseSpec.begin, i);
        const xFwd = beginIndex < 0 ? dimI + beginIndex : beginIndex;
        denseSpec.begin[i] = xFwd;
        denseSpec.end[i] = xFwd + 1;
        if (xFwd < 0 || xFwd >= dimI) {
          throw Error(`slice index ${xFwd} of dimension ${
              i} out of bounds.`);
        }
      } else {
        denseSpec.begin[i] = canonical(
            denseSpec.begin[i], 0, stride, dimI, masks,
            validRange);
        denseSpec.end[i] = canonical(
            denseSpec.end[i], 1, stride, dimI, masks, validRange);
      }
      // Update optimization values
      const takeAllInDimension = stride === 1 &&
          denseSpec.begin[i] === 0 && denseSpec.end[i] === dimI;
      isIdentity = isIdentity && takeAllInDimension;
      sliceDim0 = sliceDim0 &&
          ((i === 0 && stride === 1) || takeAllInDimension);
    } else {
      isIdentity =
          isIdentity && ((stride === 1) && beginAndEndMasked);
      sliceDim0 = sliceDim0 &&
          ((i === 0 && stride === 1) || beginAndEndMasked);
    }
    // The intermediate Eigen shape uses -1 for an unknown dimension.
    const intervalLength = knownSliceInterval(
        denseSpec, i, shrinkI, beginAndEndMasked, dimI);
    processingShape.push(intervalLength === null ? -1 :
        intervalElementCount(intervalLength, stride));
  }

  // Step 4: Compute the final shape
  //
  // newAxis will increase dimension by 1 (with a one-size dimension)
  // slices like foo[3, ...] will reduce dimension by 1.
  // This cannot be done earlier, because it depends on Step 3.
  for (let denseDim = 0; denseDim < denseSpec.finalShapeGatherIndices.length;
       ++denseDim) {
    const gatherIndex = requiredDimension(denseSpec.finalShapeGatherIndices, denseDim);
    if (gatherIndex >= 0) {
      finalShape.push(requiredDimension(processingShape, gatherIndex));
      continue;
    }
    if (gatherIndex === NEW_AXIS) {
      finalShape.push(1);
    }
  }

  const finalShapeSparse = finalShape.filter(
      (dim, i) => requiredDimension(denseSpec.finalShapeGatherIndices, i) !== NEW_AXIS);

  return {
    finalShapeSparse,
    finalShape,
    isIdentity,
    sliceDim0,
    isSimpleSlice,
    begin: denseSpec.begin,
    end: denseSpec.end,
    strides: denseSpec.strides
  };
}

/**
 * Determine the extent of a dense slice interval when the indexing metadata
 * contains enough information. Explicit bounds take precedence over masks.
 */
function knownSliceInterval(
    dense: StridedSliceDenseSpec, axis: number, shrink: boolean,
    beginAndEndMasked: boolean, dimension: number): number|null {
  if (dense.beginValid && dense.endValid) {
    return requiredDimension(dense.end, axis) - requiredDimension(dense.begin, axis);
  }
  if (shrink) {
    // A shrunk axis is always one element in the intermediate shape.
    return 1;
  }
  if (beginAndEndMasked && dimension >= 0) {
    return requiredDimension(dense.strides, axis) < 0 ? -dimension : dimension;
  }
  return null;
}

/** Compute the number of indexed elements, including incomplete strides. */
function intervalElementCount(intervalLength: number, stride: number): number {
  if (intervalLength === 0 ||
      ((intervalLength < 0) !== (stride < 0))) {
    return 0;
  }
  return Math.trunc(intervalLength / stride) +
      (intervalLength % stride !== 0 ? 1 : 0);
}

function buildDenseSpec(
    sparse: StridedSliceSparseSpec, dims: number): StridedSliceDenseSpec {
  const dense: StridedSliceDenseSpec = {
    dims,
    beginMask: 0,
    endMask: 0,
    shrinkAxisMask: 0,
    beginValid: sparse.begin != null,
    endValid: sparse.end != null,
    begin: new Array<number>(dims),
    end: new Array<number>(dims),
    strides: new Array<number>(dims),
    finalShapeGatherIndices: [],
    finalShapeGatherIndicesSparse: [],
    inputShapeGatherIndicesSparse: new Array<number>(dims)
  };
  let fullIndex = 0;

  for (let i = 0; i < sparse.dims; i++) {
    if ((1 << i) & sparse.ellipsisMask) {
      // Expand the ellipsis into the appropriate indices. We have verified
      // there is only one ellipsis, so this does not skip a second range.
      const nextIndex = Math.min(
          dense.dims - (sparse.dims - i) + 1 + sparse.numAddAxisAfterEllipsis,
          dense.dims);
      for (; fullIndex < nextIndex; fullIndex++) {
        // A new axis is not a real axis and is therefore skipped.
        dense.begin[fullIndex] = 0;
        dense.end[fullIndex] = 0;
        dense.strides[fullIndex] = 1;
        dense.beginMask |= (1 << fullIndex);
        dense.endMask |= (1 << fullIndex);
        dense.finalShapeGatherIndices.push(fullIndex);
        dense.finalShapeGatherIndicesSparse.push(-1);
        dense.inputShapeGatherIndicesSparse[fullIndex] = i;
      }
      continue;
    }
    if ((1 << i) & sparse.newAxisMask) {
      // A new axis is synthetic, so it does not consume a dense input axis.
      dense.finalShapeGatherIndices.push(NEW_AXIS);
      dense.finalShapeGatherIndicesSparse.push(-1);
      continue;
    }
    fullIndex = appendDenseSliceAxis(sparse, dense, i, fullIndex);
  }
  return dense;
}

/**
 * Append one real sparse slice axis to its dense representation.
 * The returned index is the next available dense axis.
 */
function appendDenseSliceAxis(
    sparse: StridedSliceSparseSpec, dense: StridedSliceDenseSpec,
    i: number, fullIndex: number): number {
  if (fullIndex === dense.begin.length) {
    throw Error(
        `Index out of range using input dim ${fullIndex}; input ` +
        `has only ${dense.dims} dims, ${dense.begin.length}.`);
  }

  // Gather slicing spec into the appropriate index.
  if (sparse.begin != null) {
    dense.begin[fullIndex] = sparse.begin[i];
  }
  if (sparse.end != null) {
    dense.end[fullIndex] = sparse.end[i];
  }
  dense.strides[fullIndex] = requiredDimension(sparse.strides, i);
  if (sparse.beginMask & (1 << i)) {
    dense.beginMask |= (1 << fullIndex);
  }
  if (sparse.endMask & (1 << i)) {
    dense.endMask |= (1 << fullIndex);
  }
  // A shrunk axis does not contribute to the final shape.
  if (sparse.shrinkAxisMask & (1 << i)) {
    dense.finalShapeGatherIndices.push(SHRINK_AXIS);
    dense.finalShapeGatherIndicesSparse.push(-1);
    dense.shrinkAxisMask |= (1 << fullIndex);
  } else {
    dense.finalShapeGatherIndices.push(fullIndex);
    // Remember the sparse dimension from which this dense axis originated.
    dense.finalShapeGatherIndicesSparse.push(i);
  }
  dense.inputShapeGatherIndicesSparse[fullIndex] = i;
  return fullIndex + 1;
}

function canonical(
    x: number, c: number, strideI: number, dimI: number, masks: number[],
    validRange: number[]): number {
  if (requiredDimension(masks, c)) {
    return requiredDimension(validRange, strideI > 0 ? c : (c + 1) & 1);
  }
  const xFwd = x < 0 ? dimI + x : x;  // make negative indices positive
  const lower = requiredDimension(validRange, 0);
  const upper = requiredDimension(validRange, 1);
  return xFwd < lower ? lower : xFwd > upper ? upper : xFwd;
}
