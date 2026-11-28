/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {SparseSegmentSumInputs} from '../../kernel_names';
import {Tensor, Tensor1D} from '../../tensor';
import {convertToTensor} from '../../tensor_util_env';
import {TensorLike} from '../../types';

/** Convert and check the shared sparse-segment input contract. */
export function sparseSegmentInputs(
    data: Tensor|TensorLike, indices: Tensor1D|TensorLike,
    segmentIds: Tensor1D|TensorLike,
    opName: 'sparseSegmentSum'|'sparseSegmentMean'): SparseSegmentSumInputs {
  const $data = convertToTensor(data, 'data', opName);
  const $indices =
      convertToTensor(indices, 'indices', opName, 'int32');
  const $segmentIds =
      convertToTensor(segmentIds, 'segmentIds', opName, 'int32');

  const errorIndent = opName === 'sparseSegmentMean' ? '          ' : '         ';
  if ($data.rank < 1) {
    throw new Error(
        `Data should be at least 1 dimensional but received scalar`);
  }
  if ($indices.rank !== 1) {
    throw new Error(`Indices should be Tensor1D but received shape
${errorIndent}${$indices.shape}`);
  }
  if ($segmentIds.rank !== 1) {
    throw new Error(`Segment ids should be Tensor1D but received shape
${errorIndent}${$segmentIds.shape}`);
  }

  return {
    data: $data,
    indices: $indices,
    segmentIds: $segmentIds
  };
}
