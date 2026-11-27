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

import {backend_util, DataType, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {MeanProgram} from '../mean_gpu';
import {ReduceProgram} from '../reduce_gpu';

type ReduceTypes = 'all'|'any'|'max'|'min'|'sum'|'prod'|'mean';

// Returns an array of configuration objects that describe each stage of the
// reduction.
function getReductionStages(inShape: number[]):
    Array<{inSize: number, windowSize: number, outSize: number}> {
  const stages: Array<{inSize: number, windowSize: number, outSize: number}> = [];

  let inSize = util.requiredNumericEntry(inShape, 1);
  do {
    const windowSize = backend_util.computeOptimalWindowSize(inSize);
    const outSize = Math.ceil(inSize / windowSize);
    stages.push({inSize, windowSize, outSize});
    inSize = outSize;
  } while (inSize !== 1);

  return stages;
}

export function reduce(
    x: TensorInfo, dtype: DataType, reductionType: ReduceTypes,
    backend: MathBackendWebGL): TensorInfo {
  const reductionStages = getReductionStages(x.shape);

  let result = x;
  for (const [i, {inSize, windowSize, outSize}] of reductionStages.entries()) {
    const batchSize = util.requiredNumericEntry(x.shape, 0);

    let program: ReduceProgram|MeanProgram;
    let previousResult: TensorInfo;
    if (reductionType === 'mean') {
      program = i === 0 ?
          new MeanProgram(
              {windowSize, inSize, batchSize, outSize}, inSize) :
          new MeanProgram({windowSize, inSize, batchSize, outSize});
    } else {
      program = new ReduceProgram(
          {windowSize, inSize, batchSize, outSize}, reductionType);
    }

    previousResult = result;
    result = backend.runWebGLProgram(program, [result], dtype);

    if (previousResult.dataId !== x.dataId) {
      backend.disposeIntermediateTensorInfo(previousResult);
    }
  }

  return result;
}
