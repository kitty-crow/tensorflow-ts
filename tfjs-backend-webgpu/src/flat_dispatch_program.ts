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

import {computeDispatch, flatDispatchLayout} from './webgpu_util';

/** Shared geometry and bounds-checking metadata for flat output dispatch. */
export abstract class FlatDispatchProgram {
  outputShape: number[];
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames: string[];
  workgroupSize: [number, number, number];
  size = true;

  protected constructor(
      outputShape: number[], workgroupSize: [number, number, number],
      variableNames: string[], elementsPerThread: [number, number, number] =
          [1, 1, 1]) {
    this.outputShape = outputShape;
    this.workgroupSize = workgroupSize;
    this.variableNames = variableNames;
    this.dispatchLayout = flatDispatchLayout(outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, outputShape, workgroupSize, elementsPerThread);
  }
}
