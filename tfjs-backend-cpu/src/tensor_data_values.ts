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

import type {backend_util, DataType, TensorInfo} from '@tensorflow/tfjs-core';

import type {TensorData} from './backend_cpu';

/** Reject an absent payload before passing it to a CPU kernel. */
export function requireTensorDataValues(data: TensorData<DataType>):
    backend_util.BackendValues {
  const values = data.values ?? null;
  if (values === null) {
    throw new Error('CPU tensor data has no stored values.');
  }
  return values;
}

/** Reject a complex tensor whose component ownership has not been established. */
export function requireComplexTensorInfos(data: TensorData<DataType>):
    {real: TensorInfo, imag: TensorInfo} {
  const parts = data.complexTensorInfos ?? null;
  if (parts === null) {
    throw new Error('CPU tensor data has no complex components.');
  }
  return parts;
}
