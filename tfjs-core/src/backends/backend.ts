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

import {Backend, DataToGPUOptions, GPUData, Tensor} from '../tensor';
import {DataId} from '../tensor_info';
import {BackendValues, DataType, WebGLData, WebGPUData} from '../types';

export const EPSILON_FLOAT32 = 1e-7;
export const EPSILON_FLOAT16 = 1e-4;

// Required information for all backends.
export interface BackendTimingInfo {
  kernelMs: number|{error: string};
  getExtraProfileInfo?(): string;  // a field for additional timing information
                                   // e.g. packing / unpacking for WebGL backend
}

export interface TensorStorage {
  read(dataId: DataId): Promise<BackendValues>;
  readSync(dataId: DataId): BackendValues;
  disposeData(dataId: DataId, force?: boolean): boolean;
  write(values: BackendValues, shape: number[], dtype: DataType): DataId;
  move(
      dataId: DataId, values: BackendValues, shape: number[], dtype: DataType,
      refCount: number): void;
  memory(): {unreliable: boolean;};  // Backend-specific information.
  /** Returns number of data ids currently in the storage. */
  numDataIds(): number;
  refCount(dataId: DataId): number;
}

/** Convenient class for storing tensor-related data. */
export class DataStorage<T> {
  private data = new WeakMap<DataId, T>();
  private dataIdsCount = 0;

  constructor(private backend: KernelBackend, private dataMover: DataMover) {}

  /**
   * Retrieve registered tensor data, transferring it to this backend if needed.
   * A data mover must register the value before returning.
   */
  get(dataId: DataId): T {
    if (!this.data.has(dataId)) {
      this.dataMover.moveData(this.backend, dataId);
    }
    const value = this.data.get(dataId);
    if (typeof value === 'undefined') {
      throw new Error('Tensor data is missing after backend transfer.');
    }
    return value;
  }

  set(dataId: DataId, value: T): void {
    this.dataIdsCount++;
    this.data.set(dataId, value);
  }

  has(dataId: DataId): boolean {
    return this.data.has(dataId);
  }

  delete(dataId: DataId): boolean {
    this.dataIdsCount--;
    return this.data.delete(dataId);
  }

  numDataIds(): number {
    return this.dataIdsCount;
  }
}

export interface DataMover {
  /**
   * To be called by backends whenever they see a dataId that they don't own.
   * Upon calling this method, the mover will fetch the tensor from another
   * backend and register it with the current active backend.
   */
  moveData(backend: KernelBackend, dataId: DataId): void;
}

export interface BackendTimer {
  // check if backend timer is available
  timerAvailable(): boolean;
  time(f: () => void): Promise<BackendTimingInfo>;
}

/**
 * The interface that defines the kernels that should be implemented when
 * adding a new backend. New backends don't need to implement every one of the
 * methods, this can be done gradually (throw an error for unimplemented
 * methods).
 */
export class KernelBackend implements TensorStorage, Backend, BackendTimer {
  // CG308: The backend extension contract deliberately throws for unsupported refCount.
  //@cg-expect-error
  refCount(dataId: DataId): number {
    return notYetImplemented('refCount');
  }
  // CG308: The backend extension contract deliberately throws for unsupported incRef.
  //@cg-expect-error
  incRef(dataId: DataId): void {
    return notYetImplemented('incRef');
  }
  timerAvailable(): boolean {
    return true;
  }
  // CG308: The backend extension contract deliberately throws for unsupported timing.
  //@cg-expect-error
  time(f: () => void): Promise<BackendTimingInfo> {
    return notYetImplemented('time');
  }
  // CG308: The backend extension contract deliberately throws for unsupported asynchronous reads.
  //@cg-expect-error
  read(dataId: object): Promise<BackendValues> {
    return notYetImplemented('read');
  }
  // CG308: The backend extension contract deliberately throws for unsupported synchronous reads.
  //@cg-expect-error
  readSync(dataId: object): BackendValues {
    return notYetImplemented('readSync');
  }
  // CG308/CG305: Unsupported GPU reads deliberately throw; GPU transfer options are optional.
  //@cg-expect-error
  readToGPU(dataId: object, options?: DataToGPUOptions): GPUData {
    return notYetImplemented('readToGPU');
  }
  // CG308: The backend extension contract deliberately throws if numDataIds is unsupported.
  //@cg-expect-error
  numDataIds(): number {
    return notYetImplemented('numDataIds');
  }
  // CG308/CG305: Unsupported disposal deliberately throws; the force override is optional.
  //@cg-expect-error
  disposeData(dataId: object, force?: boolean): boolean {
    return notYetImplemented('disposeData');
  }
  // CG308: The backend extension contract deliberately throws for unsupported writes.
  //@cg-expect-error
  write(values: BackendValues, shape: number[], dtype: DataType): DataId {
    return notYetImplemented('write');
  }
  // CG308: The backend extension contract deliberately throws for unsupported data migration.
  //@cg-expect-error
  move(
      dataId: DataId, values: BackendValues, shape: number[], dtype: DataType,
      refCount: number): void {
    return notYetImplemented('move');
  }

  // CG308: The backend extension contract deliberately throws for unsupported GPU imports.
  //@cg-expect-error
  createTensorFromGPUData(
      values: WebGLData|WebGPUData, shape: number[], dtype: DataType): Tensor {
    return notYetImplemented('createTensorFromGPUData');
  }

  // CG308/CG305: Unsupported memory queries deliberately throw; reliable estimates need no reasons.
  //@cg-expect-error
  memory(): {unreliable: boolean; reasons?: string[]} {
    return notYetImplemented('memory');
  }
  /** Returns the highest precision for floats in bits (e.g. 16 or 32) */
  // CG308: The backend extension contract deliberately throws for unsupported precision reporting.
  //@cg-expect-error
  floatPrecision(): 16|32 {
    return notYetImplemented('floatPrecision');
  }
  /** Returns the smallest representable number.  */
  epsilon(): number {
    return this.floatPrecision() === 32 ? EPSILON_FLOAT32 : EPSILON_FLOAT16;
  }
  // CG308: The backend extension contract deliberately throws for unsupported backend disposal.
  //@cg-expect-error
  dispose(): void {
    return notYetImplemented('dispose');
  }
}

function notYetImplemented(kernelName: string): never {
  throw new Error(
      `'${kernelName}' not yet implemented or not found in the registry. ` +
      `This kernel may not be supported by the tfjs backend you have chosen`);
}
