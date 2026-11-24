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
import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {DataStorage, EPSILON_FLOAT16, EPSILON_FLOAT32, KernelBackend} from './backend';

describeWithFlags('epsilon', ALL_ENVS, () => {
  it('Epsilon is a function of float precision', () => {
    const epsilonValue = tf.backend().floatPrecision() === 32 ?
        EPSILON_FLOAT32 :
        EPSILON_FLOAT16;
    expect(tf.backend().epsilon()).toBe(epsilonValue);
  });

  it('abs(epsilon) > 0', async () => {
    expect(await tf.abs(tf.backend().epsilon()).array()).toBeGreaterThan(0);
  });
});


describe('DataStorage', () => {
  it('returns registered data without requesting a transfer', () => {
    const backend = new KernelBackend();
    const moveData = jasmine.createSpy('moveData');
    const storage = new DataStorage<number>(backend, {moveData});
    const dataId = {};
    storage.set(dataId, 0);
    expect(storage.get(dataId)).toBe(0);
    expect(moveData).not.toHaveBeenCalled();
  });

  it('returns data registered by the mover', () => {
    const backend = new KernelBackend();
    const dataId = {};
    const storage = new DataStorage<number>(backend, {
      moveData: (target, id) => {
        expect(target).toBe(backend);
        expect(id).toBe(dataId);
        storage.set(id, 42);
      }
    });
    expect(storage.get(dataId)).toBe(42);
    expect(storage.has(dataId)).toBe(true);
  });

  it('rejects a transfer that fails to register tensor data', () => {
    const backend = new KernelBackend();
    const moveData = jasmine.createSpy('moveData');
    const storage = new DataStorage<number>(backend, {moveData});
    const dataId = {};
    expect(() => storage.get(dataId))
        .toThrowError('Tensor data is missing after backend transfer.');
    expect(moveData).toHaveBeenCalledWith(backend, dataId);
  });
});
