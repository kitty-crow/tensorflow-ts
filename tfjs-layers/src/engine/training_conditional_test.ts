/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {Tensor} from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
import '../flags_layers';

import {collectMetrics, standardizeInputData} from './training';
import {disposeNewTensors, ensureTensorsRank2OrHigher} from './training_tensors';

describe('training conditional regression coverage', () => {
  it('retains shared metric arrays and dictionary output order', () => {
    const shared = ['accuracy'];
    const result = collectMetrics(shared, ['first', 'second']);
    expect(result[0]).toBe(shared);
    expect(result[1]).toBe(shared);
    const single = collectMetrics('accuracy', ['first', 'second']);
    expect(single[0]).toBe(single[1]);
    expect(single[0]).toEqual(['accuracy']);
    expect(collectMetrics({second: 'mse'}, ['first', 'second']))
        .toEqual([[], ['mse']]);
  });

  it('preserves named input order and reports missing or excess inputs', () => {
    const a = new Tensor([1, 2], 'float32', {}, 1);
    const b = new Tensor([1, 2], 'float32', {}, 2);
    expect(standardizeInputData({a, b}, ['b', 'a'])).toEqual([b, a]);
    expect(() => standardizeInputData({a}, ['a', 'b']))
        .toThrowError(/No data provided for "b"/);
    expect(() => standardizeInputData([a, b], ['a'])).toThrowError();
    expect(standardizeInputData(null, [])).toEqual([]);
    expect(() => standardizeInputData([], [])).toThrowError();
  });

  it('preserves rank-two tensors and rejects scalars before expanding', () => {
    const matrix = new Tensor([1, 2], 'float32', {}, 1);
    const scalar = new Tensor([], 'float32', {}, 2);
    expect(ensureTensorsRank2OrHigher(matrix)[0]).toBe(matrix);
    expect(() => ensureTensorsRank2OrHigher(scalar))
        .toThrowError(/Expected tensor to be at least 1D/);
  });

  it('disposes only new tensor identities across container representations', () => {
    // Use actual Tensor instances, observing dispose without allocating a backend.
    const retained = new Tensor([1, 2], 'float32', {}, 1);
    const added = new Tensor([1, 2], 'float32', {}, 2);
    const retainedDispose = spyOn(retained, 'dispose');
    const addedDispose = spyOn(added, 'dispose');
    disposeNewTensors({retained, added}, [retained]);
    expect(retainedDispose).not.toHaveBeenCalled();
    expect(addedDispose).toHaveBeenCalledTimes(1);
    addedDispose.calls.reset();
    disposeNewTensors(added, {retained});
    expect(addedDispose).toHaveBeenCalledTimes(1);
    addedDispose.calls.reset();
    disposeNewTensors([added], added);
    expect(addedDispose).not.toHaveBeenCalled();
  });
});
