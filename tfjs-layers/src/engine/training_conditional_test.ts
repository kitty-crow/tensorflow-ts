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

import {ReLULayerConfig} from '../keras_format/layers/advanced_activation_serialization';
import {LayerMetadata} from '../keras_format/topology_config';
import {ReLU} from '../layers/advanced_activations';
import {LayerNormalization} from '../layers/normalization';
import {LSTMCell, SimpleRNNCell, StackedRNNCells, standardizeArgs} from '../layers/recurrent';
import {requiredElement, requiredEntry} from '../utils/generic_utils';
import {Conv2D} from '../layers/convolutional';
import {DepthwiseConv2D} from '../layers/convolutional_depthwise';
import {Sequential} from '../models';
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

describe('recurrent argument parsing regression coverage', () => {
  it('splits positional input, recurrent states and trailing constants', () => {
    const input = new Tensor([1, 2, 3], 'float32', {}, 11);
    const state = new Tensor([1, 4], 'float32', {}, 12);
    const constant = new Tensor([1, 5], 'float32', {}, 13);
    const parsed = standardizeArgs([input, state, constant], null, null, 1);
    expect(parsed.inputs).toBe(input);
    expect(parsed.initialState).toEqual([state]);
    expect(parsed.constants).toEqual([constant]);
    expect(standardizeArgs(input, state, constant).initialState).toEqual([state]);
    expect(() => standardizeArgs([input], state, null)).toThrowError(/neither initialState or constants/);
    expect(() => standardizeArgs([], null, null)).toThrowError(/requires its input tensor/);
  });
});

describe('shared layer metadata contract', () => {
  it('preserves runtime and Keras configuration fields and casing', () => {
    const metadata: LayerMetadata = {
      dtype: 'int32', name: 'metadata_regression', trainable: false
    };
    const serialised: ReLULayerConfig = {
      ...metadata, input_shape: [2], max_value: 6
    };
    const layer = new ReLU({...metadata, inputShape: [2], maxValue: 6});
    expect(layer.name).toBe(serialised.name);
    expect(layer.dtype).toBe(serialised.dtype);
    expect(layer.trainable).toBe(serialised.trainable);
    expect(layer.batchInputShape).toEqual([null, 2]);
    expect(layer.maxValue).toBe(serialised.max_value);
    const config = layer.getConfig();
    expect(config.name === serialised.name).toBe(true);
    expect(config.dtype === serialised.dtype).toBe(true);
    expect(config.trainable === serialised.trainable).toBe(true);
    expect(config.maxValue === serialised.max_value).toBe(true);
    const omitted: LayerMetadata = {};
    const defaultLayer = new ReLU({...omitted, inputShape: [2]});
    expect(defaultLayer.dtype).toBe('float32');
    expect(defaultLayer.trainable).toBe(true);
  });
});


describe('required layer entries', () => {
  it('preserves symbolic null dimensions, zero values and identity', () => {
    const shape = [null, 0, 3];
    expect(requiredElement(shape, 0)).toBeNull();
    expect(requiredElement(shape, 1)).toBe(0);
    expect(requiredEntry({shape}, 'shape')).toBe(shape);
    expect(requiredEntry({zero: 0}, 'zero')).toBe(0);
    expect(() => requiredElement(shape, 3)).toThrowError(/Missing required element/);
    expect(() => requiredEntry({shape}, 'other')).toThrowError(/Missing required entry/);
  });
  it('retains symbolic convolution shapes in both layouts', () => {
    const last = new Conv2D({filters: 3, kernelSize: 3, padding: 'same'});
    const first = new Conv2D({filters: 3, kernelSize: 3, padding: 'same', dataFormat: 'channelsFirst'});
    expect(last.computeOutputShape([null, 5, null, 2])).toEqual([null, 5, null, 3]);
    expect(first.computeOutputShape([null, 2, 5, null])).toEqual([null, 3, 5, null]);
  });
  it('retains unknown reference dimensions and rejects rank and size mismatches', () => {
    const a = new Tensor([1, 2], 'float32', {}, 51);
    expect(standardizeInputData([a], ['a'], [[null, 2]], false)).toEqual([a]);
    expect(() => standardizeInputData([a], ['a'], [[null, 3]], false)).toThrowError();
    expect(() => standardizeInputData([a], ['a'], [[null, 2, 1]], false)).toThrowError();
  });
});

describe('checked symbolic model and depthwise shapes', () => {
  it('keeps sequential graph outputs aligned when adding and removing layers', () => {
    const model = new Sequential({layers: [new ReLU({inputShape: [2]})]});
    const firstOutput = requiredElement(model.outputs, 0);
    expect(firstOutput.shape).toEqual([null, 2]);
    model.add(new ReLU({}));
    expect(requiredElement(model.outputs, 0).shape).toEqual([null, 2]);
    expect(model.layers.length).toBe(2);
    model.pop();
    expect(requiredElement(model.outputs, 0)).toBe(firstOutput);
    expect(model.layers.length).toBe(1);
  });
  it('keeps unknown spatial dimensions through depthwise convolution', () => {
    const layer = new DepthwiseConv2D({kernelSize: 3, depthMultiplier: 2, padding: 'same'});
    expect(layer.computeOutputShape([null, 5, null, 2])).toEqual([null, 5, null, 4]);
  });
});

describe('checked callback shape contracts', () => {
  it('normalizes a single nested shape without allocating weights', () => {
    const layer = new LayerNormalization({axis: [-1, 1], center: false, scale: false});
    layer.build([[null, 0, 4]]);
    expect(layer.built).toBe(true);
    expect(layer.getConfig().axis).toEqual([2, 1]);
    const missing = new LayerNormalization({axis: 1, center: false, scale: false});
    const shape = new Array<number>(2);
    shape[0] = 1;
    expect(() => missing.build(shape)).toThrowError(/Missing required element/);
  });

  it('passes the selected input and each preceding cell output shape in order', () => {
    const first = new SimpleRNNCell({units: 3});
    const second = new LSTMCell({units: 5});
    const third = new SimpleRNNCell({units: 7});
    const firstBuild = spyOn(first, 'build');
    const secondBuild = spyOn(second, 'build');
    const thirdBuild = spyOn(third, 'build');
    const stack = new StackedRNNCells({cells: [first, second, third]});
    const shape = [null, 2];
    stack.build([shape, [null, 9]]);
    expect(firstBuild.calls.mostRecent().args[0]).toBe(shape);
    expect(secondBuild).toHaveBeenCalledOnceWith([null, 3]);
    expect(thirdBuild).toHaveBeenCalledOnceWith([null, 5]);
    expect(stack.built).toBe(true);
    const flatShape = [0, 2];
    stack.build(flatShape);
    expect(firstBuild.calls.mostRecent().args[0]).toBe(flatShape);
    expect(secondBuild.calls.mostRecent().args[0]).toEqual([0, 3]);
    expect(thirdBuild.calls.mostRecent().args[0]).toEqual([0, 5]);
  });
});
