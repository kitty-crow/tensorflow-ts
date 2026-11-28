/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {Rank, Tensor} from '@tensorflow/tfjs-core';
import {LayerVariable} from '../variables';
import {Identity} from '../initializers';
import {BaseLogger, History} from '../base_callbacks';
import {getSourceInputs} from './topology';
import {Input, InputLayer, InputLayerArgs} from './input_layer';
import {disposeTensorsInLogs, resolveScalarsInLogs, UnresolvedLogs} from '../logs';
import {analyzeEinsumString} from '../layers/nlp/einsum_dense';
import {BytePairTokenizerCache, StaticHashTable} from '../layers/nlp/tokenizers_utils';
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
import '../flags_layers';

import {ReLULayerConfig} from '../keras_format/layers/advanced_activation_serialization';
import {LayerMetadata} from '../keras_format/topology_config';
import {ReLU} from '../layers/advanced_activations';
import {Dense, Dropout, Reshape} from '../layers/core';
import {Embedding} from '../layers/embeddings';
import {getLossOrMetricName, metricsMap} from '../metrics';
import {lossesMap} from '../losses';
import {MaxPooling1D, MaxPooling2D, MaxPooling3D} from '../layers/pooling';
import {ZeroPadding2D} from '../layers/padding';
import {Dot} from '../layers/merge';
import {TimeDistributed} from '../layers/wrappers';
import {LayerNormalization} from '../layers/normalization';
import {LSTMCell, SimpleRNNCell, StackedRNNCells, standardizeArgs} from '../layers/recurrent';
import {requiredElement, requiredEntry} from '../utils/generic_utils';
import {cloneTestJsonDict, LayerForTest, parseTestJsonDict, requireTestJsonDict} from '../utils/test_utils';
import {min, max, sum, variance} from '../utils/math_utils';
import {Conv2D} from '../layers/convolutional';
import {DepthwiseConv2D} from '../layers/convolutional_depthwise';
import {Sequential} from '../models';
import {getOptimizer} from '../optimizers';
import {collectMetrics, LayersModel, standardizeInputData} from './training';
import {standardizeClassWeights} from './training_utils';
import {disposeNewTensors, ensureTensorsRank2OrHigher} from './training_tensors';

describe('training conditional regression coverage', () => {
  it('retains shared metric arrays and dictionary output order', () => {
    const shared = ['accuracy'];
    const result = collectMetrics(shared, ['first', 'second']);
    expect(requiredElement(result, 0)).toBe(shared);
    expect(requiredElement(result, 1)).toBe(shared);
    const single = collectMetrics('accuracy', ['first', 'second']);
    expect(requiredElement(single, 0)).toBe(requiredElement(single, 1));
    expect(requiredElement(single, 0)).toEqual(['accuracy']);
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

describe('checked tokenizer lookup metadata', () => {
  it('preserves zero token IDs, empty strings and lookup defaults', () => {
    const ids = new StaticHashTable(['zero', 'one'], [0, 1], -1);
    expect(ids.get('zero')).toBe(0);
    expect(ids.get('unknown')).toBe(-1);
    const tokens = new StaticHashTable([0, 1], ['', 'one'], 'missing');
    expect(tokens.get(0)).toBe('');
    expect(tokens.get(2)).toBe('missing');
    const cache = new BytePairTokenizerCache();
    cache.insert(['empty', 'word'], ['', 'w ord']);
    expect(cache.lookup(['word', 'unknown', 'empty'])).toEqual(['w ord', '', '']);
    expect(() => cache.insert(['missing'], [])).toThrowError();
  });
});

describe('checked training history entries', () => {
  it('retains numeric zero, epoch order and independent metric arrays', async () => {
    const history = new History();
    await history.onTrainBegin();
    await history.onEpochEnd(0, {loss: 0, accuracy: 1});
    await history.onEpochEnd(1, {loss: 2, accuracy: 0});
    await history.syncData();
    expect(history.epoch).toEqual([0, 1]);
    expect(history.history).toEqual({loss: [0, 2], accuracy: [1, 0]});
  });
});


describe('numeric layer statistics', () => {
  it('preserves empty, zero and NaN cases for typed and ordinary arrays', () => {
    expect(Number.isNaN(min([]))).toBe(true);
    expect(Number.isNaN(max(new Float32Array()))).toBe(true);
    expect(sum([])).toBe(0);
    const values = new Float32Array([-2, 0, 2]);
    expect(min(values)).toBe(-2);
    expect(max(values)).toBe(2);
    expect(sum(values)).toBe(0);
    expect(variance(values)).toBeCloseTo(8 / 3);
    expect(min([NaN, 0])).toBe(0);
    expect(Number.isNaN(sum([NaN, 0]))).toBe(true);
  });
});

describe('checked model graph reconstruction', () => {
  it('reconstructs a weightless symbolic graph and checks required output indices', () => {
    const config = {
      name: 'checked_graph',
      inputLayers: [['checked_input', 0, 0]],
      outputLayers: [['checked_relu', 0, 0]],
      layers: [
        {
          className: 'InputLayer',
          config: {batchInputShape: [null, 2], dtype: 'float32', name: 'checked_input', sparse: false},
          name: 'checked_input',
          inboundNodes: [] as string[][]
        },
        {
          className: 'ReLU',
          config: {name: 'checked_relu'},
          name: 'checked_relu',
          inboundNodes: [[['checked_input', 0, 0, {}]]]
        }
      ]
    };
    const model = LayersModel.fromConfig(LayersModel, config);
    if (!(model instanceof LayersModel)) {
      throw new Error('Expected graph reconstruction to return a LayersModel');
    }
    expect(model.name).toBe('checked_graph');
    expect(model.findLayer(0)).toBe(requiredElement(model.layers, 0));
    expect(() => model.findLayer(-1)).toThrowError(/Missing required element/);
    expect(() => model.findLayer(0.5)).toThrowError(/Missing required element/);
    expect(requiredElement(model.inputs, 0).shape).toEqual([null, 2]);
    expect(requiredElement(model.outputs, 0).shape).toEqual([null, 2]);
    expect(() => LayersModel.fromConfig(LayersModel, {
      ...config, outputLayers: [['checked_relu', 0, 1]]
    })).toThrowError(/Missing required element/);
  });
});


describe('typed JSON test fixture boundaries', () => {
  it('preserves JSON values and makes independent nested copies', () => {
    const source = {config: {name: 'original', count: 0, enabled: false},
      shape: [null, 2]};
    const copy = cloneTestJsonDict(source);
    requireTestJsonDict(copy['config'], 'config')['name'] = 'changed';
    expect(source.config.name).toBe('original');
    expect(JSON.stringify(copy)).toBe(
        '{"config":{"name":"changed","count":0,"enabled":false},"shape":[null,2]}');
  });
  it('rejects non-dictionary JSON rather than passing it to fixtures', () => {
    expect(() => parseTestJsonDict('null')).toThrowError(/JSON dictionary/);
    expect(() => parseTestJsonDict('[1]')).toThrowError(/JSON dictionary/);
    expect(() => parseTestJsonDict('false')).toThrowError(/JSON dictionary/);
    expect(() => parseTestJsonDict('{')).toThrow();
    expect(() => requireTestJsonDict(null, 'config')).toThrowError(/config/);
  });
});


describe('checked training log and source graph contracts', () => {
  it('retains weighted numeric metrics and skips absent metrics', async () => {
    const logger = new BaseLogger();
    logger.setParams({metrics: ['loss', 'absent']});
    await logger.onEpochBegin(0);
    await logger.onBatchEnd(0, {size: 2, loss: 0});
    await logger.onBatchEnd(1, {size: 1, loss: 6});
    const logs: UnresolvedLogs = {};
    await logger.onEpochEnd(0, logs);
    expect(logs).toEqual({loss: 2});
  });

  it('resolves scalar logs in place and disposes only the original tensors', async () => {
    const scalar = new Tensor<Rank.R0>([], 'float32', {}, 100);
    spyOn(scalar, 'data').and.returnValue(Promise.resolve(new Float32Array([4])));
    const disposed = spyOn(scalar, 'dispose');
    const logs: UnresolvedLogs = {scalar, zero: 0};
    await resolveScalarsInLogs(logs);
    expect(logs).toEqual({scalar: 4, zero: 0});
    expect(disposed).toHaveBeenCalledTimes(1);
    disposeTensorsInLogs(logs);
    expect(disposed).toHaveBeenCalledTimes(1);
    disposeTensorsInLogs({scalar});
    expect(disposed).toHaveBeenCalledTimes(2);
  });

  it('follows symbolic source identity through multiple layers', () => {
    const model = new Sequential({layers: [new ReLU({inputShape: [2]}), new ReLU({})]});
    const input = requiredElement(model.inputs, 0);
    const output = requiredElement(model.outputs, 0);
    expect(getSourceInputs(output)).toEqual([input]);
    expect(getSourceInputs(input)).toEqual([input]);
  });
});


describe('einsum shape indexing', () => {
  it('preserves symbolic dimensions and leading or trailing ellipses', () => {
    expect(analyzeEinsumString('...b,bc->...c', 'c', [null, null, 32], [64]))
        .toEqual([[32, 64], [64], [null, null, 64]]);
    expect(analyzeEinsumString('bc...,cde->bde...', 'e', [null, 1, 2], [3, 4]))
        .toEqual([[1, 3, 4], [4, 1], [null, 3, 4, 2]]);
    expect(analyzeEinsumString('ab,bc->ac', null, [null, 0], [4]))
        .toEqual([[0, 4], null, [null, 4]]);
  });
  it('rejects a required missing input dimension', () => {
    expect(() => analyzeEinsumString('ab,bc->ac', null, [null], [4]))
        .toThrowError(/Missing einsum dimension/);
  });
});

describe('input layer shape normalisation', () => {
  it('retains symbolic batch dimensions and explicit zero dimensions', () => {
    const args: InputLayerArgs = {inputShape: [0, 2]};
    const layer = new InputLayer(args);
    expect(args.batchSize).toBeNull();
    expect(args.sparse).toBe(false);
    expect(layer.batchInputShape).toEqual([null, 0, 2]);
    expect(Input({shape: [2]}).shape).toEqual([null, 2]);
    const input = Input({batchShape: [0, 3], dtype: 'int32', sparse: true});
    expect(input.shape).toEqual([0, 3]);
    expect(input.dtype).toBe('int32');
  });

  it('rejects missing or conflicting shape arguments', () => {
    expect(() => Input({})).toThrowError(/Please provide/);
    expect(() => Input({shape: [2], batchShape: [1, 2]})).toThrowError(/not both/);
    expect(() => new InputLayer({})).toThrowError(/An InputLayer/);
    expect(() => new InputLayer({inputShape: [2], batchInputShape: [1, 2]}))
        .toThrowError(/not both/);
    expect(() => new InputLayer({batchInputShape: [0, 2], batchSize: 0}))
        .toThrowError(/Cannot specify batchSize/);
  });
});

describe('model summary traversal', () => {
  it('prints layer rows and parameter counts through a string callback', () => {
    const model = new Sequential({layers: [
      new ReLU({inputShape: [2], name: 'summary_relu'})
    ]});
    const rows: string[] = [];
    model.summary(90, null, (line: string) => { rows.push(line); });
    expect(rows.some(line => line.includes('summary_relu (ReLU)'))).toBe(true);
    expect(rows).toContain('Total params: 0');
    expect(rows).toContain('Trainable params: 0');
    expect(rows).toContain('Non-trainable params: 0');
  });
});

describe('stacked RNN weight assignment', () => {
  it('uses cell order, leaves the supplied array intact, and checks shapes first', () => {
    const first = new SimpleRNNCell({units: 2});
    const second = new SimpleRNNCell({units: 3});
    const a = jasmine.createSpyObj<LayerVariable>('firstWeight', ['read', 'write'],
        {shape: [1, 2]});
    const b = jasmine.createSpyObj<LayerVariable>('secondWeight', ['read', 'write'],
        {shape: [2, 3]});
    spyOnProperty(first, 'weights', 'get').and.returnValue([a]);
    spyOnProperty(second, 'weights', 'get').and.returnValue([b]);
    const stack = new StackedRNNCells({cells: [first, second]});
    const firstValue = new Tensor([1, 2], 'float32', {}, 201);
    const secondValue = new Tensor([2, 3], 'float32', {}, 202);
    const weights = [firstValue, secondValue];
    a.read.and.returnValue(firstValue);
    b.read.and.returnValue(secondValue);
    expect(stack.getWeights()).toEqual(weights);
    expect(() => stack.setWeights([firstValue])).toThrowError(/Expected 2 weights/);
    expect(() => stack.setWeights([firstValue, firstValue])).toThrowError(/Weight 1 shape/);
    expect(a.write).not.toHaveBeenCalled();
    expect(b.write).not.toHaveBeenCalled();
    stack.setWeights(weights);
    expect(a.write).toHaveBeenCalledOnceWith(firstValue);
    expect(b.write).toHaveBeenCalledOnceWith(secondValue);
    expect(weights).toEqual([firstValue, secondValue]);
  });
});

describe('layer boundary contracts', () => {
  it('uses input dimensions only for absent dropout noise dimensions', () => {
    class InspectableDropout extends Dropout {
      noiseShapeFor(input: Tensor) {
        return this.getNoiseShape(input);
      }
    }
    const input = new Tensor<Rank.R2>([2, 3], 'float32', {}, 0);
    expect(new InspectableDropout({rate: 0.5, noiseShape: [null, 1]})
               .noiseShapeFor(input)).toEqual([2, 1]);
    expect(new InspectableDropout({rate: 0.5, noiseShape: [0, 1]})
               .noiseShapeFor(input)).toEqual([0, 1]);
    expect(() => new InspectableDropout(
        {rate: 0.5, noiseShape: [1, 1, null]}).noiseShapeFor(input))
        .toThrowError();
  });

  it('forwards the public updates contract through nested wrappers', () => {
    const updates = [new Tensor<Rank.R1>([1], 'float32', {}, 0)];
    class UpdatedLayer extends LayerForTest {
      override get updates(): Tensor[] {
        return updates;
      }
    }
    const inner = new TimeDistributed({layer: new UpdatedLayer()});
    const outer = new TimeDistributed({layer: inner});
    expect(inner.updates).toBe(updates);
    expect(outer.updates).toBe(updates);
  });
});

describe('optimizer name dispatch', () => {
  it('rejects unknown names, including inherited object members', () => {
    for (const name of ['constructor', 'toString', 'ADAM', 'unknown']) {
      expect(() => getOptimizer(name)).toThrowError(/Unknown Optimizer/);
    }
  });
});

describe('omitted public layer configuration', () => {
  it('retains defaults and explicit false or zero overrides', () => {
    const recurrent = new SimpleRNNCell({units: 2}).getConfig();
    expect(requiredEntry(recurrent, 'useBias')).toBe(true);
    expect(requiredEntry(recurrent, 'dropout')).toBe(0);
    expect(requiredEntry(recurrent, 'recurrentDropout')).toBe(0);
    const explicit =
        new SimpleRNNCell({units: 2, useBias: false, dropout: 0}).getConfig();
    expect(requiredEntry(explicit, 'useBias')).toBe(false);
    const convolution = new Conv2D({filters: 2, kernelSize: 3}).getConfig();
    expect(requiredEntry(convolution, 'strides')).toEqual([1, 1]);
    expect(requiredEntry(convolution, 'padding')).toBe('valid');
    expect(requiredEntry(convolution, 'dataFormat')).toBe('channelsLast');
    const normalization =
        new LayerNormalization({center: false, scale: false}).getConfig();
    expect(requiredEntry(normalization, 'center')).toBe(false);
    expect(requiredEntry(normalization, 'scale')).toBe(false);
    expect(requiredEntry(normalization, 'epsilon')).toBe(0.001);
  });
});


describe('single-read configuration defaults', () => {
  it('keeps explicit zero gain and reads the supplied property once', () => {
    let reads = 0;
    const initializer = new Identity({get gain() {
      reads++;
      return 0;
    }});
    expect(reads).toBe(1);
    expect(requiredEntry(initializer.getConfig(), 'gain')).toBe(0);
    expect(requiredEntry(new Identity({}).getConfig(), 'gain')).toBe(1);
  });
});

describe('pooling and padding shape contracts', () => {
  it('preserves pooling stride defaults and explicit tuple identity', () => {
    const pool: [number, number] = [2, 3];
    const stride: [number, number, number] = [1, 2, 3];
    const one = new MaxPooling1D({poolSize: 3});
    const two = new MaxPooling2D({poolSize: pool});
    const three = new MaxPooling3D({poolSize: 2, strides: stride});
    expect(one.getConfig().strides).toEqual([3]);
    expect(two.getConfig().poolSize).toBe(pool);
    expect(two.getConfig().strides).toBe(pool);
    expect(three.getConfig().strides).toBe(stride);
    expect(three.getConfig().poolSize).toEqual([2, 2, 2]);
    expect(() => new MaxPooling1D({strides: 0})).toThrowError();
    expect(() => new MaxPooling2D({poolSize: 0})).toThrowError();
    expect(() => new MaxPooling3D({strides: 0})).toThrowError();
  });

  it('normalizes padding and preserves unknown spatial dimensions', () => {
    expect(new ZeroPadding2D().padding).toEqual([[1, 1], [1, 1]]);
    expect(new ZeroPadding2D({padding: 0}).padding).toEqual([[0, 0], [0, 0]]);
    const layer = new ZeroPadding2D({padding: [2, 3]});
    expect(layer.computeOutputShape([null, 4, -1, 2]))
        .toEqual([null, 8, null, 2]);
    const first = new ZeroPadding2D({
      padding: [[1, 2], [3, 4]], dataFormat: 'channelsFirst'
    });
    expect(first.computeOutputShape([null, 2, 4, null]))
        .toEqual([null, 2, 7, null]);
  });
});

describe('layer shape defaults and metric names', () => {
  it('preserves explicit zero batch size and omitted batch size', () => {
    expect(new Dense({units: 2, inputDim: 3, batchSize: 0}).batchInputShape)
        .toEqual([0, 3]);
    expect(new Dense({units: 2, inputDim: 3}).batchInputShape)
        .toEqual([null, 3]);
    const embedding = new Embedding({inputDim: 8, outputDim: 2, inputLength: [3, 4]});
    expect(embedding.batchInputShape).toEqual([null, 3, 4]);
    expect(embedding.computeOutputShape([null, 3, 4])).toEqual([null, 3, 4, 2]);
    expect(() => embedding.computeOutputShape([null, 3, 5])).toThrowError();
  });
  it('preserves the first registered loss or metric name and custom names', () => {
    for (const [key, fn] of Object.entries(lossesMap)) {
      const first = Object.entries(lossesMap).find(([, loss]) => loss === fn);
      expect(getLossOrMetricName(fn)).toBe(first == null ? key : first[0]);
    }
    expect(getLossOrMetricName(requiredEntry(metricsMap, 'binaryAccuracy')))
        .toBe('binaryAccuracy');
    const customMetric = (actual: Tensor, predicted: Tensor) => actual;
    expect(getLossOrMetricName(customMetric)).toBe('customMetric');
    expect(getLossOrMetricName('custom')).toBe('custom');
  });
});

describe('dot shape validation', () => {
  it('shares shape validation between build and inference', () => {
    const dot = new Dot({axes: 1});
    dot.build([[null, 3], [null, 3]]);
    expect(dot.computeOutputShape([[null, 3], [null, 3]])).toEqual([null, 1]);
    expect(() => dot.build([[null, 2], [null, 3]]))
        .toThrowError(/Dimension incompatibility/);
    const negative = new Dot({axes: [-1, -1]});
    negative.build([[null, 3], [null, 3]]);
    expect(negative.computeOutputShape([[null, 3], [null, 3]])).toEqual([null, 1]);
    expect(() => dot.computeOutputShape([null, 3]))
        .toThrowError(/exactly 2 inputs/);
  });
});


describe('class weight normalization branches', () => {
  it('preserves a single array, named weights, and output ordering', () => {
    const first = {0: 2};
    const second = {1: 3};
    const single = [first];
    expect(standardizeClassWeights(single, ['first'])).toBe(single);
    expect(standardizeClassWeights({first}, ['first'])).toEqual([first]);
    expect(standardizeClassWeights(first, ['first'])).toEqual([first]);
    expect(standardizeClassWeights({second, first}, ['first', 'second']))
        .toEqual([first, second]);
    expect(() => standardizeClassWeights(single, ['first', 'second']))
        .toThrowError(/model has 2 outputs/);
  });
});

describe('reshape dimension validation regressions', () => {
  it('infers one unknown dimension and preserves fully known shapes', () => {
    expect(new Reshape({targetShape: [-1, 2]}).computeOutputShape([2, 2, 3]))
        .toEqual([2, 3, 2]);
    expect(new Reshape({targetShape: [3, 2]}).computeOutputShape([2, 2, 3]))
        .toEqual([2, 3, 2]);
    expect(new Reshape({targetShape: [0, 2]}).computeOutputShape([2, 0, 3]))
        .toEqual([2, 0, 2]);
  });

  it('rejects multiple unknown dimensions and mismatched total sizes', () => {
    expect(() => new Reshape({targetShape: [-1, -1]})
        .computeOutputShape([2, 2, 3]))
        .toThrowError('Can only specifiy one unknown dimension.');
    expect(() => new Reshape({targetShape: [-1, 4]})
        .computeOutputShape([2, 2, 3]))
        .toThrowError('Total size of new array must be unchanged.');
    expect(() => new Reshape({targetShape: [4, 2]})
        .computeOutputShape([2, 2, 3]))
        .toThrowError('Total size of new array must be unchanged.');
  });
});
