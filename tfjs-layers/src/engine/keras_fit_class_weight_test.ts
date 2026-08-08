/**
 * @license
 * Copyright 2026
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import * as tfc from '@tensorflow/tfjs-core';

import * as tfl from '../index';
import {describeMathCPU} from '../utils/test_utils';

function makeClassifier(): tfl.LayersModel {
  const input = tfl.input({shape: [1], name: 'input'});
  const output = tfl.layers.dense({
    units: 2,
    activation: 'softmax',
    useBias: false,
    kernelInitializer: 'zeros',
    name: 'classes'
  }).apply(input) as tfl.SymbolicTensor;
  return tfl.model({inputs: input, outputs: output});
}

describeMathCPU('Keras fit class_weight compatibility', () => {
  it('accepts canonical class_weight and defaults omitted classes to one',
     async () => {
       const model = makeClassifier();
       const xs = tfc.zeros([2, 1]);
       const ys = tfc.tensor2d([[1, 0], [0, 1]]);
       try {
         model.compile({
           optimizer: tfc.train.sgd(0.01),
           loss: 'categoricalCrossentropy'
         });
         const history = await model.fit(xs, ys, {
           batchSize: 2,
           epochs: 1,
           verbose: 0,
           shuffle: false,
           class_weight: {1: 3}
         });
         expect(history.history.loss.length).toBe(1);
         // The only batch begins from zero logits, so the returned training
         // loss is the weighted pre-update categorical crossentropy.
         expect(history.history.loss[0] as number)
             .toBeCloseTo(2 * Math.log(2), 5);
       } finally {
         model.dispose();
         tfc.dispose([xs, ys]);
       }
     });

  it('rejects class_weight and sample_weight together in fit', async () => {
    const model = makeClassifier();
    const xs = tfc.zeros([2, 1]);
    const ys = tfc.tensor2d([[1, 0], [0, 1]]);
    const sampleWeight = tfc.ones([2]);
    let caught: Error;
    try {
      model.compile({optimizer: 'sgd', loss: 'categoricalCrossentropy'});
      try {
        await model.fit(xs, ys, {
          epochs: 1,
          verbose: 0,
          class_weight: {0: 1, 1: 2},
          sample_weight: sampleWeight
        });
      } catch (error) {
        caught = error;
      }
      expect(caught.message).toMatch(/sample_weight.*class_weight.*same time/);
    } finally {
      model.dispose();
      tfc.dispose([xs, ys, sampleWeight]);
    }
  });
});
