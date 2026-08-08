# TensorFlow.js Layers: High-Level Machine Learning Model API

A part of the TensorFlow.js ecosystem, TensorFlow.js Layers is a high-level
API built on [TensorFlow.js Core](/tfjs-core),
enabling users to build, train and execute deep learning models in the browser.
TensorFlow.js Layers is modeled after
[Keras](https://keras.io/) and
[tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras) and can
load models saved from those libraries.

## Importing

There are three ways to import TensorFlow.js Layers

1. You can access TensorFlow.js Layers through the union package
   between the TensorFlow.js Core and Layers:
   [@tensorflow/tfjs](https://www.npmjs.com/package/@tensorflow/tfjs)
2. You can get [TensorFlow.js](https://github.com/tensorflow/tfjs) Layers as a module:
   [@tensorflow/tfjs-layers](https://www.npmjs.com/package/@tensorflow/tfjs-layers).
   Note that `tfjs-layers` has peer dependency on tfjs-core, so if you import
   `@tensorflow/tfjs-layers`, you also need to import
   `@tensorflow/tfjs-core`.
3. As a standalone through [unpkg](https://unpkg.com/).

Option 1 is the most convenient, but leads to a larger bundle size (we will be
adding more packages to it in the future). Use option 2 if you care about bundle
size.

## Getting started

### Building, training and executing a model

The following example shows how to build a toy model with only one `dense` layer
to perform linear regression.

```js
import * as tf from '@tensorflow/tfjs';

// A sequential model is a container which you can add layers to.
const model = tf.sequential();

// Add a dense layer with 1 output unit.
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Specify the loss type and optimizer for training.
model.compile({loss: 'meanSquaredError', optimizer: 'SGD'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

// Train the model.
await model.fit(xs, ys, {epochs: 500});

// After the training, perform inference.
const output = model.predict(tf.tensor2d([[5]], [1, 1]));
output.print();
```

### Keras-compatible multi-output loss weights

This fork supports Keras-compatible `loss_weights` in `LayersModel.compile()`.
Each output loss is reduced independently, multiplied by its configured weight,
and then included in the total optimisation loss. Per-output loss metrics remain
unweighted, matching Keras behaviour.

Use the same snake-case name as Keras:

```js
model.compile({
  optimizer: 'adam',
  loss: {
    category: 'categoricalCrossentropy',
    offset: 'meanSquaredError'
  },
  loss_weights: {
    category: 2,
    offset: 0.5
  }
});
```

For multi-output models, `loss_weights` may be an array in model-output order or
a dictionary keyed by output name. A numeric scalar is accepted for a
single-output model. The TensorFlow.js-style `lossWeights` spelling remains an
alias for JavaScript and TypeScript callers, but serialized Keras training
configuration uses `loss_weights`.

The weighted total is computed from scalar-reduced output losses, so outputs do
not need to have mutually broadcastable loss tensor shapes.

### Keras-compatible sample and class weights

This fork also supports Keras-compatible `sample_weight` and `class_weight`
training semantics. These solve a different problem from `loss_weights`:
`loss_weights` scales whole model outputs, while `sample_weight` scales elements
of an output's unreduced loss before that output is reduced.

For a single-output model, pass one sample-weight tensor. For a multi-output
model, pass one tensor per output as an array or output-name dictionary. A
single rank-1 (or `[batch, 1]`) sample-wise tensor can be shared by all outputs.
Non-scalar structural weights must match the output structure, for example:

```js
await model.fit(xs, {
  category: categoryTargets,
  geometry: geometryTargets
}, {
  sample_weight: {
    category: tf.tensor2d(categoryWeights, [batch, groups]),
    geometry: tf.tensor3d(geometryWeights, [batch, groups, elements])
  },
  epochs: 10
});
```

The corresponding unreduced losses may have shapes such as `[batch, groups]`
or `[batch, groups, elements]`. TensorFlow broadcasting is used when the
sample-weight tensor is multiplied into the unreduced loss. The result is then
reduced and any configured `loss_weights` value is applied to the resulting
per-output scalar.

The canonical Keras batch entry points are available:

```js
await model.train_on_batch(
  xs,
  ys,
  sampleWeight,
  null,  // class_weight
  true   // return_dict
);

await model.test_on_batch(
  xs,
  ys,
  sampleWeight,
  true   // return_dict
);
```

The TensorFlow.js camelCase forms `trainOnBatch()` and `testOnBatch()` route
through the same implementation. `predict_on_batch()` is also provided as an
alias of `predictOnBatch()`.

`fit()` accepts both the canonical `sample_weight` / `class_weight` spellings
and the existing TensorFlow.js `sampleWeight` / `classWeight` spellings.
Specifying both spellings for the same option is an error. Supplying
`sample_weight` and `class_weight` at the same time is also an error, matching
Keras.

Current Keras restricts `class_weight` to single-output models. Missing class
indices use the canonical default weight `1.0`. Use `sample_weight` for
multi-output, temporal or structural weighting. Validation sample weights are
supported both through `(valX, valY, valSampleWeight)` and when
`validationSplit` slices weighted tensor data.

Sample-weight tensors supplied by the caller remain caller-owned and are not
disposed by `fit()`, `train_on_batch()` or `test_on_batch()`.

### Loading a pretrained Keras model

You can also load a model previously trained and saved from elsewhere (e.g.,
from Python Keras) and use it for inference or transfer learning in the browser.

For example, in Python, save your Keras model using
[tensorflowjs](https://pypi.org/project/tensorflowjs/),
which can be installed using `pip install tensorflowjs`.


```python
import tensorflowjs as tfjs

# ... Create and train your Keras model.

# Save your Keras model in TensorFlow.js format.
tfjs.converters.save_keras_model(model, '/path/to/tfjs_artifacts/')

# Then use your favorite web server to serve the directory at a URL, say
#   http://foo.bar/tfjs_artifacts/model.json
```

To load the model with TensorFlow.js Layers:

```js
import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel('http://foo.bar/tfjs_artifacts/model.json');
// Now the model is ready for inference, evaluation or re-training.
```

## For more information

- [TensorFlow.js API documentation](https://js.tensorflow.org/api/latest/)
- [TensorFlow.js Tutorials](https://js.tensorflow.org/tutorials/)
