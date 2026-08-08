# TensorFlow TypeScript

A focused, reusable TensorFlow.js fork for training and deploying machine-learning models in Node.js and web browsers.

This fork retains the general-purpose TensorFlow.js packages needed for TypeScript model development, native Node.js execution and browser inference, while removing unrelated products and obsolete repository infrastructure.

## Keras-compatible training

The retained Layers API tracks canonical Keras training semantics where the
upstream TensorFlow.js implementation is incomplete. Current fork extensions
include Keras-compatible multi-output `loss_weights` and weighted tensor
training through `sample_weight` / `class_weight`, including structural
multi-output sample weights and the canonical `train_on_batch()` and
`test_on_batch()` entry points. TensorFlow.js camelCase aliases remain available.

These features are generic framework capabilities. Application-specific loss
policies belong in consuming models rather than in this fork.

See [`tfjs-layers/README.md`](tfjs-layers/README.md) for the supported weighting
forms and examples.

## Retained packages

- `tfjs`
- `tfjs-core`
- `tfjs-layers`
- `tfjs-data`
- `tfjs-converter` runtime
- `tfjs-backend-cpu`
- `tfjs-backend-webgl`
- `tfjs-backend-webgpu`
- `tfjs-node`
- `tfjs-node-gpu`

## Removed scope

The fork does not maintain React Native applications, TensorFlow Lite or Decision Forest bindings, AutoML helpers, visualisation packages, Python model-conversion tooling, obsolete experimental backends, cloud release infrastructure, broad upstream demos or platform-specific publishing machinery outside the retained Node.js and browser runtimes.

## Licence

Apache License 2.0. Original TensorFlow.js copyright and licence notices are retained.
