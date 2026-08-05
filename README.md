# TensorFlow TypeScript

A focused, reusable TensorFlow.js fork for training and deploying machine-learning models in Node.js and web browsers.

This fork retains the general-purpose TensorFlow.js packages needed for TypeScript model development, native Node.js execution and browser inference, while removing unrelated products and obsolete repository infrastructure.

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
