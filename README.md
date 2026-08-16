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

## Pretrained model runtime

The retained CPU runtime is continuously verified by building `tfjs-core`, `tfjs-backend-cpu` and `tfjs-converter` from this repository, then loading and executing an official pretrained TensorFlow.js GraphModel.

Converter operation mappings are kept as checked-in TypeScript snapshots generated from the retained JSON metadata. `scripts/check-source-policy.mjs` verifies that those snapshots are current, so pretrained-model support does not depend on the removed Python converter tooling.

## Removed scope

The fork does not maintain React Native applications, TensorFlow Lite or Decision Forest bindings, AutoML helpers, visualisation packages, Python model-conversion tooling, obsolete experimental backends, cloud release infrastructure, broad upstream demos or platform-specific publishing machinery outside the retained Node.js and browser runtimes.

## Licence

Apache License 2.0. Original TensorFlow.js copyright and licence notices are retained.