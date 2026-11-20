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

## Strict source review

The [Code Guards](https://github.com/kitty-crow/code-guards) analyser is a pinned Git submodule at `vendor/code-guards`. It checks the TypeScript sources and tests across the maintained TensorFlow packages. CI runs every available guard in strict mode: **both warnings and errors fail the check**. The analyser is independent of TensorFlow's source and does not modify it.

To reproduce the CI scan locally:

```sh
git submodule update --init --recursive
npm install --prefix vendor/code-guards --no-audit --no-fund
npm run build --prefix vendor/code-guards
npm run guards:strict
```

The scan uses `tsconfig.code-guards.json` to include sources omitted by the upstream root TypeScript configuration, including Node, WebGL and WebGPU modules. Existing violations must be fixed in source; no guards are disabled. Generated output, installed dependencies and the vendored analyser are excluded from the target scan.
