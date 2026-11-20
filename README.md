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

The [Code Guards](https://github.com/kitty-crow/code-guards) analyser is a pinned Git submodule at `vendor/code-guards`. It checks the TypeScript sources and tests across the maintained TensorFlow packages. Bun installs, compiles and executes Code Guards. CI runs every available guard in strict mode: **both warnings and errors fail the check**. The analyser is independent of TensorFlow's source and does not modify it.

To reproduce the CI scan locally:

```sh
git submodule update --init --recursive
bun install --frozen-lockfile --ignore-scripts
(cd vendor/code-guards && bun install --ignore-scripts && bun run build)
bun run guards:strict
```

The scan uses `tsconfig.code-guards.json` to include sources omitted by the upstream root TypeScript configuration, including Node, WebGL and WebGPU modules. Existing violations must be fixed in source; no guards are disabled. Generated output, installed dependencies and the vendored analyser are excluded from the target scan.

The Code Guards and source-policy CI steps run on Bun. The remaining legacy TensorFlow build, native addon and Bazel paths still use their original Yarn/Node.js tooling where compatibility has not been verified. The root `build` and `test` scripts invoke the Bun source-policy check before handing off to those existing paths.

For live granularity during the strict scan, use `bun run guards:strict --progress`. The scanner reports each stage, elapsed time, findings and MobileBERT inference windows to standard error; GitHub CI displays these messages while retaining GitHub-format diagnostics. Code Guards excludes vendored source and documentation, including the pinned submodules listed in `.gitmodules`.

## Temporary nested-conditional validation

The CI workflow currently runs only the strict `no-nested-if` Code Guards rule while neural analysis is being optimised upstream. Source files and tests remain in scope; vendored dependencies are excluded. Run the same local check with `bun run guards:nested --format github --progress`. The existing `bun run guards:strict` command remains available for the complete 23-rule scan. Restore full CI enforcement when Code Guards performance is suitable for this repository.

The temporary nested-conditional CI also publishes an unmodified `nested-conditional-inventory` JSON artifact, and prints the most affected source files and the first exact locations in the job log. The final strict scan still fails on any finding.

## Nested conditional refactoring baseline

The strict Code Guards `no-nested-if` CI guard reports **errors**, not advisory warnings. The current baseline after behaviour-preserving refactoring is **850 CG001 findings**. An AST-guided maintenance tool at `scripts/flatten-nested-conditionals.ts` supports `--apply` for narrowly justified transformations. The separate `Safe Nested Conditional Refactoring` workflow generates a patch artifact for review but **does not automatically push changes**. CI keeps the exact source location inventory available as an artifact. Remaining findings must be reviewed and refactored without changing TensorFlow's external APIs or runtime behaviour.
