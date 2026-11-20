# Incremental Code Guards checks on KittyWin

TensorFlow.js pins Code Guards at `vendor/code-guards`. The built-in
preflight GitHub Action is **manual-only**, and normal source CI does not run
neural inference. GPU scans happen locally on KittyWin using both RTX 2080 Ti
adapters; completed MobileBERT evidence-window scores are persisted in a
SQLite WAL database. Only changed or previously unseen windows require
new GPU inference, while the current project's full cross-file candidate index
is still evaluated on each scan.

## Windows PowerShell

Run these commands from the TensorFlow.js checkout:

```powershell
git pull --ff-only origin master
git submodule update --init --recursive
$env:CODE_GUARDS_WEBGPU_PYTHON = "$env:TEMP\code-guards-webgpu\Scripts\python.exe"
bun vendor/code-guards/src/cli.ts all --project tsconfig.code-guards.json --strict --include-tests --progress --accelerate batch,parallel,webgpu --gpu-ids 0,1 --concurrency 4 --batch-size 4 --inference-cache-dir "$env:TEMP\code-guards-inference" --cache-remote https://github.com/kitty-crow/tensorflow-ts.git --sync-inference-cache --format json > "$env:TEMP\tensorflow-ts-findings.json"
```

The `--sync-inference-cache` flag fetches existing inference results before
scanning and, on normal exit, publishes the merged cache to the remote
`inference-cache` branch under `inference/`. Ensure Git is authenticated for
pushes. Cache updates never alter `master` and never trigger TensorFlow.js
CI. The cache contains hashes and model scores, not private source text.

The cache is checkpointed after every GPU batch. After `Ctrl+C` or a crash,
restart the same command to reuse locally completed windows. If the remote
publish fails, local results remain intact and can be synchronised during
the next run. Results are invalidated by changed token windows, hypothesis
labels, model weights, tokenizer or inference scoring version. Model startup,
TypeScript candidate discovery and deterministic checks still run each time.

For a CPU-only or GitHub preflight workload estimate, invoke the manual
`Inference Workload Preflight` action. It is not a neural scan.

The cache exchange uses Git and Python; it does not depend on GitHub's
slower hosted inference runners.
