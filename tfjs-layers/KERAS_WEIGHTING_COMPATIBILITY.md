# Keras weighting compatibility contract

This fork treats the Keras training API as the behavioural reference for model
weighting features that are missing or incomplete in upstream TensorFlow.js
Layers.

The implementation in this branch was written against **Keras 3.15.0**,
particularly:

- `keras/src/backend/tensorflow/trainer.py`
- `keras/src/trainers/data_adapters/array_data_adapter.py`
- `keras/src/trainers/data_adapters/data_adapter_utils.py`

No consuming application's model layout, label names or weighting policy belongs
in TensorFlow.js Layers. Consumers construct ordinary weight tensors and pass
them through the public Keras-compatible API.

## Distinct weighting levels

`loss_weights` and `sample_weight` intentionally operate at different levels:

1. The output loss function produces an unreduced loss tensor.
2. `sample_weight` is multiplied into that unreduced tensor, with normal
   TensorFlow broadcasting and sample-wise expansion across trailing structural
   dimensions.
3. The weighted tensor is reduced to the output's scalar loss.
4. `loss_weights` scales that scalar output loss.
5. Weighted output losses and regularisation losses form the optimisation loss.

Per-output loss metrics include `sample_weight`, because it is part of that
output's loss, but do not include `loss_weights`, matching Keras' distinction
between per-output losses and the final optimisation weighting.

Ordinary metrics remain unweighted. Keras reserves weighted metric behaviour for
`weighted_metrics`, which is outside this compatibility increment.

## `sample_weight`

A single-output model accepts one Tensor.

A multi-output model accepts one Tensor per output as an array or output-name
dictionary. A single rank-1 Tensor, or a rank-2 Tensor shaped `[batch, 1]`, is a
sample-wise weight and may be shared across all outputs.

Non-scalar structural weights must be supplied per output. Examples include:

- `[batch, timesteps]`
- `[batch, groups]`
- `[batch, groups, elements]`

If a sample-wise tensor has fewer dimensions than an unreduced loss, singleton
axes are appended so the batch dimension stays the leading dimension during
broadcasting.

Caller-provided sample-weight tensors remain caller-owned. The training API
clones standardized weights before training and disposes only its own clones.

## `class_weight`

Current Keras supports `class_weight` only for a model with one output. The
public compatibility API follows that rule.

Class weights are converted to one scalar sample weight per batch item. For
one-hot targets the class is selected with `argmax`; single-column class-index
targets are squeezed. An omitted class index receives the Keras default weight
`1.0`.

Legacy TensorFlow.js `ClassWeight[]` and `ClassWeightMap` normalization helpers
remain exported so existing source that imports those utilities does not break.
They do not change the public Keras-compatible restriction on multi-output
`class_weight`.

`sample_weight` and `class_weight` are mutually exclusive in one training call.

## Batch methods

Canonical Keras spellings are provided:

```text
train_on_batch(x, y, sample_weight=None, class_weight=None, return_dict=False)
test_on_batch(x, y, sample_weight=None, return_dict=False)
predict_on_batch(x)
```

TensorFlow.js camelCase forms are retained and use the same implementation:

```text
trainOnBatch(...)
testOnBatch(...)
predictOnBatch(...)
```

When `return_dict` / `returnDict` is true, batch loss/metric values are keyed by
the model's deduplicated metric names. Otherwise the historical TensorFlow.js
single-value-or-array return convention is preserved.

## `fit`

`fit()` accepts both Keras snake-case and TensorFlow.js camelCase spellings for
the new weighting options:

```text
sample_weight / sampleWeight
class_weight  / classWeight
```

Specifying both spellings of the same option is an error.

Weighted validation is supported through a three-item validation tuple
`(valX, valY, valSampleWeight)`. `validationSplit` slices training sample-weight
tensors at the same boundary as inputs and targets.

## Serialization

`sample_weight` and `class_weight` are call-time data, not persistent model
configuration, so they are not serialized.

`loss_weights` remains serialized in training configuration using the canonical
snake-case key.

## Compatibility policy

When Keras and historical TensorFlow.js naming differ, the canonical Keras name
is added without removing the existing TensorFlow.js name where preserving the
alias is practical.

Behavioural tests should prefer invariant Keras semantics over details of a
specific application. Any future extension should first be expressible as a
general Keras/TensorFlow training capability before it is added to this fork.
