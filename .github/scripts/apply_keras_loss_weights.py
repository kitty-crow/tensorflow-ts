from pathlib import Path

path = Path("tfjs-layers/src/engine/training.ts")
source = path.read_text()


def replace_once(old: str, new: str) -> None:
    global source
    count = source.count(old)
    if count != 1:
        raise SystemExit(
            f"expected exactly one match, found {count}: {old[:100]!r}"
        )
    source = source.replace(old, new, 1)


replace_once(
    "import {MetricsIdentifier, TrainingConfig} from '../keras_format/training_config';",
    "import {LossWeights, MetricsIdentifier, TrainingConfig} from '../keras_format/training_config';",
)

replace_once(
    """    return nestedMetrics;
  }
}

export interface ModelEvaluateArgs {""",
    """    return nestedMetrics;
  }
}

/**
 * Normalizes Keras `loss_weights` to model output order.
 *
 * Keras accepts a scalar for a single-output model, or one scalar per
 * output as a list/tuple or dictionary for a multi-output model.
 */
export function standardizeLossWeights(
    lossWeights: LossWeights, outputNames: string[]): number[] {
  if (lossWeights == null) {
    return outputNames.map(_ => 1);
  }

  if (typeof lossWeights === 'number') {
    if (outputNames.length !== 1) {
      throw new ValueError(
          '`loss_weights` may be a scalar only for a single-output model. ' +
          `Received ${outputNames.length} outputs.`);
    }
    return [lossWeights];
  }

  if (Array.isArray(lossWeights)) {
    if (lossWeights.length !== outputNames.length) {
      throw new ValueError(
          '`loss_weights` must match the number of model outputs. ' +
          `Expected ${outputNames.length} weight(s), received ` +
          `${lossWeights.length}.`);
    }
    for (const weight of lossWeights) {
      if (typeof weight !== 'number') {
        throw new ValueError(
            '`loss_weights` list entries must be numeric scalars.');
      }
    }
    return lossWeights.slice();
  }

  if (typeof lossWeights === 'object') {
    for (const name of Object.keys(lossWeights)) {
      if (outputNames.indexOf(name) === -1) {
        throw new ValueError(
            `Unknown output in \`loss_weights\` dictionary: \"${name}\". ` +
            `Expected: ${outputNames}`);
      }
    }
    const missing = outputNames.filter(name => lossWeights[name] == null);
    if (missing.length > 0) {
      throw new ValueError(
          `\`loss_weights\` dictionary is missing output(s): ${missing}.`);
    }
    return outputNames.map(name => {
      const weight = lossWeights[name];
      if (typeof weight !== 'number') {
        throw new ValueError(
            `\`loss_weights\` value for \"${name}\" must be a numeric scalar.`);
      }
      return weight;
    });
  }

  throw new ValueError(
      'Expected `loss_weights` to be a numeric scalar, list, or dictionary.');
}

export interface ModelEvaluateArgs {""",
)

replace_once(
    """  // TODO(cais): Add lossWeights, sampleWeightMode, weightedMetrics, and
  //   targetTensors.""",
    """  /**
   * Keras-compatible loss weights. A scalar is valid for a single output;
   * multi-output models accept an Array or output-name dictionary.
   */
  loss_weights?: LossWeights;

  /**
   * TensorFlow.js camelCase alias for `loss_weights`.
   */
  lossWeights?: LossWeights;

  // TODO(cais): Add sampleWeightMode, weightedMetrics, and targetTensors.""",
)

replace_once(
    """  lossFunctions: LossOrMetricFn[];

  // TODO(cais): These private variables should probably not have the string""",
    """  lossFunctions: LossOrMetricFn[];
  /** Exact Keras spelling retained for API and serialization parity. */
  loss_weights?: LossWeights;
  /** TensorFlow.js-style alias of `loss_weights`. */
  lossWeights?: LossWeights;
  private lossWeightValues: number[];

  // TODO(cais): These private variables should probably not have the string""",
)

replace_once(
    """    this.loss = args.loss;

    if (typeof args.optimizer === 'string') {""",
    """    this.loss = args.loss;

    if (args.loss_weights != null && args.lossWeights != null) {
      throw new ValueError(
          'Specify only one of `loss_weights` and `lossWeights`.');
    }
    const configuredLossWeights = args.loss_weights != null ?
        args.loss_weights : args.lossWeights;
    this.loss_weights = configuredLossWeights;
    this.lossWeights = configuredLossWeights;
    this.lossWeightValues =
        standardizeLossWeights(configuredLossWeights, this.outputNames);

    if (typeof args.optimizer === 'string') {""",
)

replace_once(
    """    // TODO(cais): Add lossWeights.
    // TODO(cais): Add sampleWeightMode.""",
    """    // TODO(cais): Add sampleWeightMode.""",
)

replace_once(
    """        // TODO(cais): Add weightedLoss, sampleWeight and mask.
        //   The following line should be weightedLoss
        const weightedLoss = this.lossFunctions[i];
        if (this.outputs.length > 1) {
          this.metricsTensors.push([weightedLoss, i]);""",
    """        // Keras reports per-output loss metrics before applying
        // `loss_weights`; only the total optimization loss is weighted.
        const outputLoss = this.lossFunctions[i];
        if (this.outputs.length > 1) {
          this.metricsTensors.push([outputLoss, i]);""",
)

replace_once(
    """        let totalLoss: Tensor;
        for (let i = 0; i < this.lossFunctions.length; ++i) {""",
    """        let totalLoss: Scalar;
        for (let i = 0; i < this.lossFunctions.length; ++i) {""",
)

replace_once(
    """          // TODO(cais): push Scalar instead.
          const meanLoss: Scalar = tfc.mean(loss);
          // TODO(cais): Use a scope() instead, to avoid ownership.
          lossValues.push(meanLoss);
          if (i === 0) {
            totalLoss = loss;
          } else {
            totalLoss = tfc.add(totalLoss, loss);
          }""",
    """          const meanLoss: Scalar = tfc.mean(loss);
          // Keep the raw per-output loss for Keras-compatible metrics.
          lossValues.push(meanLoss);

          const lossWeight = this.lossWeightValues[i];
          const weightedMeanLoss: Scalar = lossWeight === 1 ?
              meanLoss : tfc.mul(meanLoss, lossWeight) as Scalar;
          if (i === 0) {
            totalLoss = weightedMeanLoss;
          } else {
            totalLoss = tfc.add(totalLoss, weightedMeanLoss) as Scalar;
          }""",
)

replace_once(
    """        totalLoss = tfc.mean(totalLoss);

        // Add regularizer penalties.""",
    """        // Individual output losses are reduced before their weighted
        // sum, matching Keras and allowing heterogeneous output shapes.

        // Add regularizer penalties.""",
)

old_test = """  private makeTestFunction() {
    this.testFunction = (data: Tensor[]) => {
      return tfc.tidy(() => {
        const valOutputs: Scalar[] = [];
        let totalLoss: Scalar;
        const inputs = data.slice(0, this.inputs.length);
        const targets = data.slice(
            this.inputs.length, this.inputs.length + this.outputs.length);
        const feeds = [];
        for (let i = 0; i < this.inputs.length; ++i) {
          feeds.push({key: this.inputs[i], value: inputs[i]});
        }
        const feedDict = new FeedDict(feeds);
        const outputs = execute(this.outputs, feedDict) as Tensor[];
        // Compute total loss.
        for (let i = 0; i < this.lossFunctions.length; ++i) {
          const lossFunction = this.lossFunctions[i];
          // TODO(cais): Add sample weighting and replace the simple
          // averaging.
          const loss: Scalar = tfc.mean(lossFunction(targets[i], outputs[i]));
          if (i === 0) {
            totalLoss = loss;
          } else {
            totalLoss = tfc.add(totalLoss, loss);
          }
          valOutputs.push(totalLoss);
        }
        // Compute the metrics.
        for (let i = 0; i < this.metricsTensors.length; ++i) {
          const metric = this.metricsTensors[i][0];
          const outputIndex = this.metricsTensors[i][1];
          // TODO(cais): Replace K.mean() with a proper weighting function.
          const meanMetric =
              tfc.mean(metric(targets[outputIndex], outputs[outputIndex]));
          valOutputs.push(meanMetric as Scalar);
        }
        return valOutputs;
      });
    };
  }"""
new_test = """  private makeTestFunction() {
    this.testFunction = (data: Tensor[]) => {
      return tfc.tidy(() => {
        const valOutputs: Scalar[] = [];
        const lossValues: Scalar[] = [];
        let totalLoss: Scalar;
        const inputs = data.slice(0, this.inputs.length);
        const targets = data.slice(
            this.inputs.length, this.inputs.length + this.outputs.length);
        const feeds = [];
        for (let i = 0; i < this.inputs.length; ++i) {
          feeds.push({key: this.inputs[i], value: inputs[i]});
        }
        const feedDict = new FeedDict(feeds);
        const outputs = execute(this.outputs, feedDict) as Tensor[];

        // Compute raw per-output losses and the Keras-compatible weighted
        // total loss. Per-output loss metrics remain unweighted.
        for (let i = 0; i < this.lossFunctions.length; ++i) {
          const lossFunction = this.lossFunctions[i];
          // TODO(cais): Add sample weighting and replace the simple
          // averaging.
          const loss: Scalar =
              tfc.mean(lossFunction(targets[i], outputs[i]));
          lossValues.push(loss);
          const lossWeight = this.lossWeightValues[i];
          const weightedLoss: Scalar = lossWeight === 1 ?
              loss : tfc.mul(loss, lossWeight) as Scalar;
          if (i === 0) {
            totalLoss = weightedLoss;
          } else {
            totalLoss = tfc.add(totalLoss, weightedLoss) as Scalar;
          }
        }
        valOutputs.push(totalLoss);

        // Compute the metrics in the same order as makeTrainFunction().
        for (let i = 0; i < this.metricsTensors.length; ++i) {
          let meanMetric: Scalar;
          if (this.outputs.length > 1 && i < this.outputs.length) {
            meanMetric = lossValues[i];
          } else {
            const metric = this.metricsTensors[i][0];
            const outputIndex = this.metricsTensors[i][1];
            // TODO(cais): Replace K.mean() with a proper weighting function.
            meanMetric = tfc.mean(
                metric(targets[outputIndex], outputs[outputIndex]));
          }
          valOutputs.push(meanMetric);
        }
        return valOutputs;
      });
    };
  }"""
replace_once(old_test, new_test)

replace_once(
    """      metrics: this.getMetricIdentifiers(),
      optimizer_config: {""",
    """      metrics: this.getMetricIdentifiers(),
      loss_weights: this.loss_weights,
      optimizer_config: {""",
)

replace_once(
    """    // TODO(cais): Add weight_metrics when they are supported.
    // TODO(cais): Add sample_weight_mode when it's supported.
    // TODO(cais): Add loss_weights when it's supported.""",
    """    // TODO(cais): Add weight_metrics when they are supported.
    // TODO(cais): Add sample_weight_mode when it's supported.""",
)

replace_once(
    """    if (trainingConfig.loss_weights != null) {
      throw new Error('Loading loss_weights is not supported yet.');
    }
""",
    """,
)

replace_once(
    """    this.compile({loss, metrics, optimizer});""",
    """    this.compile({
      loss,
      metrics,
      optimizer,
      loss_weights: trainingConfig.loss_weights
    });""",
)

path.write_text(source)
