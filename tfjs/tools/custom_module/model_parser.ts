/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {io} from '@tensorflow/tfjs-core';
import * as fs from 'fs';
import {CustomTFJSBundleConfig} from './types';
import {bail} from './util';

export type KernelToOps = Record<string, string[]>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function requireRecord(value: unknown, description: string): Record<string, unknown> {
  if (!isRecord(value)) {
    throw new Error(`Expected an object for ${description}`);
  }
  return value;
}

function readOptionalArray(value: unknown, description: string): unknown[] {
  if (value == null) {
    return [];
  }
  if (!Array.isArray(value)) {
    throw new Error(`Expected an array for ${description}`);
  }
  return value;
}

/** Validate the mapping at the JSON boundary before using its operation names. */
export function parseKernelToOps(text: string): KernelToOps {
  const value: unknown = JSON.parse(text);
  const record = requireRecord(value, 'kernel to ops mapping');
  const mapping: KernelToOps = Object.create(null);
  for (const [kernel, ops] of Object.entries(record)) {
    if (!Array.isArray(ops) || !ops.every(op => typeof op === 'string')) {
      throw new Error(`Expected string operation names for kernel ${kernel}`);
    }
    mapping[kernel] = ops;
  }
  return mapping;
}

export function getOpsForConfig(
    config: CustomTFJSBundleConfig, kernelToOps: KernelToOps): string[] {
  const results = new Set<string>();
  for (const modelJsonPath of config.models ?? []) {
    let modelJson: Record<string, unknown>;
    try {
      const parsed: unknown = JSON.parse(fs.readFileSync(modelJsonPath, 'utf-8'));
      modelJson = requireRecord(parsed, 'model artifacts');
    } catch (e) {
      return bail(`Error loading JSON file ${modelJsonPath}`);
    }
    for (const op of getTopologyOps(modelJson.modelTopology, kernelToOps)) {
      results.add(op);
    }
  }
  return Array.from(results);
}

export function getOps(
    modelJson: io.ModelArtifacts, kernelToOp: KernelToOps): string[] {
  return getTopologyOps(modelJson.modelTopology, kernelToOp);
}

function getTopologyOps(topology: unknown, kernelToOp: KernelToOps): string[] {
  const results = new Set<string>();
  const graph = requireRecord(topology, 'model topology');

  function addNodes(nodes: unknown, description: string): void {
    for (const rawNode of readOptionalArray(nodes, description)) {
      const node = requireRecord(rawNode, 'graph node');
      const kernel = node.op;
      if (typeof kernel !== 'string') {
        throw new Error('Expected a string operation name for graph node');
      }
      const ops = kernelToOp[kernel] ?? null;
      if (ops === null) {
        const message = `Kernel => Op warning: could not find op mapping for kernel ${kernel}`;
        console.warn(message);
        throw new Error(message);
      }
      for (const op of ops) {
        results.add(op);
      }
    }
  }

  addNodes(graph.node, 'graph nodes');
  if (graph.library != null) {
    const library = requireRecord(graph.library, 'function library');
    for (const rawFunction of readOptionalArray(library.function, 'functions')) {
      const functionDef = requireRecord(rawFunction, 'function definition');
      addNodes(functionDef.nodeDef, 'function nodes');
    }
  }
  return Array.from(results);
}
