/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import {ENGINE} from '../engine';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {makeTypesMatch} from '../tensor_util';
import {convertToTensor} from '../tensor_util_env';
import {DataType, TensorLike} from '../types';
import {isPromise} from '../util';

import {assertAndGetBroadcastShape} from './broadcast_util';

export const OP_SCOPE_SUFFIX = '__op';

/**
 * Used for wrapping functions that perform math operations on
 * Tensors. The function will be wrapped in a named scope that cleans all
 * memory usage after the function is done.
 */
export function op<T extends Function>(f: {[name: string]: T}): T {
  const keys = Object.keys(f);
  if (keys.length !== 1) {
    throw new Error(
        `Please provide an object with a single key ` +
        `(operation name) mapping to a function. Got an object with ` +
        `${keys.length} keys.`);
  }

  let opName = keys[0];
  const fn = f[opName];

  // Strip the underscore from the end of the function name.
  if (opName.endsWith('_')) {
    opName = opName.substring(0, opName.length - 1);
  }

  // add an __op suffix to distinguish ops from kernels in tf.profile
  opName = opName + OP_SCOPE_SUFFIX;

  // tslint:disable-next-line:no-any
  const f2 = (...args: any[]) => {
    ENGINE.startScope(opName);
    try {
      const result = fn(...args);
      if (isPromise(result)) {
        console.error('Cannot return a Promise inside of tidy.');
      }
      ENGINE.endScope(result);
      return result;
    } catch (ex) {
      ENGINE.endScope(null);
      throw ex;
    }
  };
  Object.defineProperty(f2, 'name', {value: opName, configurable: true});

  // tslint:disable-next-line:no-any
  return f2 as any as T;
}

/** Converts a unary operand before dispatching its specific kernel. */
export function runUnaryKernel<T extends Tensor>(
    x: T|TensorLike, operationName: string, kernelName: string,
    dtype: DataType|'numeric'|'string_or_numeric' = 'numeric',
    inputName: 'x'|'input' = 'x'): T {
  const input = convertToTensor(x, inputName, operationName, dtype);
  const inputs: NamedTensorMap = {[inputName]: input};
  return ENGINE.runKernel<T>(kernelName, inputs);
}

/** Applies the common operand conversion and dtype matching contract. */
export function matchedBinaryInputs(
    a: Tensor|TensorLike, b: Tensor|TensorLike, operationName: string,
    dtype: 'numeric'|'string_or_numeric' = 'numeric'): {a: Tensor, b: Tensor} {
  const left = convertToTensor(a, 'a', operationName, dtype);
  const right = convertToTensor(b, 'b', operationName, dtype);
  const [matchedA, matchedB] = makeTypesMatch(left, right);
  return {a: matchedA, b: matchedB};
}

/** Dispatches a numeric binary operation after matching operand types. */
export function runMatchedBinaryKernel<T extends Tensor>(
    a: Tensor|TensorLike, b: Tensor|TensorLike, operationName: string,
    kernelName: string): T {
  return ENGINE.runKernel<T>(
      kernelName, matchedBinaryInputs(a, b, operationName));
}

/** Comparisons accept strings and validate broadcast shapes before dispatch. */
export function runComparisonKernel<T extends Tensor>(
    a: Tensor|TensorLike, b: Tensor|TensorLike, operationName: string,
    kernelName: string): T {
  const inputs = matchedBinaryInputs(a, b, operationName, 'string_or_numeric');
  assertAndGetBroadcastShape(inputs.a.shape, inputs.b.shape);
  return ENGINE.runKernel<T>(kernelName, inputs);
}

/** Boolean reductions share input validation and axis/keepDims forwarding. */
export function runBooleanReduction<T extends Tensor>(
    x: Tensor|TensorLike, axis: number|number[], keepDims: boolean,
    operationName: string, kernelName: string): T {
  const input = convertToTensor(x, 'x', operationName, 'bool');
  return ENGINE.runKernel<T>(kernelName, {x: input}, {axis, keepDims});
}
