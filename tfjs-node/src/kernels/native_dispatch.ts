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

import {backend_util, DataType, Dilation2DAttrs, KernelAttributes, KernelConfig, ResizeBilinearAttrs, ResizeBilinearInputs, scalar, Tensor, TensorInfo, tensor1d, util} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';
import {TFEOpAttr} from '../tfjs_binding';

/** Native unary kernels with the standard x input and T dtype attribute. */
export function createUnaryKernelConfig(kernelName: string): KernelConfig {
  return {
    kernelName,
    backendName: 'tensorflow',
    kernelFunc: args => {
      const {x} = args.inputs as {x: TensorInfo};
      const backend = args.backend as NodeJSKernelBackend;
      return backend.executeSingleInput(kernelName, x);
    }
  };
}

/** Fourier transforms use the input key and the native Tcomplex attribute. */
export function createComplexTransformConfig(kernelName: 'FFT'|'IFFT'): KernelConfig {
  return {
    kernelName,
    backendName: 'tensorflow',
    kernelFunc: args => {
      const {input} = args.inputs as {input: TensorInfo};
      const backend = args.backend as NodeJSKernelBackend;
      return backend.executeSingleOutput(
          kernelName, [createTensorsTypeOpAttr('Tcomplex', input.dtype)], [input]);
    }
  };
}

/** Complex magnitude always has a float32 output, including the Abs adapter. */
export function executeComplexAbs(backend: NodeJSKernelBackend, x: TensorInfo): TensorInfo {
  return backend.executeSingleOutput('ComplexAbs', [
    createTensorsTypeOpAttr('T', x.dtype),
    createTensorsTypeOpAttr('Tout', 'float32')
  ], [x]);
}

/** Owns the temporary int32 vector passed as the final native operand. */
export function executeWithInt32Vector(
    backend: NodeJSKernelBackend, name: string, attrs: TFEOpAttr[],
    inputs: TensorInfo[], values: number[]): TensorInfo {
  const vector = tensor1d(values, 'int32');
  try {
    return backend.executeSingleOutput(name, attrs, [...inputs, vector]);
  } finally {
    vector.dispose();
  }
}

/** Forward and gradient LRN share the exact native attribute layout. */
export function localResponseAttrs(
    backend: NodeJSKernelBackend, dtype: DataType,
    {depthRadius, bias, alpha, beta}: {
      depthRadius: number, bias: number, alpha: number, beta: number
    }): TFEOpAttr[] {
  return [
    createTensorsTypeOpAttr('T', dtype),
    {name: 'depth_radius', type: backend.binding.TF_ATTR_INT, value: depthRadius},
    {name: 'bias', type: backend.binding.TF_ATTR_FLOAT, value: bias},
    {name: 'alpha', type: backend.binding.TF_ATTR_FLOAT, value: alpha},
    {name: 'beta', type: backend.binding.TF_ATTR_FLOAT, value: beta}
  ];
}

/** Native pool attributes share binding types; spatial rank determines layout. */
function poolAttributes(
    backend: NodeJSKernelBackend, dtype: DataType, padding: string,
    {ksize, strides, dataFormat, paddingLabel}: {
      ksize: number[], strides: number[], dataFormat: string,
      paddingLabel: 'padding'|'padding type'
    }): TFEOpAttr[] {
  if (padding !== 'VALID' && padding !== 'SAME') {
    throw new Error(
        `TF Backend supports only 'valid' and 'same' padding ` +
        `while ${paddingLabel} was ${padding}`);
  }
  return [
    createTensorsTypeOpAttr('T', dtype),
    {name: 'ksize', type: backend.binding.TF_ATTR_INT, value: ksize},
    {name: 'strides', type: backend.binding.TF_ATTR_INT, value: strides},
    {name: 'padding', type: backend.binding.TF_ATTR_STRING, value: padding},
    {name: 'data_format', type: backend.binding.TF_ATTR_STRING, value: dataFormat}
  ];
}

export function pool2DAttrs(
    backend: NodeJSKernelBackend, dtype: DataType,
    info: backend_util.Conv2DInfo, paddingLabel: 'padding'|'padding type'): TFEOpAttr[] {
  return poolAttributes(backend, dtype, info.padInfo.type, {
    ksize: [1, info.filterHeight, info.filterWidth, 1],
    strides: [1, info.strideHeight, info.strideWidth, 1],
    dataFormat: info.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW',
    paddingLabel
  });
}

export function pool3DAttrs(
    backend: NodeJSKernelBackend, dtype: DataType,
    info: backend_util.Conv3DInfo, paddingLabel: 'padding'|'padding type'): TFEOpAttr[] {
  return poolAttributes(backend, dtype, info.padInfo.type, {
    ksize: [1, info.filterDepth, info.filterHeight, info.filterWidth, 1],
    strides: [1, info.strideDepth, info.strideHeight, info.strideWidth, 1],
    dataFormat: info.dataFormat === 'channelsLast' ? 'NDHWC' : 'NCDHW',
    paddingLabel
  });
}

/** All and Any share the boolean reduction contract without a T attribute. */
export function createBooleanReductionConfig(kernelName: 'All'|'Any'): KernelConfig {
  return {
    kernelName,
    backendName: 'tensorflow',
    kernelFunc: (args: {
      inputs: {x: TensorInfo}, backend: NodeJSKernelBackend,
      attrs: {axis: number|number[], keepDims: boolean}
    }) => {
      const {x} = args.inputs;
      const {axis, keepDims} = args.attrs;
      const backend = args.backend;
      const axes = util.parseAxisParam(axis, x.shape);
      const attrs = [
        {name: 'keep_dims', type: backend.binding.TF_ATTR_BOOL, value: keepDims},
        createTensorsTypeOpAttr('Tidx', 'int32')
      ];
      return executeWithInt32Vector(backend, kernelName, attrs, [x], axes);
    }
  };
}

/** Numeric reductions share axis parsing and native dtype attributes. */
export function executeNumericReduction(
    backend: NodeJSKernelBackend, kernelName: 'Sum'|'Max'|'Min'|'Prod',
    x: TensorInfo, axis: number|number[], keepDims: boolean): TensorInfo {
  const axes = util.parseAxisParam(axis, x.shape);
  return executeWithInt32Vector(
      backend, kernelName, backend.createReductionOpAttrs(x, keepDims), [x], axes);
}

/** ArgMin and ArgMax share casting, scalar-axis and temporary ownership. */
export function createArgExtremaConfig(kernelName: 'ArgMin'|'ArgMax'): KernelConfig {
  return {
    kernelName,
    backendName: 'tensorflow',
    kernelFunc: (args: {
      inputs: {x: TensorInfo}, backend: NodeJSKernelBackend, attrs: {axis: number}
    }) => {
      const {x} = args.inputs;
      const backend = args.backend;
      const toDispose: Tensor[] = [];
      let xInput = x;
      try {
        if (x.dtype === 'bool') {
          xInput = (x as Tensor).toInt();
          toDispose.push(xInput as Tensor);
        }
        const axisScalar = scalar(args.attrs.axis, 'int32');
        toDispose.push(axisScalar);
        const attrs = [
          createTensorsTypeOpAttr('T', xInput.dtype),
          createTensorsTypeOpAttr('Tidx', 'int32'),
          createTensorsTypeOpAttr('output_type', 'int32')
        ];
        return backend.executeSingleOutput(kernelName, attrs, [xInput, axisScalar]);
      } finally {
        toDispose.forEach(t => t.dispose());
      }
    }
  };
}

/** Both resize algorithms use the same native size and coordinate attributes. */
export function createResizeKernelConfig(
    kernelName: 'ResizeBilinear'|'ResizeNearestNeighbor'): KernelConfig {
  return {
    kernelName,
    backendName: 'tensorflow',
    kernelFunc: (args: {
      inputs: ResizeBilinearInputs, backend: NodeJSKernelBackend,
      attrs: KernelAttributes<ResizeBilinearAttrs>
    }) => {
      const {images} = args.inputs;
      const backend = args.backend;
      const {alignCorners, halfPixelCenters, size} = args.attrs;
      const attrs = [
        createTensorsTypeOpAttr('T', images.dtype),
        {name: 'align_corners', type: backend.binding.TF_ATTR_BOOL, value: alignCorners},
        {name: 'half_pixel_centers', type: backend.binding.TF_ATTR_BOOL, value: halfPixelCenters}
      ];
      const [newHeight, newWidth] = size;
      return executeWithInt32Vector(backend, kernelName, attrs, [images], [newHeight, newWidth]);
    }
  };
}

/** Gather and scatter native ops use int32 indices with a named value dtype. */
export function indexedTensorAttrs(valueAttr: 'T'|'Tparams', dtype: DataType): TFEOpAttr[] {
  return [
    createTensorsTypeOpAttr(valueAttr, dtype),
    createTensorsTypeOpAttr('Tindices', 'int32')
  ];
}

/** The NMS versions share the scalar operands, including their exact dtypes. */
export function withNmsInputs<T>(
    boxes: TensorInfo, scores: TensorInfo,
    {maxOutputSize, iouThreshold, scoreThreshold}: {
      maxOutputSize: number, iouThreshold: number, scoreThreshold: number
    }, execute: (inputs: [TensorInfo, TensorInfo, ...Tensor[]]) => T,
    extraThresholds: number[] = []): T {
  const owned: Tensor[] = [];
  try {
    owned.push(scalar(maxOutputSize, 'int32'));
    owned.push(scalar(iouThreshold, 'float32'));
    owned.push(scalar(scoreThreshold, 'float32'));
    for (const threshold of extraThresholds) {
      owned.push(scalar(threshold, 'float32'));
    }
    return execute([boxes, scores, ...owned]);
  } finally {
    owned.forEach(tensor => tensor.dispose());
  }
}

/** Dilation forward and both adjoints use the same NHWC native attributes. */
export function dilation2DAttrs(
    backend: NodeJSKernelBackend, x: TensorInfo, filter: TensorInfo,
    {strides, pad, dilations}: Dilation2DAttrs): TFEOpAttr[] {
  const {dilationHeight, dilationWidth, padInfo, strideHeight, strideWidth} =
      backend_util.computeDilation2DInfo(
          x.shape as [number, number, number, number],
          filter.shape as [number, number, number], strides, pad, 'NHWC',
          dilations);
  return [
    createTensorsTypeOpAttr('T', x.dtype),
    {name: 'strides', type: backend.binding.TF_ATTR_INT,
      value: [1, strideHeight, strideWidth, 1]},
    {name: 'rates', type: backend.binding.TF_ATTR_INT,
      value: [1, dilationHeight, dilationWidth, 1]},
    {name: 'padding', type: backend.binding.TF_ATTR_STRING, value: padInfo.type}
  ];
}

/** Conv3D and its adjoints agree on layout, stride, padding and dilation. */
export function conv3DAttrs(
    backend: NodeJSKernelBackend, dtype: DataType,
    info: backend_util.Conv3DInfo): TFEOpAttr[] {
  if (!backend.isGPUPackage && info.dilationDepth > 1) {
    throw new Error('CPU Dilation depth must be 1');
  }
  return [
    createTensorsTypeOpAttr('T', dtype),
    {name: 'strides', type: backend.binding.TF_ATTR_INT,
      value: [1, info.strideDepth, info.strideHeight, info.strideWidth, 1]},
    {name: 'padding', type: backend.binding.TF_ATTR_STRING, value: info.padInfo.type},
    {name: 'data_format', type: backend.binding.TF_ATTR_STRING,
      value: info.dataFormat === 'channelsLast' ? 'NDHWC' : 'NCDHW'},
    {name: 'dilations', type: backend.binding.TF_ATTR_INT,
      value: [1, info.dilationDepth, info.dilationHeight, info.dilationWidth, 1]}
  ];
}

/** Native shape/perm/multiples vectors all use int32; the attr name is distinct. */
export function executeTypedInt32Vector(
    backend: NodeJSKernelBackend, name: string, x: TensorInfo,
    vectorAttr: 'Tshape'|'Tperm'|'Tmultiples', values: number[]): TensorInfo {
  return executeWithInt32Vector(backend, name, [
    createTensorsTypeOpAttr('T', x.dtype),
    createTensorsTypeOpAttr(vectorAttr, 'int32')
  ], [x], values);
}

/** Both resize adjoints use the same dtype and align-corners flag. */
export function resizeGradientAttrs(
    backend: NodeJSKernelBackend, dtype: DataType, alignCorners: boolean): TFEOpAttr[] {
  return [
    createTensorsTypeOpAttr('T', dtype),
    {name: 'align_corners', type: backend.binding.TF_ATTR_BOOL, value: alignCorners}
  ];
}

/** Native logical binary ops have no explicit dtype attributes. */
export function createLogicalBinaryConfig(kernelName: 'LogicalAnd'|'LogicalOr'): KernelConfig {
  return {
    kernelName,
    backendName: 'tensorflow',
    kernelFunc: (args: {inputs: {a: TensorInfo, b: TensorInfo}, backend: NodeJSKernelBackend}) => {
      const {a, b} = args.inputs;
      return args.backend.executeSingleOutput(kernelName, [], [a, b]);
    }
  };
}

/** Like-shaped constant ops use the selected backend's dtype mapping. */
export function createLikeKernelConfig(kernelName: 'OnesLike'|'ZerosLike'): KernelConfig {
  return {
    kernelName,
    backendName: 'tensorflow',
    kernelFunc: (args: {inputs: {x: TensorInfo}, backend: NodeJSKernelBackend}) => {
      const {x} = args.inputs;
      const backend = args.backend;
      const attrs = [{
        name: 'T',
        type: backend.binding.TF_ATTR_TYPE,
        value: backend.getDTypeInteger(x.dtype)
      }];
      return backend.executeSingleOutput(kernelName, attrs, [x]);
    }
  };
}
