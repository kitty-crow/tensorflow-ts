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
// Allow UpperCamelCase variable names
// tslint:disable: variable-name
// Unfortunately just enabling PascalCase per file (tslint:enable:
// allow-pascal-case) doesn't work.
import {NamedTensorInfoMap} from './kernel_registry';
import {ExplicitPadding} from './ops/conv_util';
import {Activation} from './ops/fused_types';
import {TensorInfo} from './tensor_info';
import {DataType, DrawOptions, PixelData} from './types';

/**
 * Required, named tensor inputs for a concrete kernel. The general registry
 * map may contain absent values, but required input slots may not.
 * Keep genuinely optional slots explicit in their individual contracts.
 */
export type RequiredKernelInputs<K extends string> = Record<K, TensorInfo>;

export const Abs = 'Abs';
export type AbsInputs = UnaryInputs;

export const Acos = 'Acos';
export type AcosInputs = UnaryInputs;

export const Acosh = 'Acosh';
export type AcoshInputs = UnaryInputs;

export const Add = 'Add';
export type AddInputs = BinaryInputs;

export const AddN = 'AddN';
export type AddNInputs = TensorInfo[];

export const All = 'All';
export type AllInputs = RequiredKernelInputs<'x'>;
export interface AllAttrs {
  axis: number|number[];
  keepDims: boolean;
}

export const Any = 'Any';
export type AnyInputs = RequiredKernelInputs<'x'>;
export interface AnyAttrs {
  axis: number|number[];
  keepDims: boolean;
}

export const ArgMax = 'ArgMax';
export type ArgMaxInputs = RequiredKernelInputs<'x'>;
export interface ArgMaxAttrs {
  axis: number;
}

export const ArgMin = 'ArgMin';
export type ArgMinInputs = RequiredKernelInputs<'x'>;
export interface ArgMinAttrs {
  axis: number;
}

export const Asin = 'Asin';
export type AsinInputs = UnaryInputs;

export const Asinh = 'Asinh';
export type AsinhInputs = UnaryInputs;

export const Atan = 'Atan';
export type AtanInputs = UnaryInputs;

export const Atanh = 'Atanh';
export type AtanhInputs = UnaryInputs;

export const Atan2 = 'Atan2';
export type Atan2Inputs = BinaryInputs;

export const AvgPool = 'AvgPool';
export type AvgPoolInputs = RequiredKernelInputs<'x'>;
export interface AvgPoolAttrs {
  filterSize: [number, number]|number;
  strides: [number, number]|number;
  pad: 'valid'|'same'|number|ExplicitPadding;
  // CG305: The public convolution/pooling operation permits no explicit rounding override; omission is part of the kernel attribute contract.
  //@cg-expect-error
  dimRoundingMode?: 'floor'|'round'|'ceil';
}

export const AvgPoolGrad = 'AvgPoolGrad';
export type AvgPoolGradInputs = RequiredKernelInputs<'dy'|'input'>;
export interface AvgPoolGradAttrs {
  filterSize: [number, number]|number;
  strides: [number, number]|number;
  pad: 'valid'|'same'|number|ExplicitPadding;
}

export const AvgPool3D = 'AvgPool3D';
export type AvgPool3DInputs = RequiredKernelInputs<'x'>;
export interface AvgPool3DAttrs {
  filterSize: [number, number, number]|number;
  strides: [number, number, number]|number;
  pad: 'valid'|'same'|number;
  // CG305: The public convolution/pooling operation permits no explicit rounding override; omission is part of the kernel attribute contract.
  //@cg-expect-error
  dimRoundingMode?: 'floor'|'round'|'ceil';
  dataFormat: 'NDHWC'|'NCDHW';
}

export const AvgPool3DGrad = 'AvgPool3DGrad';
export type AvgPool3DGradInputs = RequiredKernelInputs<'dy'|'input'>;
export interface AvgPool3DGradAttrs {
  filterSize: [number, number, number]|number;
  strides: [number, number, number]|number;
  pad: 'valid'|'same'|number;
  // CG305: The public convolution/pooling operation permits no explicit rounding override; omission is part of the kernel attribute contract.
  //@cg-expect-error
  dimRoundingMode?: 'floor'|'round'|'ceil';
}

export const BatchMatMul = 'BatchMatMul';
export type BatchMatMulInputs = RequiredKernelInputs<'a'|'b'>;
export interface BatchMatMulAttrs {
  transposeA: boolean;
  transposeB: boolean;
}

export const BatchToSpaceND = 'BatchToSpaceND';
export type BatchToSpaceNDInputs = RequiredKernelInputs<'x'>;
export interface BatchToSpaceNDAttrs {
  blockShape: number[];
  crops: number[][];
}

export type BinaryInputs = RequiredKernelInputs<'a'|'b'>;

export const Bincount = 'Bincount';
export type BincountInputs = RequiredKernelInputs<'x'|'weights'>;
export interface BincountAttrs {
  size: number;
}

export const BitwiseAnd = 'BitwiseAnd';
export type BitwiseAndInputs = BinaryInputs;

export const BroadcastTo = 'BroadcastTo';
export type BroadcastToInputs = RequiredKernelInputs<'x'>;
export interface BroadCastToAttrs {
  shape: number[];
  inputShape: number[];  // for gradient
}

export const BroadcastArgs = 'BroadcastArgs';
export type BroadcastArgsInputs = RequiredKernelInputs<'s0'|'s1'>;

export const Cast = 'Cast';
export type CastInputs = UnaryInputs;
export interface CastAttrs {
  dtype: DataType;
}

export const Ceil = 'Ceil';
export type CeilInputs = UnaryInputs;

export const ClipByValue = 'ClipByValue';
export type ClipByValueInputs = UnaryInputs;
export interface ClipByValueAttrs {
  clipValueMin: number;
  clipValueMax: number;
}

export const Complex = 'Complex';
export type ComplexInputs = RequiredKernelInputs<'real'|'imag'>;

export const ComplexAbs = 'ComplexAbs';
export type ComplexAbsInputs = UnaryInputs;

export const Concat = 'Concat';
export type ConcatInputs = TensorInfo[];
export interface ConcatAttrs {
  axis: number;
}

export const Conv2D = 'Conv2D';
export type Conv2DInputs = RequiredKernelInputs<'x'|'filter'>;
export interface Conv2DAttrs {
  strides: [number, number]|number;
  pad: 'valid'|'same'|number|ExplicitPadding;
  dataFormat: 'NHWC'|'NCHW';
  dilations: [number, number]|number;
  // CG305: The public convolution/pooling operation permits no explicit rounding override; omission is part of the kernel attribute contract.
  //@cg-expect-error
  dimRoundingMode?: 'floor'|'round'|'ceil';
}

export const Conv2DBackpropFilter = 'Conv2DBackpropFilter';
export type Conv2DBackpropFilterInputs = RequiredKernelInputs<'x'|'dy'>;
export interface Conv2DBackpropFilterAttrs {
  strides: [number, number]|number;
  pad: 'valid'|'same'|number|ExplicitPadding;
  dataFormat: 'NHWC'|'NCHW';
  // CG305: The public convolution/pooling operation permits no explicit rounding override; omission is part of the kernel attribute contract.
  //@cg-expect-error
  dimRoundingMode?: 'floor'|'round'|'ceil';
  filterShape: [number, number, number, number];
}

export const Conv2DBackpropInput = 'Conv2DBackpropInput';
export type Conv2DBackpropInputInputs = RequiredKernelInputs<'dy'|'filter'>;
export interface Conv2DBackpropInputAttrs {
  strides: [number, number]|number;
  pad: 'valid'|'same'|number|ExplicitPadding;
  dataFormat: 'NHWC'|'NCHW';
  // CG305: The public convolution/pooling operation permits no explicit rounding override; omission is part of the kernel attribute contract.
  //@cg-expect-error
  dimRoundingMode?: 'floor'|'round'|'ceil';
  inputShape: [number, number, number, number];
}

export const Conv3D = 'Conv3D';
export type Conv3DInputs = RequiredKernelInputs<'x'|'filter'>;
export interface Conv3DAttrs {
  strides: [number, number, number]|number;
  pad: 'valid'|'same';
  dataFormat: 'NDHWC'|'NCDHW';
  dilations: [number, number, number]|number;
}

export const Conv3DBackpropFilterV2 = 'Conv3DBackpropFilterV2';
export type Conv3DBackpropFilterV2Inputs = RequiredKernelInputs<'x'|'dy'>;

export interface Conv3DBackpropFilterV2Attrs {
  strides: [number, number, number]|number;
  pad: 'valid'|'same';
  filterShape: [number, number, number, number, number];
}

export const Conv3DBackpropInputV2 = 'Conv3DBackpropInputV2';
export type Conv3DBackpropInputV2Inputs =
    RequiredKernelInputs<'dy'|'filter'>;
export interface Conv3DBackpropInputV2Attrs {
  strides: [number, number, number]|number;
  pad: 'valid'|'same';
  inputShape: [number, number, number, number, number];
}

export const Cos = 'Cos';
export type CosInputs = UnaryInputs;

export const Cosh = 'Cosh';
export type CoshInputs = UnaryInputs;

export const Cumprod = 'Cumprod';
export type CumprodInputs = RequiredKernelInputs<'x'>;
export interface CumprodAttrs {
  axis: number;
  exclusive: boolean;
  reverse: boolean;
}

export const Cumsum = 'Cumsum';
export type CumsumInputs = RequiredKernelInputs<'x'>;
export interface CumsumAttrs {
  axis: number;
  exclusive: boolean;
  reverse: boolean;
}

export const CropAndResize = 'CropAndResize';
export type CropAndResizeInputs =
    RequiredKernelInputs<'image'|'boxes'|'boxInd'>;
export interface CropAndResizeAttrs {
  cropSize: [number, number];
  method: 'bilinear'|'nearest';
  extrapolationValue: number;
}

export const DenseBincount = 'DenseBincount';
export type DenseBincountInputs = RequiredKernelInputs<'x'|'weights'>;
export interface DenseBincountAttrs {
  size: number;
  // CG305: Dense bincount supports count accumulation without the binary-output override.
  //@cg-expect-error
  binaryOutput?: boolean;
}

export const DepthToSpace = 'DepthToSpace';
export type DepthToSpaceInputs = RequiredKernelInputs<'x'>;
export interface DepthToSpaceAttrs {
  blockSize: number;
  dataFormat: 'NHWC'|'NCHW';
}

export const DepthwiseConv2dNative = 'DepthwiseConv2dNative';
export type DepthwiseConv2dNativeInputs =
    RequiredKernelInputs<'x'|'filter'>;
export interface DepthwiseConv2dNativeAttrs {
  strides: [number, number]|number;
  pad: 'valid'|'same'|number|ExplicitPadding;
  dataFormat: 'NHWC'|'NCHW';
  dilations: [number, number]|number;
  // CG305: The public convolution/pooling operation permits no explicit rounding override; omission is part of the kernel attribute contract.
  //@cg-expect-error
  dimRoundingMode?: 'floor'|'round'|'ceil';
}

export const DepthwiseConv2dNativeBackpropFilter =
    'DepthwiseConv2dNativeBackpropFilter';
export type DepthwiseConv2dNativeBackpropFilterInputs =
    RequiredKernelInputs<'x'|'dy'>;
export interface DepthwiseConv2dNativeBackpropFilterAttrs {
  strides: [number, number]|number;
  dilations: [number, number]|number;
  pad: 'valid'|'same'|number|ExplicitPadding;
  // CG305: The public convolution/pooling operation permits no explicit rounding override; omission is part of the kernel attribute contract.
  //@cg-expect-error
  dimRoundingMode?: 'floor'|'round'|'ceil';
  filterShape: [number, number, number, number];
}

export const DepthwiseConv2dNativeBackpropInput =
    'DepthwiseConv2dNativeBackpropInput';
export type DepthwiseConv2dNativeBackpropInputInputs =
    RequiredKernelInputs<'dy'|'filter'>;
export interface DepthwiseConv2dNativeBackpropInputAttrs {
  strides: [number, number]|number;
  dilations: [number, number]|number;
  pad: 'valid'|'same'|number|ExplicitPadding;
  // CG305: The public convolution/pooling operation permits no explicit rounding override; omission is part of the kernel attribute contract.
  //@cg-expect-error
  dimRoundingMode?: 'floor'|'round'|'ceil';
  inputShape: [number, number, number, number];
}

export const Diag = 'Diag';
export type DiagInputs = RequiredKernelInputs<'x'>;

export const Dilation2D = 'Dilation2D';
export type Dilation2DInputs = RequiredKernelInputs<'x'|'filter'>;
export interface Dilation2DAttrs {
  strides: [number, number]|number;
  pad: 'valid'|'same'|number;
  dilations: [number, number]|number;
}

export const Dilation2DBackpropInput = 'Dilation2DBackpropInput';
export type Dilation2DBackpropInputInputs =
    RequiredKernelInputs<'x'|'filter'|'dy'>;

export const Dilation2DBackpropFilter = 'Dilation2DBackpropFilter';
export type Dilation2DBackpropFilterInputs =
    RequiredKernelInputs<'x'|'filter'|'dy'>;

export const Draw = 'Draw';
export type DrawInputs = RequiredKernelInputs<'image'>;
export interface DrawAttrs {
  canvas: HTMLCanvasElement;
  // CG305: Drawing accepts default rendering options.
  //@cg-expect-error
  options?: DrawOptions;
}

export const RealDiv = 'RealDiv';
export type RealDivInputs = BinaryInputs;

export const Einsum = 'Einsum';
export type EinsumInputs = TensorInfo[];
export interface EinsumAttrs {
  equation: string;
}

export const Elu = 'Elu';
export type EluInputs = RequiredKernelInputs<'x'>;

export const EluGrad = 'EluGrad';
export type EluGradInputs = RequiredKernelInputs<'dy'|'y'>;

export const Erf = 'Erf';
export type ErfInputs = UnaryInputs;

export const Equal = 'Equal';
export type EqualInputs = BinaryInputs;

export const Exp = 'Exp';
export type ExpInputs = UnaryInputs;

export const ExpandDims = 'ExpandDims';
export type ExpandDimsInputs = RequiredKernelInputs<'input'>;
export interface ExpandDimsAttrs {
  dim: number;
}

export const Expm1 = 'Expm1';
export type Expm1Inputs = UnaryInputs;

export const FFT = 'FFT';
export type FFTInputs = RequiredKernelInputs<'input'>;

export const Fill = 'Fill';
export interface FillAttrs {
  shape: number[];
  value: number|string;
  dtype: DataType;
}

export const FlipLeftRight = 'FlipLeftRight';
export type FlipLeftRightInputs = RequiredKernelInputs<'image'>;

export const Floor = 'Floor';
export type FloorInputs = UnaryInputs;

export const FloorDiv = 'FloorDiv';
export type FloorDivInputs = BinaryInputs;

export const FusedBatchNorm = 'FusedBatchNorm';
export type FusedBatchNormInputs =
    RequiredKernelInputs<'x'|'mean'|'variance'> &
    Pick<NamedTensorInfoMap, 'scale'|'offset'>;
export interface FusedBatchNormAttrs {
  varianceEpsilon: number;
}

export const GatherV2 = 'GatherV2';
export type GatherV2Inputs = RequiredKernelInputs<'x'|'indices'>;
export interface GatherV2Attrs {
  axis: number;
  batchDims: number;
}

export const GatherNd = 'GatherNd';
export type GatherNdInputs = RequiredKernelInputs<'params'|'indices'>;

export const Greater = 'Greater';
export type GreaterInputs = BinaryInputs;

export const GreaterEqual = 'GreaterEqual';
export type GreaterEqualInputs = BinaryInputs;

export const Identity = 'Identity';
export type IdentityInputs = RequiredKernelInputs<'x'>;

export const IFFT = 'IFFT';
export type IFFTInputs = RequiredKernelInputs<'input'>;

export const Imag = 'Imag';
export type ImagInputs = RequiredKernelInputs<'input'>;

export const IsFinite = 'IsFinite';
export type IsFiniteInputs = UnaryInputs;

export const IsInf = 'IsInf';
export type IsInfInputs = UnaryInputs;

export const IsNan = 'IsNan';
export type IsNanInputs = UnaryInputs;

export const LeakyRelu = 'LeakyRelu';
export type LeakyReluInputs = RequiredKernelInputs<'x'>;
export interface LeakyReluAttrs {
  alpha: number;
}

export const Less = 'Less';
export type LessInputs = BinaryInputs;

export const LessEqual = 'LessEqual';
export type LessEqualInputs = BinaryInputs;

export const LinSpace = 'LinSpace';
export interface LinSpaceAttrs {
  start: number;
  stop: number;
  num: number;
}
export const Log = 'Log';
export type LogInputs = UnaryInputs;

export const Log1p = 'Log1p';
export type Log1pInputs = UnaryInputs;

export const LogicalAnd = 'LogicalAnd';
export type LogicalAndInputs = BinaryInputs;

export const LogicalNot = 'LogicalNot';
export type LogicalNotInputs = RequiredKernelInputs<'x'>;

export const LogicalOr = 'LogicalOr';
export type LogicalOrInputs = BinaryInputs;

export const LogicalXor = 'LogicalXor';
export type LogicalXorInputs = BinaryInputs;

export const LogSoftmax = 'LogSoftmax';
export type LogSoftmaxInputs = RequiredKernelInputs<'logits'>;
export interface LogSoftmaxAttrs {
  axis: number;
}

export const LowerBound = 'LowerBound';
export type LowerBoundInputs =
    RequiredKernelInputs<'sortedSequence'|'values'>;

export const LRN = 'LRN';
export type LRNInputs = RequiredKernelInputs<'x'>;
export interface LRNAttrs {
  depthRadius: number;
  bias: number;
  alpha: number;
  beta: number;
}

export const LRNGrad = 'LRNGrad';
export type LRNGradInputs = RequiredKernelInputs<'x'|'y'|'dy'>;
export interface LRNGradAttrs {
  depthRadius: number;
  bias: number;
  alpha: number;
  beta: number;
}

export const MatrixBandPart = 'MatrixBandPart';
export type MatrixBandPartInputs =
    RequiredKernelInputs<'input'|'numLower'|'numUpper'>;
export interface MatrixBandPartAttrs {}

export const Max = 'Max';
export type MaxInputs = RequiredKernelInputs<'x'>;
export interface MaxAttrs {
  reductionIndices: number|number[];
  keepDims: boolean;
}

export const Maximum = 'Maximum';
export type MaximumInputs = BinaryInputs;

export const MaxPool = 'MaxPool';
export type MaxPoolInputs = RequiredKernelInputs<'x'>;
export interface MaxPoolAttrs {
  filterSize: [number, number]|number;
  strides: [number, number]|number;
  pad: 'valid'|'same'|number|ExplicitPadding;
  // CG305: The public convolution/pooling operation permits no explicit rounding override; omission is part of the kernel attribute contract.
  //@cg-expect-error
  dimRoundingMode?: 'floor'|'round'|'ceil';
}

export const MaxPoolGrad = 'MaxPoolGrad';
export type MaxPoolGradInputs = RequiredKernelInputs<'dy'|'input'|'output'>;
export interface MaxPoolGradAttrs {
  filterSize: [number, number]|number;
  strides: [number, number]|number;
  pad: 'valid'|'same'|number|ExplicitPadding;
  // CG305: The public convolution/pooling operation permits no explicit rounding override; omission is part of the kernel attribute contract.
  //@cg-expect-error
  dimRoundingMode?: 'floor'|'round'|'ceil';
}

export const MaxPool3D = 'MaxPool3D';
export type MaxPool3DInputs = RequiredKernelInputs<'x'>;
export interface MaxPool3DAttrs {
  filterSize: [number, number, number]|number;
  strides: [number, number, number]|number;
  pad: 'valid'|'same'|number;
  dataFormat: 'NDHWC'|'NCDHW';
  // CG305: The public convolution/pooling operation permits no explicit rounding override; omission is part of the kernel attribute contract.
  //@cg-expect-error
  dimRoundingMode?: 'floor'|'round'|'ceil';
}

export const MaxPool3DGrad = 'MaxPool3DGrad';
export type MaxPool3DGradInputs =
    RequiredKernelInputs<'dy'|'input'|'output'>;
export interface MaxPool3DGradAttrs {
  filterSize: [number, number, number]|number;
  strides: [number, number, number]|number;
  pad: 'valid'|'same'|number;
  // CG305: The public convolution/pooling operation permits no explicit rounding override; omission is part of the kernel attribute contract.
  //@cg-expect-error
  dimRoundingMode?: 'floor'|'round'|'ceil';
}

export const MaxPoolWithArgmax = 'MaxPoolWithArgmax';
export type MaxPoolWithArgmaxInputs = RequiredKernelInputs<'x'>;
export interface MaxPoolWithArgmaxAttrs {
  filterSize: [number, number]|number;
  strides: [number, number]|number;
  pad: 'valid'|'same'|number;
  includeBatchInIndex: boolean;
}

export const Mean = 'Mean';
export type MeanInputs = RequiredKernelInputs<'x'>;
export interface MeanAttrs {
  axis: number|number[];
  keepDims: boolean;
}

export const Min = 'Min';
export type MinInputs = RequiredKernelInputs<'x'>;
export interface MinAttrs {
  axis: number|number[];
  keepDims: boolean;
}

export const Minimum = 'Minimum';
export type MinimumInputs = BinaryInputs;

export const MirrorPad = 'MirrorPad';
export type MirrorPadInputs = RequiredKernelInputs<'x'>;
export interface MirrorPadAttrs {
  paddings: Array<[number, number]>;
  mode: 'reflect'|'symmetric';
}

export const Mod = 'Mod';
export type ModInputs = BinaryInputs;

export const Multinomial = 'Multinomial';
export type MultinomialInputs = RequiredKernelInputs<'logits'>;
export interface MultinomialAttrs {
  numSamples: number;
  seed: number;
  normalized: boolean;
}

export const Multiply = 'Multiply';
export type MultiplyInputs = BinaryInputs;

export const Neg = 'Neg';
export type NegInputs = UnaryInputs;

export const NotEqual = 'NotEqual';
export type NotEqualInputs = BinaryInputs;

export const NonMaxSuppressionV3 = 'NonMaxSuppressionV3';
export type NonMaxSuppressionV3Inputs =
    RequiredKernelInputs<'boxes'|'scores'>;
export interface NonMaxSuppressionV3Attrs {
  maxOutputSize: number;
  iouThreshold: number;
  scoreThreshold: number;
}

export const NonMaxSuppressionV4 = 'NonMaxSuppressionV4';
export type NonMaxSuppressionV4Inputs =
    RequiredKernelInputs<'boxes'|'scores'>;
export interface NonMaxSuppressionV4Attrs {
  maxOutputSize: number;
  iouThreshold: number;
  scoreThreshold: number;
  padToMaxOutputSize: boolean;
}

export const NonMaxSuppressionV5 = 'NonMaxSuppressionV5';
export type NonMaxSuppressionV5Inputs =
    RequiredKernelInputs<'boxes'|'scores'>;
export interface NonMaxSuppressionV5Attrs {
  maxOutputSize: number;
  iouThreshold: number;
  scoreThreshold: number;
  softNmsSigma: number;
}

export const OnesLike = 'OnesLike';
export type OnesLikeInputs = UnaryInputs;

export const OneHot = 'OneHot';
export type OneHotInputs = RequiredKernelInputs<'indices'>;
export interface OneHotAttrs {
  depth: number;
  onValue: number;
  offValue: number;
  dtype: DataType;
}

export const Pack = 'Pack';
export type PackInputs = TensorInfo[];
export interface PackAttrs {
  axis: number;
}

export const PadV2 = 'PadV2';
export type PadV2Inputs = RequiredKernelInputs<'x'>;
export interface PadV2Attrs {
  paddings: Array<[number, number]>;
  constantValue: number;
}

export const Pool = 'Pool';
export type PoolInputs = RequiredKernelInputs<'input'>;

export const Pow = 'Pow';
export type PowInputs = BinaryInputs;

export const Prelu = 'Prelu';
export type PreluInputs = RequiredKernelInputs<'x'|'alpha'>;

export const Prod = 'Prod';
export type ProdInputs = RequiredKernelInputs<'x'>;
export interface ProdAttrs {
  axis: number|number[];
  keepDims: boolean;
}

export const RaggedGather = 'RaggedGather';
export type RaggedGatherInputs = {
  paramsNestedSplits: TensorInfo[]
}&RequiredKernelInputs<'paramsDenseValues'|'indices'>;
export interface RaggedGatherAttrs {
  outputRaggedRank: number;
}

export const RaggedRange = 'RaggedRange';
export type RaggedRangeInputs =
    RequiredKernelInputs<'starts'|'limits'|'deltas'>;

export const RaggedTensorToTensor = 'RaggedTensorToTensor';
export type RaggedTensorToTensorInputs =
    RequiredKernelInputs<'shape'|'values'|'defaultValue'>&
    {rowPartitionTensors: TensorInfo[]};
export interface RaggedTensorToTensorAttrs {
  rowPartitionTypes: string[];
}

export const Range = 'Range';
export interface RangeAttrs {
  start: number;
  stop: number;
  step: number;
  dtype: 'float32'|'int32';
}

export const Real = 'Real';
export type RealInputs = RequiredKernelInputs<'input'>;

export const Reciprocal = 'Reciprocal';
export type ReciprocalInputs = UnaryInputs;

export const Relu = 'Relu';
export type ReluInputs = RequiredKernelInputs<'x'>;

export const Reshape = 'Reshape';
export type ReshapeInputs = RequiredKernelInputs<'x'>;
export interface ReshapeAttrs {
  shape: number[];
}

export const ResizeNearestNeighbor = 'ResizeNearestNeighbor';
export type ResizeNearestNeighborInputs = RequiredKernelInputs<'images'>;
export interface ResizeNearestNeighborAttrs {
  alignCorners: boolean;
  halfPixelCenters: boolean;
  size: [number, number];
}

export const ResizeNearestNeighborGrad = 'ResizeNearestNeighborGrad';
export type ResizeNearestNeighborGradInputs =
    RequiredKernelInputs<'images'|'dy'>;
export type ResizeNearestNeighborGradAttrs = ResizeNearestNeighborAttrs;

export const ResizeBilinear = 'ResizeBilinear';
export type ResizeBilinearInputs = RequiredKernelInputs<'images'>;
export interface ResizeBilinearAttrs {
  alignCorners: boolean;
  halfPixelCenters: boolean;
  size: [number, number];
}

export const ResizeBilinearGrad = 'ResizeBilinearGrad';
export type ResizeBilinearGradInputs = RequiredKernelInputs<'images'|'dy'>;
export type ResizeBilinearGradAttrs = ResizeBilinearAttrs;

export const Relu6 = 'Relu6';
export type Relu6Inputs = RequiredKernelInputs<'x'>;

export const Reverse = 'Reverse';
export type ReverseInputs = RequiredKernelInputs<'x'>;
export interface ReverseAttrs {
  dims: number|number[];
}

export const Round = 'Round';
export type RoundInputs = UnaryInputs;

export const Rsqrt = 'Rsqrt';
export type RsqrtInputs = UnaryInputs;

export const ScatterNd = 'ScatterNd';
export type ScatterNdInputs = RequiredKernelInputs<'indices'|'updates'>;
export interface ScatterNdAttrs {
  shape: number[];
}

export const TensorScatterUpdate = 'TensorScatterUpdate';
export type TensorScatterUpdateInputs =
    RequiredKernelInputs<'tensor'|'indices'|'updates'>;
export interface TensorScatterUpdateAttrs {}

export const SearchSorted = 'SearchSorted';
export type SearchSortedInputs =
    RequiredKernelInputs<'sortedSequence'|'values'>;
export interface SearchSortedAttrs {
  side: 'left'|'right';
}

export const Select = 'Select';
export type SelectInputs = RequiredKernelInputs<'condition'|'t'|'e'>;

export const Selu = 'Selu';
export type SeluInputs = RequiredKernelInputs<'x'>;

export const Slice = 'Slice';
export type SliceInputs = RequiredKernelInputs<'x'>;
export interface SliceAttrs {
  begin: number|number[];
  size: number|number[];
}
export const Sin = 'Sin';
export type SinInputs = UnaryInputs;

export const Sinh = 'Sinh';
export type SinhInputs = UnaryInputs;

export const Sign = 'Sign';
export type SignInputs = UnaryInputs;

export const Sigmoid = 'Sigmoid';
export type SigmoidInputs = UnaryInputs;

export const Softplus = 'Softplus';
export type SoftplusInputs = UnaryInputs;

export const Sqrt = 'Sqrt';
export type SqrtInputs = UnaryInputs;

export const Sum = 'Sum';
export type SumInputs = RequiredKernelInputs<'x'>;
export interface SumAttrs {
  axis: number|number[];
  keepDims: boolean;
}

export const SpaceToBatchND = 'SpaceToBatchND';
export type SpaceToBatchNDInputs = RequiredKernelInputs<'x'>;
export interface SpaceToBatchNDAttrs {
  blockShape: number[];
  paddings: number[][];
}

export const SplitV = 'SplitV';
export type SplitVInputs = RequiredKernelInputs<'x'>;
export interface SplitVAttrs {
  numOrSizeSplits: number[]|number;
  axis: number;
}

export const Softmax = 'Softmax';
export type SoftmaxInputs = RequiredKernelInputs<'logits'>;
export interface SoftmaxAttrs {
  dim: number;
}

export const SparseFillEmptyRows = 'SparseFillEmptyRows';
export type SparseFillEmptyRowsInputs =
    RequiredKernelInputs<'indices'|'values'|'denseShape'|'defaultValue'>;

export const SparseReshape = 'SparseReshape';
export type SparseReshapeInputs =
    RequiredKernelInputs<'inputIndices'|'inputShape'|'newShape'>;

export const SparseSegmentMean = 'SparseSegmentMean';
export type SparseSegmentMeanInputs =
    RequiredKernelInputs<'data'|'indices'|'segmentIds'>;

export const SparseSegmentSum = 'SparseSegmentSum';
export type SparseSegmentSumInputs =
    RequiredKernelInputs<'data'|'indices'|'segmentIds'>;

export const SparseToDense = 'SparseToDense';
export type SparseToDenseInputs =
    RequiredKernelInputs<'sparseIndices'|'sparseValues'|'defaultValue'>;
export interface SparseToDenseAttrs {
  outputShape: number[];
}

export const SquaredDifference = 'SquaredDifference';
export type SquaredDifferenceInputs = BinaryInputs;

export const Square = 'Square';
export type SquareInputs = RequiredKernelInputs<'x'>;

export const StaticRegexReplace = 'StaticRegexReplace';
export type StaticRegexReplaceInputs = UnaryInputs;
export interface StaticRegexReplaceAttrs {
  pattern: string;
  rewrite: string;
  replaceGlobal: boolean;
}

export const StridedSlice = 'StridedSlice';
export type StridedSliceInputs = RequiredKernelInputs<'x'>;
export interface StridedSliceAttrs {
  begin: number[];
  end: number[];
  strides: number[];
  beginMask: number;
  endMask: number;
  ellipsisMask: number;
  newAxisMask: number;
  shrinkAxisMask: number;
}

export const StringNGrams = 'StringNGrams';
export type StringNGramsInputs = RequiredKernelInputs<'data'|'dataSplits'>;
export interface StringNGramsAttrs {
  separator: string;
  nGramWidths: number[];
  leftPad: string;
  rightPad: string;
  padWidth: number;
  preserveShortSequences: boolean;
}

export const StringSplit = 'StringSplit';
export type StringSplitInputs = RequiredKernelInputs<'input'|'delimiter'>;
export interface StringSplitAttrs {
  skipEmpty: boolean;
}

export const StringToHashBucketFast = 'StringToHashBucketFast';
export type StringToHashBucketFastInputs = RequiredKernelInputs<'input'>;
export interface StringToHashBucketFastAttrs {
  numBuckets: number;
}

export const Sub = 'Sub';
export type SubInputs = BinaryInputs;

export const Tan = 'Tan';
export type TanInputs = UnaryInputs;

export const Tanh = 'Tanh';
export type TanhInputs = UnaryInputs;

export const Tile = 'Tile';
export type TileInputs = RequiredKernelInputs<'x'>;
export interface TileAttrs {
  reps: number[];
}

export const TopK = 'TopK';
export type TopKInputs = RequiredKernelInputs<'x'>;
export interface TopKAttrs {
  k: number;
  sorted: boolean;
}

export const Transform = 'Transform';
export type TransformInputs = RequiredKernelInputs<'image'|'transforms'>;
export interface TransformAttrs {
  interpolation: 'nearest'|'bilinear';
  fillMode: 'constant'|'reflect'|'wrap'|'nearest';
  fillValue: number;
  // CG305: Transform uses the input image dimensions if no output shape is requested.
  //@cg-expect-error
  outputShape?: [number, number];
}

export const Transpose = 'Transpose';
export type TransposeInputs = RequiredKernelInputs<'x'>;
export interface TransposeAttrs {
  perm: number[];
}

export const Unique = 'Unique';
export type UniqueInputs = RequiredKernelInputs<'x'>;
export interface UniqueAttrs {
  axis: number;
}

export type UnaryInputs = RequiredKernelInputs<'x'>;

export const Unpack = 'Unpack';
export type UnpackInputs = RequiredKernelInputs<'value'>;
export interface UnpackAttrs {
  axis: number;
}

export const UnsortedSegmentSum = 'UnsortedSegmentSum';
export type UnsortedSegmentSumInputs =
    RequiredKernelInputs<'x'|'segmentIds'>;
export interface UnsortedSegmentSumAttrs {
  numSegments: number;
}

export const UpperBound = 'UpperBound';
export type UpperBoundInputs =
    RequiredKernelInputs<'sortedSequence'|'values'>;

export const ZerosLike = 'ZerosLike';
export type ZerosLikeInputs = UnaryInputs;

/**
 * TensorFlow.js-only kernels
 */
export const Step = 'Step';
export type StepInputs = UnaryInputs;
export interface StepAttrs {
  alpha: number;
}

export const FromPixels = 'FromPixels';
export interface FromPixelsInputs {
  pixels: PixelData|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement|ImageBitmap;
}
export interface FromPixelsAttrs {
  numChannels: number;
}

export const RotateWithOffset = 'RotateWithOffset';
export type RotateWithOffsetInputs = RequiredKernelInputs<'image'>;
export interface RotateWithOffsetAttrs {
  radians: number;
  fillValue: number|[number, number, number];
  center: number|[number, number];
}

export const _FusedMatMul = '_FusedMatMul';
// tslint:disable-next-line: class-name
export interface _FusedMatMulInputs extends NamedTensorInfoMap {
  a: TensorInfo;
  b: TensorInfo;
  // CG305: Fused operations support execution without a bias tensor.
  //@cg-expect-error
  bias?: TensorInfo;
  // CG305: Only PReLU activation requires the additional weights tensor.
  //@cg-expect-error
  preluActivationWeights?: TensorInfo;
}
// tslint:disable-next-line: class-name
export interface _FusedMatMulAttrs {
  transposeA: boolean;
  transposeB: boolean;
  activation: Activation;
  // CG305: An alpha override is only relevant for leaky ReLU activation.
  //@cg-expect-error
  leakyreluAlpha?: number;
}

export const FusedConv2D = 'FusedConv2D';
export interface FusedConv2DInputs extends NamedTensorInfoMap {
  x: TensorInfo;
  filter: TensorInfo;
  // CG305: Fused operations support execution without a bias tensor.
  //@cg-expect-error
  bias?: TensorInfo;
  // CG305: Only PReLU activation requires the additional weights tensor.
  //@cg-expect-error
  preluActivationWeights?: TensorInfo;
}
export interface FusedConv2DAttrs {
  strides: [number, number]|number;
  pad: 'valid'|'same'|number|ExplicitPadding;
  dataFormat: 'NHWC'|'NCHW';
  dilations: [number, number]|number;
  dimRoundingMode: 'floor'|'round'|'ceil';
  activation: Activation;
  // CG305: An alpha override is only relevant for leaky ReLU activation.
  //@cg-expect-error
  leakyreluAlpha?: number;
}

export const FusedDepthwiseConv2D = 'FusedDepthwiseConv2D';
export interface FusedDepthwiseConv2DInputs extends NamedTensorInfoMap {
  x: TensorInfo;
  filter: TensorInfo;
  // CG305: Fused operations support execution without a bias tensor.
  //@cg-expect-error
  bias?: TensorInfo;
  // CG305: Only PReLU activation requires the additional weights tensor.
  //@cg-expect-error
  preluActivationWeights?: TensorInfo;
}
export interface FusedDepthwiseConv2DAttrs {
  strides: [number, number]|number;
  pad: 'valid'|'same'|number|ExplicitPadding;
  dataFormat: 'NHWC'|'NCHW';
  dilations: [number, number]|number;
  dimRoundingMode: 'floor'|'round'|'ceil';
  activation: Activation;
  // CG305: An alpha override is only relevant for leaky ReLU activation.
  //@cg-expect-error
  leakyreluAlpha?: number;
}
