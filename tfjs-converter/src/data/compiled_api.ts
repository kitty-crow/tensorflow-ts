/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
 *
 * =============================================================================
 */

/* tslint:disable */

/** Properties of an Any. */
export declare interface IAny {
  /** Any typeUrl */
  // CG305: Any.typeUrl (api.proto field 1). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  typeUrl?: (string|null);

  /** Any value */
  // CG305: Any.value (api.proto field 2). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  value?: (Uint8Array|null);
}

/** DataType enum. */
export enum DataType {
  // These properties must be quoted since they are used by parseDtypeParam
  // in tfjs-converter/src/operations/operation_mapper.ts to look up dtypes
  // by string name. If they are not quoted, Closure will mangle their names.

  // Not a legal value for DataType.  Used to indicate a DataType field
  // has not been set.
  'DT_INVALID' = 0,

  // Data types that all computation devices are expected to be
  // capable to support.
  'DT_FLOAT' = 1,
  'DT_DOUBLE' = 2,
  'DT_INT32' = 3,
  'DT_UINT8' = 4,
  'DT_INT16' = 5,
  'DT_INT8' = 6,
  'DT_STRING' = 7,
  'DT_COMPLEX64' = 8,  // Single-precision complex
  'DT_INT64' = 9,
  'DT_BOOL' = 10,
  'DT_QINT8' = 11,     // Quantized int8
  'DT_QUINT8' = 12,    // Quantized uint8
  'DT_QINT32' = 13,    // Quantized int32
  'DT_BFLOAT16' = 14,  // Float32 truncated to 16 bits.  Only for cast ops.
  'DT_QINT16' = 15,    // Quantized int16
  'DT_QUINT16' = 16,   // Quantized uint16
  'DT_UINT16' = 17,
  'DT_COMPLEX128' = 18,  // Double-precision complex
  'DT_HALF' = 19,
  'DT_RESOURCE' = 20,
  'DT_VARIANT' = 21,  // Arbitrary C++ data types
  'DT_UINT32' = 22,
  'DT_UINT64' = 23,

  // Do not use!  These are only for parameters.  Every enum above
  // should have a corresponding value below (verified by types_test).
  'DT_FLOAT_REF' = 101,
  'DT_DOUBLE_REF' = 102,
  'DT_INT32_REF' = 103,
  'DT_UINT8_REF' = 104,
  'DT_INT16_REF' = 105,
  'DT_INT8_REF' = 106,
  'DT_STRING_REF' = 107,
  'DT_COMPLEX64_REF' = 108,
  'DT_INT64_REF' = 109,
  'DT_BOOL_REF' = 110,
  'DT_QINT8_REF' = 111,
  'DT_QUINT8_REF' = 112,
  'DT_QINT32_REF' = 113,
  'DT_BFLOAT16_REF' = 114,
  'DT_QINT16_REF' = 115,
  'DT_QUINT16_REF' = 116,
  'DT_UINT16_REF' = 117,
  'DT_COMPLEX128_REF' = 118,
  'DT_HALF_REF' = 119,
  'DT_RESOURCE_REF' = 120,
  'DT_VARIANT_REF' = 121,
  'DT_UINT32_REF' = 122,
  'DT_UINT64_REF' = 123,
}

/** Properties of a TensorShape. */
export declare interface ITensorShape {
  /** TensorShape dim */
  // CG305: TensorShape.dim (api.proto field 2). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  dim?: (TensorShape.IDim[]|null);

  /** TensorShape unknownRank */
  // CG305: TensorShape.unknownRank (api.proto field 3). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  unknownRank?: (boolean|null);
}

export namespace TensorShape {
  /** Properties of a Dim. */
  export declare interface IDim {
    /** Dim size */
    // CG305: TensorShape.Dim.size (api.proto field 1). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    size?: (number|string|null);

    /** Dim name */
    // CG305: TensorShape.Dim.name (api.proto field 2). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    name?: (string|null);
  }
}

/** Properties of a Tensor. */
export declare interface ITensor {
  /** Tensor dtype */
  // CG305: Tensor.dtype (api.proto field 1). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  dtype?: (DataType|null);

  /** Tensor tensorShape */
  // CG305: Tensor.tensorShape (api.proto field 2). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  tensorShape?: (ITensorShape|null);

  /** Tensor versionNumber */
  // CG305: Tensor.versionNumber (api.proto field 3). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  versionNumber?: (number|null);

  /** Tensor tensorContent */
  // CG305: Tensor.tensorContent (api.proto field 4). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  tensorContent?: (Uint8Array|null);

  /** Tensor floatVal */
  // CG305: Tensor.floatVal (api.proto field 5). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  floatVal?: (number[]|null);

  /** Tensor doubleVal */
  // CG305: Tensor.doubleVal (api.proto field 6). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  doubleVal?: (number[]|null);

  /** Tensor intVal */
  // CG305: Tensor.intVal (api.proto field 7). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  intVal?: (number[]|null);

  /** Tensor stringVal */
  // CG305: Tensor.stringVal (api.proto field 8). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  stringVal?: (Uint8Array[]|null);

  /** Tensor scomplexVal */
  // CG305: Tensor.scomplexVal (api.proto field 9). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  scomplexVal?: (number[]|null);

  /** Tensor int64Val */
  // CG305: Tensor.int64Val (api.proto field 10). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  int64Val?: ((number | string)[]|null);

  /** Tensor boolVal */
  // CG305: Tensor.boolVal (api.proto field 11). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  boolVal?: (boolean[]|null);

  /** Tensor uint32Val */
  // CG305: Tensor.uint32Val (api.proto field 16). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  uint32Val?: (number[]|null);

  /** Tensor uint64Val */
  // CG305: Tensor.uint64Val (api.proto field 17). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  uint64Val?: ((number | string)[]|null);
}

/** Properties of an AttrValue. */
export declare interface IAttrValue {
  /** AttrValue list */
  // CG305: AttrValue.list (api.proto field 1). An unselected protobuf oneof alternative is absent.
  //@cg-expect-error
  list?: (AttrValue.IListValue|null);

  /** AttrValue s */
  // CG305: AttrValue.s (api.proto field 2). An unselected protobuf oneof alternative is absent.
  //@cg-expect-error
  s?: (string|null);

  /** AttrValue i */
  // CG305: AttrValue.i (api.proto field 3). An unselected protobuf oneof alternative is absent.
  //@cg-expect-error
  i?: (number|string|null);

  /** AttrValue f */
  // CG305: AttrValue.f (api.proto field 4). An unselected protobuf oneof alternative is absent.
  //@cg-expect-error
  f?: (number|null);

  /** AttrValue b */
  // CG305: AttrValue.b (api.proto field 5). An unselected protobuf oneof alternative is absent.
  //@cg-expect-error
  b?: (boolean|null);

  /** AttrValue type */
  // CG305: AttrValue.type (api.proto field 6). An unselected protobuf oneof alternative is absent.
  //@cg-expect-error
  type?: (DataType|null);

  /** AttrValue shape */
  // CG305: AttrValue.shape (api.proto field 7). An unselected protobuf oneof alternative is absent.
  //@cg-expect-error
  shape?: (ITensorShape|null);

  /** AttrValue tensor */
  // CG305: AttrValue.tensor (api.proto field 8). An unselected protobuf oneof alternative is absent.
  //@cg-expect-error
  tensor?: (ITensor|null);

  /** AttrValue placeholder */
  // CG305: AttrValue.placeholder (api.proto field 9). An unselected protobuf oneof alternative is absent.
  //@cg-expect-error
  placeholder?: (string|null);

  /** AttrValue func */
  // CG305: AttrValue.func (api.proto field 10). An unselected protobuf oneof alternative is absent.
  //@cg-expect-error
  func?: (INameAttrList|null);
}

export namespace AttrValue {
  /** Properties of a ListValue. */
  export declare interface IListValue {
    /** ListValue s */
    // CG305: AttrValue.ListValue.s (api.proto field 2). An empty protobuf collection can be absent from JSON.
    //@cg-expect-error
    s?: (string[]|null);

    /** ListValue i */
    // CG305: AttrValue.ListValue.i (api.proto field 3). An empty protobuf collection can be absent from JSON.
    //@cg-expect-error
    i?: ((number | string)[]|null);

    /** ListValue f */
    // CG305: AttrValue.ListValue.f (api.proto field 4). An empty protobuf collection can be absent from JSON.
    //@cg-expect-error
    f?: (number[]|null);

    /** ListValue b */
    // CG305: AttrValue.ListValue.b (api.proto field 5). An empty protobuf collection can be absent from JSON.
    //@cg-expect-error
    b?: (boolean[]|null);

    /** ListValue type */
    // CG305: AttrValue.ListValue.type (api.proto field 6). An empty protobuf collection can be absent from JSON.
    //@cg-expect-error
    type?: (DataType[]|null);

    /** ListValue shape */
    // CG305: AttrValue.ListValue.shape (api.proto field 7). An empty protobuf collection can be absent from JSON.
    //@cg-expect-error
    shape?: (ITensorShape[]|null);

    /** ListValue tensor */
    // CG305: AttrValue.ListValue.tensor (api.proto field 8). An empty protobuf collection can be absent from JSON.
    //@cg-expect-error
    tensor?: (ITensor[]|null);

    /** ListValue func */
    // CG305: AttrValue.ListValue.func (api.proto field 9). An empty protobuf collection can be absent from JSON.
    //@cg-expect-error
    func?: (INameAttrList[]|null);
  }
}

/** Properties of a NameAttrList. */
export declare interface INameAttrList {
  /** NameAttrList name */
  // CG305: NameAttrList.name (api.proto field 1). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  name?: (string|null);

  /** NameAttrList attr */
  // CG305: NameAttrList.attr (api.proto field 2). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  attr?: ({[k: string]: IAttrValue}|null);
}

/** Properties of a NodeDef. */
export declare interface INodeDef {
  /** NodeDef name */
  // CG305: NodeDef.name (api.proto field 1). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  name?: (string|null);

  /** NodeDef op */
  // CG305: NodeDef.op (api.proto field 2). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  op?: (string|null);

  /** NodeDef input */
  // CG305: NodeDef.input (api.proto field 3). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  input?: (string[]|null);

  /** NodeDef device */
  // CG305: NodeDef.device (api.proto field 4). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  device?: (string|null);

  /** NodeDef attr */
  // CG305: NodeDef.attr (api.proto field 5). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  attr?: ({[k: string]: IAttrValue}|null);
}

/** Properties of a VersionDef. */
export declare interface IVersionDef {
  /** VersionDef producer */
  // CG305: VersionDef.producer (api.proto field 1). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  producer?: (number|null);

  /** VersionDef minConsumer */
  // CG305: VersionDef.minConsumer (api.proto field 2). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  minConsumer?: (number|null);

  /** VersionDef badConsumers */
  // CG305: VersionDef.badConsumers (api.proto field 3). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  badConsumers?: (number[]|null);
}

/** Properties of a GraphDef. */
export declare interface IGraphDef {
  /** GraphDef node */
  // CG305: GraphDef.node (api.proto field 1). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  node?: (INodeDef[]|null);

  /** GraphDef versions */
  // CG305: GraphDef.versions (api.proto field 4). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  versions?: (IVersionDef|null);

  /** GraphDef library */
  // CG305: GraphDef.library (api.proto field 2). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  library?: (IFunctionDefLibrary|null);
}

/** Properties of a CollectionDef. */
export declare interface ICollectionDef {
  /** CollectionDef nodeList */
  // CG305: CollectionDef.nodeList (api.proto field 1). An unselected protobuf oneof alternative is absent.
  //@cg-expect-error
  nodeList?: (CollectionDef.INodeList|null);

  /** CollectionDef bytesList */
  // CG305: CollectionDef.bytesList (api.proto field 2). An unselected protobuf oneof alternative is absent.
  //@cg-expect-error
  bytesList?: (CollectionDef.IBytesList|null);

  /** CollectionDef int64List */
  // CG305: CollectionDef.int64List (api.proto field 3). An unselected protobuf oneof alternative is absent.
  //@cg-expect-error
  int64List?: (CollectionDef.IInt64List|null);

  /** CollectionDef floatList */
  // CG305: CollectionDef.floatList (api.proto field 4). An unselected protobuf oneof alternative is absent.
  //@cg-expect-error
  floatList?: (CollectionDef.IFloatList|null);

  /** CollectionDef anyList */
  // CG305: CollectionDef.anyList (api.proto field 5). An unselected protobuf oneof alternative is absent.
  //@cg-expect-error
  anyList?: (CollectionDef.IAnyList|null);
}

export namespace CollectionDef {
  /** Properties of a NodeList. */
  export declare interface INodeList {
    /** NodeList value */
    // CG305: CollectionDef.NodeList.value (api.proto field 1). An empty protobuf collection can be absent from JSON.
    //@cg-expect-error
    value?: (string[]|null);
  }

  /** Properties of a BytesList. */
  export declare interface IBytesList {
    /** BytesList value */
    // CG305: CollectionDef.BytesList.value (api.proto field 1). An empty protobuf collection can be absent from JSON.
    //@cg-expect-error
    value?: (Uint8Array[]|null);
  }

  /** Properties of an Int64List. */
  export declare interface IInt64List {
    /** Int64List value */
    // CG305: CollectionDef.Int64List.value (api.proto field 1). An empty protobuf collection can be absent from JSON.
    //@cg-expect-error
    value?: ((number | string)[]|null);
  }

  /** Properties of a FloatList. */
  export declare interface IFloatList {
    /** FloatList value */
    // CG305: CollectionDef.FloatList.value (api.proto field 1). An empty protobuf collection can be absent from JSON.
    //@cg-expect-error
    value?: (number[]|null);
  }

  /** Properties of an AnyList. */
  export declare interface IAnyList {
    /** AnyList value */
    // CG305: CollectionDef.AnyList.value (api.proto field 1). An empty protobuf collection can be absent from JSON.
    //@cg-expect-error
    value?: (IAny[]|null);
  }
}

/** Properties of a SaverDef. */
export declare interface ISaverDef {
  /** SaverDef filenameTensorName */
  // CG305: SaverDef.filenameTensorName (api.proto field 1). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  filenameTensorName?: (string|null);

  /** SaverDef saveTensorName */
  // CG305: SaverDef.saveTensorName (api.proto field 2). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  saveTensorName?: (string|null);

  /** SaverDef restoreOpName */
  // CG305: SaverDef.restoreOpName (api.proto field 3). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  restoreOpName?: (string|null);

  /** SaverDef maxToKeep */
  // CG305: SaverDef.maxToKeep (api.proto field 4). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  maxToKeep?: (number|null);

  /** SaverDef sharded */
  // CG305: SaverDef.sharded (api.proto field 5). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  sharded?: (boolean|null);

  /** SaverDef keepCheckpointEveryNHours */
  // CG305: SaverDef.keepCheckpointEveryNHours (api.proto field 6). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  keepCheckpointEveryNHours?: (number|null);

  /** SaverDef version */
  // CG305: SaverDef.version (api.proto field 7). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  version?: (SaverDef.CheckpointFormatVersion|null);
}

export namespace SaverDef {
  /** CheckpointFormatVersion enum. */
  export enum CheckpointFormatVersion {'LEGACY' = 0, 'V1' = 1, 'V2' = 2}
}

/** Properties of a TensorInfo. */
export declare interface ITensorInfo {
  /** TensorInfo name */
  // CG305: TensorInfo.name (api.proto field 1). An unselected protobuf oneof alternative is absent.
  //@cg-expect-error
  name?: (string|null);

  /** TensorInfo cooSparse */
  // CG305: TensorInfo.cooSparse (api.proto field 4). An unselected protobuf oneof alternative is absent.
  //@cg-expect-error
  cooSparse?: (TensorInfo.ICooSparse|null);

  /** TensorInfo dtype */
  // CG305: TensorInfo.dtype (api.proto field 2). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  dtype?: (DataType|string|null);

  /** TensorInfo tensorShape */
  // CG305: TensorInfo.tensorShape (api.proto field 3). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  tensorShape?: (ITensorShape|null);

  /** Resource id tensor was originally assigned to.  */
  resourceId?: (number|null);
}

export namespace TensorInfo {
  /** Properties of a CooSparse. */
  export declare interface ICooSparse {
    /** CooSparse valuesTensorName */
    // CG305: TensorInfo.CooSparse.valuesTensorName (api.proto field 1). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    valuesTensorName?: (string|null);

    /** CooSparse indicesTensorName */
    // CG305: TensorInfo.CooSparse.indicesTensorName (api.proto field 2). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    indicesTensorName?: (string|null);

    /** CooSparse denseShapeTensorName */
    // CG305: TensorInfo.CooSparse.denseShapeTensorName (api.proto field 3). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    denseShapeTensorName?: (string|null);
  }
}

/** Properties of a SignatureDef. */
export declare interface ISignatureDef {
  /** SignatureDef inputs */
  // CG305: SignatureDef.inputs (api.proto field 1). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  inputs?: ({[k: string]: ITensorInfo}|null);

  /** SignatureDef outputs */
  // CG305: SignatureDef.outputs (api.proto field 2). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  outputs?: ({[k: string]: ITensorInfo}|null);

  /** SignatureDef methodName */
  // CG305: SignatureDef.methodName (api.proto field 3). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  methodName?: (string|null);
}

/** Properties of an AssetFileDef. */
export declare interface IAssetFileDef {
  /** AssetFileDef tensorInfo */
  // CG305: AssetFileDef.tensorInfo (api.proto field 1). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  tensorInfo?: (ITensorInfo|null);

  /** AssetFileDef filename */
  // CG305: AssetFileDef.filename (api.proto field 2). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  filename?: (string|null);
}

/** Properties of an OpDef. */
export declare interface IOpDef {
  /** OpDef name */
  // CG305: OpDef.name (api.proto field 1). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  name?: (string|null);

  /** OpDef inputArg */
  // CG305: OpDef.inputArg (api.proto field 2). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  inputArg?: (OpDef.IArgDef[]|null);

  /** OpDef outputArg */
  // CG305: OpDef.outputArg (api.proto field 3). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  outputArg?: (OpDef.IArgDef[]|null);

  /** OpDef attr */
  // CG305: OpDef.attr (api.proto field 4). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  attr?: (OpDef.IAttrDef[]|null);

  /** OpDef deprecation */
  // CG305: OpDef.deprecation (api.proto field 8). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  deprecation?: (OpDef.IOpDeprecation|null);

  /** OpDef summary */
  // CG305: OpDef.summary (api.proto field 5). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  summary?: (string|null);

  /** OpDef description */
  // CG305: OpDef.description (api.proto field 6). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  description?: (string|null);

  /** OpDef isCommutative */
  // CG305: OpDef.isCommutative (api.proto field 18). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  isCommutative?: (boolean|null);

  /** OpDef isAggregate */
  // CG305: OpDef.isAggregate (api.proto field 16). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  isAggregate?: (boolean|null);

  /** OpDef isStateful */
  // CG305: OpDef.isStateful (api.proto field 17). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  isStateful?: (boolean|null);

  /** OpDef allowsUninitializedInput */
  // CG305: OpDef.allowsUninitializedInput (api.proto field 19). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  allowsUninitializedInput?: (boolean|null);
}

export namespace OpDef {
  /** Properties of an ArgDef. */
  export declare interface IArgDef {
    /** ArgDef name */
    // CG305: OpDef.ArgDef.name (api.proto field 1). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    name?: (string|null);

    /** ArgDef description */
    // CG305: OpDef.ArgDef.description (api.proto field 2). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    description?: (string|null);

    /** ArgDef type */
    // CG305: OpDef.ArgDef.type (api.proto field 3). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    type?: (DataType|null);

    /** ArgDef typeAttr */
    // CG305: OpDef.ArgDef.typeAttr (api.proto field 4). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    typeAttr?: (string|null);

    /** ArgDef numberAttr */
    // CG305: OpDef.ArgDef.numberAttr (api.proto field 5). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    numberAttr?: (string|null);

    /** ArgDef typeListAttr */
    // CG305: OpDef.ArgDef.typeListAttr (api.proto field 6). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    typeListAttr?: (string|null);

    /** ArgDef isRef */
    // CG305: OpDef.ArgDef.isRef (api.proto field 16). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    isRef?: (boolean|null);
  }

  /** Properties of an AttrDef. */
  export declare interface IAttrDef {
    /** AttrDef name */
    // CG305: OpDef.AttrDef.name (api.proto field 1). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    name?: (string|null);

    /** AttrDef type */
    // CG305: OpDef.AttrDef.type (api.proto field 2). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    type?: (string|null);

    /** AttrDef defaultValue */
    // CG305: OpDef.AttrDef.defaultValue (api.proto field 3). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    defaultValue?: (IAttrValue|null);

    /** AttrDef description */
    // CG305: OpDef.AttrDef.description (api.proto field 4). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    description?: (string|null);

    /** AttrDef hasMinimum */
    // CG305: OpDef.AttrDef.hasMinimum (api.proto field 5). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    hasMinimum?: (boolean|null);

    /** AttrDef minimum */
    // CG305: OpDef.AttrDef.minimum (api.proto field 6). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    minimum?: (number|string|null);

    /** AttrDef allowedValues */
    // CG305: OpDef.AttrDef.allowedValues (api.proto field 7). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    allowedValues?: (IAttrValue|null);
  }

  /** Properties of an OpDeprecation. */
  export declare interface IOpDeprecation {
    /** OpDeprecation version */
    // CG305: OpDef.OpDeprecation.version (api.proto field 1). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    version?: (number|null);

    /** OpDeprecation explanation */
    // CG305: OpDef.OpDeprecation.explanation (api.proto field 2). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    explanation?: (string|null);
  }
}

/** Properties of an OpList. */
export declare interface IOpList {
  /** OpList op */
  // CG305: OpList.op (api.proto field 1). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  op?: (IOpDef[]|null);
}

/** Properties of a MetaGraphDef. */
export declare interface IMetaGraphDef {
  /** MetaGraphDef metaInfoDef */
  // CG305: MetaGraphDef.metaInfoDef (api.proto field 1). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  metaInfoDef?: (MetaGraphDef.IMetaInfoDef|null);

  /** MetaGraphDef graphDef */
  // CG305: MetaGraphDef.graphDef (api.proto field 2). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  graphDef?: (IGraphDef|null);

  /** MetaGraphDef saverDef */
  // CG305: MetaGraphDef.saverDef (api.proto field 3). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  saverDef?: (ISaverDef|null);

  /** MetaGraphDef collectionDef */
  // CG305: MetaGraphDef.collectionDef (api.proto field 4). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  collectionDef?: ({[k: string]: ICollectionDef}|null);

  /** MetaGraphDef signatureDef */
  // CG305: MetaGraphDef.signatureDef (api.proto field 5). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  signatureDef?: ({[k: string]: ISignatureDef}|null);

  /** MetaGraphDef assetFileDef */
  // CG305: MetaGraphDef.assetFileDef (api.proto field 6). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  assetFileDef?: (IAssetFileDef[]|null);
}

export namespace MetaGraphDef {
  /** Properties of a MetaInfoDef. */
  export declare interface IMetaInfoDef {
    /** MetaInfoDef metaGraphVersion */
    // CG305: MetaGraphDef.MetaInfoDef.metaGraphVersion (api.proto field 1). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    metaGraphVersion?: (string|null);

    /** MetaInfoDef strippedOpList */
    // CG305: MetaGraphDef.MetaInfoDef.strippedOpList (api.proto field 2). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    strippedOpList?: (IOpList|null);

    /** MetaInfoDef anyInfo */
    // CG305: MetaGraphDef.MetaInfoDef.anyInfo (api.proto field 3). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    anyInfo?: (IAny|null);

    /** MetaInfoDef tags */
    // CG305: MetaGraphDef.MetaInfoDef.tags (api.proto field 4). An empty protobuf collection can be absent from JSON.
    //@cg-expect-error
    tags?: (string[]|null);

    /** MetaInfoDef tensorflowVersion */
    // CG305: MetaGraphDef.MetaInfoDef.tensorflowVersion (api.proto field 5). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    tensorflowVersion?: (string|null);

    /** MetaInfoDef tensorflowGitVersion */
    // CG305: MetaGraphDef.MetaInfoDef.tensorflowGitVersion (api.proto field 6). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
    //@cg-expect-error
    tensorflowGitVersion?: (string|null);
  }
}

/** Properties of a SavedModel. */
export declare interface ISavedModel {
  /** SavedModel savedModelSchemaVersion */
  // CG305: SavedModel.savedModelSchemaVersion (api.proto field 1). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  savedModelSchemaVersion?: (number|string|null);

  /** SavedModel metaGraphs */
  // CG305: SavedModel.metaGraphs (api.proto field 2). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  metaGraphs?: (IMetaGraphDef[]|null);
}

/** Properties of a FunctionDefLibrary. */
export declare interface IFunctionDefLibrary {
  /** FunctionDefLibrary function */
  // CG305: FunctionDefLibrary.function (api.proto field 1). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  'function'?: (IFunctionDef[]|null);

  /** FunctionDefLibrary gradient */
  // CG305: FunctionDefLibrary.gradient (api.proto field 2). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  gradient?: (IGradientDef[]|null);
}

/** Properties of a FunctionDef. */
export declare interface IFunctionDef {
  /** FunctionDef signature */
  // CG305: FunctionDef.signature (api.proto field 1). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  signature?: (IOpDef|null);

  /** FunctionDef attr */
  // CG305: FunctionDef.attr (api.proto field 5). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  attr?: ({[k: string]: IAttrValue}|null);

  /** FunctionDef nodeDef */
  // CG305: FunctionDef.nodeDef (api.proto field 3). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  nodeDef?: (INodeDef[]|null);

  /** FunctionDef ret */
  // CG305: FunctionDef.ret (api.proto field 4). An empty protobuf collection can be absent from JSON.
  //@cg-expect-error
  ret?: ({[k: string]: string}|null);
}

/** Properties of a GradientDef. */
export declare interface IGradientDef {
  /** GradientDef functionName */
  // CG305: GradientDef.functionName (api.proto field 1). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  functionName?: (string|null);

  /** GradientDef gradientFunc */
  // CG305: GradientDef.gradientFunc (api.proto field 2). Proto3 permits this field to be absent on the wire; consumers resolve presence or defaults.
  //@cg-expect-error
  gradientFunc?: (string|null);
}
