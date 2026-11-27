/**
 * @license
 * Copyright 2023 Google LLC.
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

import {DataType} from '@tensorflow/tfjs-core';

import {getMainHeaderString as main, PixelsOpType, WebGPUProgram} from './webgpu_program';
import {FlatDispatchProgram} from './flat_dispatch_program';

export class DrawProgram extends FlatDispatchProgram implements WebGPUProgram {
  uniforms = 'alpha: f32,';
  shaderKey: string;
  type: DataType;
  textureFormat: GPUTextureFormat;
  pixelsOpType = PixelsOpType.DRAW;

  constructor(
      outShape: number[], type: DataType, textureFormat: GPUTextureFormat) {
    super(outShape, [64, 1, 1], ['Image']);
    this.type = type;
    this.textureFormat = textureFormat;
    this.shaderKey = `draw_${type}_${textureFormat}`;
  }

  getUserCode(): string {
    const value = this.type === 'float32' ? 'value' : 'value / 255.0';
    const calculateResult = `
      if (uniforms.numChannels == 1) {
        rgba[0] = ${value};
        rgba[1] = ${value};
        rgba[2] = ${value};
      } else {
        rgba[d] = ${value};
      }`;

    const userCode = `
       @group(0) @binding(0) var outImage : texture_storage_2d<${
        this.textureFormat}, write>;
       ${main('index')} {
         if (index < uniforms.size) {
           var rgba = vec4<f32>(0.0, 0.0, 0.0, uniforms.alpha);
           for (var d = 0; d < uniforms.numChannels; d = d + 1) {
             let value = f32(inBuf[index * uniforms.numChannels + d]);
             ${calculateResult}
           }
           rgba.x = rgba.x * rgba.w;
           rgba.y = rgba.y * rgba.w;
           rgba.z = rgba.z * rgba.w;
           let coords = getCoordsFromIndex(index);
           textureStore(outImage, vec2<i32>(coords.yx), rgba);
         }
       }
      `;
    return userCode;
  }
}
