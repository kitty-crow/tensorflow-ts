/**
 * @license
 * Copyright 2022 Google LLC.
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

import {backend_util} from '@tensorflow/tfjs-core';

interface PoolBackpropUniform {
  type: 'int32'|'float32';
  data: number[];
}

/** Uniform order follows both 2D backprop shaders, including dilation. */
export function pool2DBackpropUniforms(
    info: backend_util.Conv2DInfo): PoolBackpropUniform[] {
  return [
    {type: 'int32', data: [info.strideHeight, info.strideWidth]},
    {
      type: 'int32',
      data: [
        info.effectiveFilterHeight - 1 - info.padInfo.top,
        info.effectiveFilterWidth - 1 - info.padInfo.left
      ]
    },
    {type: 'int32', data: [info.dilationHeight, info.dilationWidth]},
    {
      type: 'int32',
      data: [info.effectiveFilterHeight, info.effectiveFilterWidth]
    },
    {type: 'int32', data: [info.outHeight]},
    {type: 'int32', data: [info.outWidth]}
  ];
}

/** Both 3D backprop shaders use unit dilation and omit that uniform. */
export function pool3DBackpropUniforms(
    info: backend_util.Conv3DInfo): PoolBackpropUniform[] {
  return [
    {
      type: 'int32',
      data: [info.strideDepth, info.strideHeight, info.strideWidth]
    },
    {
      type: 'int32',
      data: [
        info.effectiveFilterDepth - 1 - info.padInfo.front,
        info.effectiveFilterHeight - 1 - info.padInfo.top,
        info.effectiveFilterWidth - 1 - info.padInfo.left
      ]
    },
    {
      type: 'int32',
      data: [
        info.effectiveFilterDepth, info.effectiveFilterHeight,
        info.effectiveFilterWidth
      ]
    },
    {type: 'int32', data: [info.outDepth]},
    {type: 'int32', data: [info.outHeight]},
    {type: 'int32', data: [info.outWidth]}
  ];
}
