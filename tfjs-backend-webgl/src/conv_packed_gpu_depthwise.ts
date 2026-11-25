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

import {backend_util} from '@tensorflow/tfjs-core';

import {GPGPUProgram, useShapeUniforms} from './gpgpu_math';
import {selectActivationSource} from './activation_source';
import {packedConvColumnSource} from './packed_conv_column';

export class DepthwiseConvPacked2DProgram implements GPGPUProgram {
  variableNames = ['x', 'W'];
  packedInputs = true;
  packedOutput = true;
  outputShape: number[];
  userCode: string;
  enableShapeUniforms: boolean;
  customUniforms = [
    {name: 'pads', type: 'ivec2' as const },
    {name: 'strides', type: 'ivec2' as const },
    {name: 'dilations', type: 'ivec2' as const },
    {name: 'inDims', type: 'ivec2' as const },
  ];

  constructor(
      convInfo: backend_util.Conv2DInfo, addBias = false,
      activation: string = null, hasPreluActivation = false,
      hasLeakyReluAlpha = false) {
    this.outputShape = convInfo.outShape;
    this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);
    const channelMul = convInfo.outChannels / convInfo.inChannels;
    const padLeft = convInfo.padInfo.left;
    const strideWidth = convInfo.strideWidth;
    const dilationWidth = convInfo.dilationWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const texelsAcross = filterWidth;

    let mainLoop = `
      int xR; int xC; int xCOffset;
      vec4 wTexel; vec4 previous; vec4 final;`;

    for (let c = 0; c < filterWidth; c++) {
      mainLoop += `
          vec4 xTexelC${c * 2};
          int xTexelC${c * 2}Ready;
          vec4 xTexelC${c * 2 + 1};
          int xTexelC${c * 2 + 1}Ready;
          vec4 xC${c};`;
    }

    /**
     * This vectorized implementation works by gathering the values needed for
     * each output channel's dot product into vec4's and then multiplying them
     * all together (this happens in the final double for-loop below). Most of
     * the main loop consists of constructing these vec4's with the minimum
     * number of texture2D calls, which means making use of all four returned
     * values from a texture2D call at once.
     */
    mainLoop += `
    for (int r = 0; r < ${filterHeight}; r++) {
      `;
    for (let c = 0; c < filterWidth; c++) {
      mainLoop += `
          xTexelC${c * 2} = vec4(0.0);
          xTexelC${c * 2}Ready = 0;
          xTexelC${c * 2 + 1} = vec4(0.0);
          xTexelC${c * 2 + 1}Ready = 0;
          xC${c} = vec4(0.0);`;
    }
    mainLoop += `
        xR = xRCorner + r * dilations[0];
        if (xR >=0 && xR < inDims[0]) {
      `;

    for (let texelC = 0; texelC < (texelsAcross + 1) / 2; texelC++) {
      const colIndex = texelC * 2;

      mainLoop += `
          xC = xCCorner + ${colIndex * dilationWidth};
          `;

      mainLoop += packedConvColumnSource(
          colIndex, filterWidth, strideWidth, padLeft, dilationWidth);

      // localize the dotProd accumulation within the loop, the theory is for
      // GPU with limited cache, accumulate sum across large amount of
      // veriables will cause lots of cache misses. (i.e. 5x5 filter will have
      // 50 variables)
      if (!(colIndex < filterWidth)) continue;
      {
        mainLoop += `
            wTexel = getW(r, ${colIndex}, d1, q);
            dotProd += xC${colIndex} * vec4(wTexel.xz, wTexel.xz);
          `;

        if (colIndex + 1 < filterWidth) {
          mainLoop += `
              wTexel = getW(r, ${colIndex + 1}, d1, q);
              dotProd += xC${colIndex + 1} * vec4(wTexel.xz, wTexel.xz);
            `;
        }
      }
    }
    mainLoop += `
    }
  `;
    mainLoop += `
      }
    `;

    const activationSnippet = selectActivationSource(
        activation, hasPreluActivation, hasLeakyReluAlpha,
        () => `vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${activation}
        }`,
        () => `vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${activation}
        }`,
        () => `vec4 activation(vec4 x) {
          ${activation}
        }`);
    const applyActivationSnippet = activation ? `result = activation(result);` : '';

    const addBiasSnippet = addBias ? 'result += getBiasAtOutCoords();' : '';
    if (addBias) {
      this.variableNames.push('bias');
    }

    if (hasPreluActivation) {
      this.variableNames.push('preluActivationWeights');
    }
    if (hasLeakyReluAlpha) {
      this.variableNames.push('leakyreluAlpha');
    }

    this.userCode = `
      ${activationSnippet}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${channelMul};
        int q = d2 - d1 * ${channelMul};
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
        vec4 dotProd = vec4(0.000000000000001);

        ${mainLoop}

        vec4 result = dotProd - vec4(0.000000000000001);
        ${addBiasSnippet}
        ${applyActivationSnippet}
        setOutput(result);
      }
    `;
  }
}
