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

import {util} from '@tensorflow/tfjs-core';

/** Generate the first column of a stride-one packed convolution sample. */
function strideOneFirstColumn(
    colIndex: number, padLeft: number, dilationWidth: number): string {
  let mainLoop = '';
  // Padding is even, so xRC corresponds to a single texel.
  if (padLeft % 2 !== 1) {
    return `
                if (xC >= 0 && xC < inDims[1] && xTexelC${colIndex}Ready == 0) {
                  xTexelC${colIndex} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${colIndex}.zw = vec2(0.0);
                  }
                  xTexelC${colIndex}Ready = 1;
                }

                xC${colIndex} = xTexelC${colIndex};
                `;
  }
  // If padding is odd, the outer texels have to be composed.
  // TODO: Ensure vec4 previous does not result in redundant sample,
  // and avoid setting xTexelRC's that exceed the boundary in the
  // first place rather than resetting them to vec4(0)).

  // To compute xCOffset:
  // - If padding is odd, we must add 1 to ensure we ask for an
  // even-numbered row.
  // - We subtract 2 to access the previous texel.

  mainLoop += `
                xCOffset = xC + 1;
                if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${
                colIndex}Ready == 0) {
                  xTexelC${colIndex} = getX(batch, xR, xCOffset, d1);

                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${colIndex}.zw = vec2(0.0);
                  }
                  xTexelC${colIndex}Ready = 1;
                }
              `;
  // This texel has been read in previous iteration if the dilation
  // is 1.
  if (dilationWidth === 1 && colIndex > 0) {
    mainLoop += `
                xC${colIndex} = vec4(xTexelC${colIndex - 2}.zw, xTexelC${
                  colIndex}.xy);
                `;
  } else {
    mainLoop += `
                  xCOffset = xC + 1 - 2;

                  if (xCOffset >= 0 && xCOffset < inDims[1]) {
                    previous = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      previous.zw = vec2(0.0);
                    }

                    xC${colIndex} = vec4(previous.zw, xTexelC${colIndex}.xy);
                  } else {
                    xC${colIndex} = vec4(0.0, 0.0, xTexelC${colIndex}.xy);
                  }
                  `;
  }
  return mainLoop;
}

/** Generate the second column when two adjacent outputs share texels. */
function strideOneSecondColumn(
    colIndex: number, padLeft: number, dilationWidth: number): string {
  let mainLoop = '';
  // If dilation is even, the second entry should match the first
  // (either both are composed or both are single samples). But if
  // dilation is odd, then the second entry should be the opposite
  // of the first (if the first is composed, the second is a single
  // sample, and vice versa.)

  const nextTexelOffset = padLeft % 2 === 0 ?
      util.nearestLargerEven(dilationWidth) :
      dilationWidth;

  const composed = (dilationWidth % 2 === 0 && padLeft % 2 === 1) ||
      (dilationWidth % 2 !== 0 && padLeft % 2 !== 1);
  // If dilation is 1 and padding is odd, we have already read the
  // texel when constructing the previous x value. Here we can
  // simply skip the texture read.
  if (!composed && nextTexelOffset === 1) {
    return `
                    xC${colIndex + 1} = xTexelC${colIndex};
                    `;
  }
  if (!composed) {
    return `
                    xCOffset = xC + ${nextTexelOffset};

                    if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${
                    colIndex + 1}Ready == 0) {
                      xTexelC${colIndex + 1} = getX(batch, xR, xCOffset, d1);
                      if (xCOffset + 1 >= inDims[1]) {
                        xTexelC${colIndex + 1}.zw = vec2(0.0);
                      }
                      xTexelC${colIndex + 1}Ready = 1;
                    }

                    xC${colIndex + 1} = xTexelC${colIndex + 1};
                    `;
  }
  mainLoop += `
                  xCOffset = xC + imod(pads[1], 2) + ${nextTexelOffset};

                  if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${
                  colIndex + 1}Ready == 0) {
                    xTexelC${colIndex + 1} = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      xTexelC${colIndex + 1}.zw = vec2(0.0);
                    }
                    xTexelC${colIndex + 1}Ready = 1;
                  }
                  `;

  // If dilation > 1 then the xRC's will not be able to share any
  // values, so each xRC will require two unique calls to getX.
  if (dilationWidth > 1) {
    mainLoop += `
                    xCOffset -= 2;
                    if (xCOffset >= 0 && xCOffset < inDims[1]) {
                     previous = getX(batch, xR, xCOffset, d1);
                     xC${colIndex + 1} = vec4(previous.zw, xTexelC${
                       colIndex + 1}.xy);
                    } else {
                     xC${colIndex + 1} = vec4(0.0, 0.0, xTexelC${
                       colIndex + 1}.xy);
                    }
                    `;
  } else {
    mainLoop += `
                    xC${colIndex + 1} = vec4(xTexelC${colIndex}.zw, xTexelC${
                    colIndex + 1}.xy);
                    `;
  }
  return mainLoop;
}

/** Generate stride-two samples for odd padding. */
function strideTwoOddColumns(colIndex: number, filterWidth: number): string {
  let mainLoop = '';
  mainLoop += `
                xCOffset = xC + 1 - strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${
                colIndex}Ready == 0) {
                  xTexelC${colIndex} = getX(batch, xR, xCOffset, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${colIndex}.zw = vec2(0.0);
                  }
                  xTexelC${colIndex}Ready = 1;
                }

                if(xC + 1 >= 0 && xC + 1 < inDims[1] && xTexelC${
                colIndex + 1}Ready == 0) {
                  xTexelC${colIndex + 1} = getX(batch, xR, xC + 1, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xC + 2 >= inDims[1]) {
                    xTexelC${colIndex + 1}.zw = vec2(0.0);
                  }
                  xTexelC${colIndex + 1}Ready = 1;
                }

                xC${colIndex} = vec4(xTexelC${colIndex}.zw, xTexelC${
                colIndex + 1}.zw);
              `;

  if (colIndex + 1 < filterWidth) {
    mainLoop += `
                  final = vec4(0.0);
                  xCOffset = xC + 1 + strides[1];
                  if(xCOffset >= 0 && xCOffset < inDims[1]) {
                    final = getX(batch, xR, xCOffset, d1);
                  }
                  xC${colIndex + 1} = vec4(xTexelC${colIndex + 1}.xy, final.xy);
                `;
  }
  return mainLoop;
}

/** Generate stride-two samples for even padding. */
function strideTwoEvenColumns(colIndex: number, filterWidth: number): string {
  let mainLoop = '';
  mainLoop += `
                if(xC >= 0 && xC < inDims[1] && xTexelC${colIndex}Ready == 0) {
                  xTexelC${colIndex} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${colIndex}.zw = vec2(0.0);
                  }
                  xTexelC${colIndex}Ready = 1;
                }

                xCOffset = xC + strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${
                colIndex + 1}Ready == 0) {
                  xTexelC${colIndex + 1} = getX(batch, xR, xCOffset, d1);
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${colIndex + 1}.zw = vec2(0.);
                  }
                  xTexelC${colIndex + 1}Ready = 1;
                }

                xC${colIndex} = vec4(
                  xTexelC${colIndex}.xy, xTexelC${colIndex + 1}.xy);
              `;

  if (colIndex + 1 < filterWidth) {
    mainLoop += `
                  xC${colIndex + 1} = vec4(xTexelC${colIndex}.zw, xTexelC${
                  colIndex + 1}.zw);
                `;
  }
  return mainLoop;
}

/** Shared texture sampling for packed Conv2D and packed depthwise Conv2D. */
export function packedConvColumnSource(colIndex: number, filterWidth: number,
    strideWidth: number, padLeft: number, dilationWidth: number): string {
  if (!(colIndex < filterWidth)) {
    return '';
  }
  if (strideWidth !== 1) {
    return padLeft % 2 === 1 ? strideTwoOddColumns(colIndex, filterWidth) :
        strideTwoEvenColumns(colIndex, filterWidth);
  }
  let source = strideOneFirstColumn(colIndex, padLeft, dilationWidth);
  if (colIndex + 1 < filterWidth) {
    source += strideOneSecondColumn(colIndex, padLeft, dilationWidth);
  }
  return source;
}
