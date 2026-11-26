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

/** Geometry shared by the two resize gradient shaders. The uniforms remain
 * operation-specific because nearest-neighbour does not use the forward scale.
 */
export function resizeBackpropGeometry(
    inputShape: [number, number, number, number],
    outputShape: [number, number, number, number], alignCorners: boolean) {
  const [, xHeight, xWidth] = inputShape;
  const [, yHeight, yWidth] = outputShape;

  const effectiveXSize: [number, number] = [
    (alignCorners && yHeight > 1) ? xHeight - 1 : xHeight,
    (alignCorners && yWidth > 1) ? xWidth - 1 : xWidth
  ];

  const effectiveYSize: [number, number] = [
    (alignCorners && yHeight > 1) ? yHeight - 1 : yHeight,
    (alignCorners && yWidth > 1) ? yWidth - 1 : yWidth
  ];

  const heightScale = effectiveXSize[0] / effectiveYSize[0];
  const widthScale = effectiveXSize[1] / effectiveYSize[1];

  const invHeightScale = 1 / heightScale;
  const invWidthScale = 1 / widthScale;

  // This defines the size of the window of values around a particular
  // index in dy that we want to search for contributions to dx.
  const winHeight = (Math.ceil(invHeightScale) * 2) + 2;
  const winWidth = (Math.ceil(invWidthScale) * 2) + 2;

  return {effectiveXSize, effectiveYSize, heightScale, widthScale,
    invHeightScale, invWidthScale, winHeight, winWidth};
}
