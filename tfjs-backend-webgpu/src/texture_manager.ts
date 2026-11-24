/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

export class TextureManager {
  private numUsedTextures = 0;
  private numFreeTextures = 0;
  private freeTextures: Map<string, GPUTexture[]> = new Map<string, GPUTexture[]>();
  private usedTextures: Map<string, GPUTexture[]> = new Map<string, GPUTexture[]>();

  public numBytesUsed = 0;
  public numBytesAllocated = 0;

  constructor(private device: GPUDevice) {}

  acquireTexture(
      width: number, height: number, format: GPUTextureFormat,
      usage: GPUTextureUsageFlags): GPUTexture {
    const byteSize = width * height * getBytesPerElement(format);
    const key = getTextureKey(width, height, format, usage);
    const free = this.freeTextures.get(key);
    let texture: GPUTexture;
    const recycled = free == null ? null : free.shift();
    if (recycled != null) {
      texture = recycled;
      this.numFreeTextures--;
    } else {
      texture = this.device.createTexture({
        size: [width, height],
        format,
        usage,
      });
      this.numBytesAllocated += byteSize;
    }

    let used = this.usedTextures.get(key);
    if (used == null) {
      used = [];
      this.usedTextures.set(key, used);
    }
    used.push(texture);
    this.numBytesUsed += byteSize;
    this.numUsedTextures++;
    return texture;
  }

  releaseTexture(texture: GPUTexture) {
    if (this.usedTextures.size === 0 && this.freeTextures.size === 0) {
      return;
    }

    const {width, height, format, usage} = texture;
    const key = getTextureKey(width, height, format, usage);
    const used = this.usedTextures.get(key);
    if (used == null) {
      throw new Error(
          'Cannot release a texture that was never provided by this ' +
          'texture manager');
    }
    const index = used.indexOf(texture);
    if (index < 0) {
      throw new Error(
          'Cannot release a texture that was never provided by this ' +
          'texture manager');
    }
    const byteSize = width * height * getBytesPerElement(format);
    used.splice(index, 1);

    let free = this.freeTextures.get(key);
    if (free == null) {
      free = [];
      this.freeTextures.set(key, free);
    }
    free.push(texture);
    this.numFreeTextures++;
    this.numUsedTextures--;
    this.numBytesUsed -= byteSize;
  }

  getNumUsedTextures(): number {
    return this.numUsedTextures;
  }

  getNumFreeTextures(): number {
    return this.numFreeTextures;
  }

  dispose() {
    this.freeTextures.forEach((textures, key) => {
      textures.forEach(texture => {
        texture.destroy();
      });
    });

    this.usedTextures.forEach((textures, key) => {
      textures.forEach(texture => {
        texture.destroy();
      });
    });

    this.freeTextures = new Map<string, GPUTexture[]>();
    this.usedTextures = new Map<string, GPUTexture[]>();
    this.numUsedTextures = 0;
    this.numFreeTextures = 0;
    this.numBytesUsed = 0;
    this.numBytesAllocated = 0;
  }
}

function getTextureKey(
    width: number, height: number, format: GPUTextureFormat,
    usage: GPUTextureUsageFlags) {
  return `${width}_${height}_${format}_${usage}`;
}

function getBytesPerElement(format: GPUTextureFormat) {
  if (format === 'rgba8unorm') {
    return 16;
  } else {
    throw new Error(`${format} is not supported!`);
  }
}
