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
 * =============================================================================
 */

export class BufferManager {
  private numUsedBuffers = 0;
  private numFreeBuffers = 0;
  private freeBuffers: Map<string, GPUBuffer[]> = new Map<string, GPUBuffer[]>();
  private usedBuffers: Map<string, GPUBuffer[]> = new Map<string, GPUBuffer[]>();

  public numBytesUsed = 0;
  public numBytesAllocated = 0;

  constructor(private device: GPUDevice) {}

  acquireBuffer(
      size: number, usage: GPUBufferUsageFlags, mappedAtCreation = false,
      reuse = true): GPUBuffer {
    const key = getBufferKey(size, usage);
    const free = this.freeBuffers.get(key);
    let buffer: GPUBuffer;
    const recycled = reuse && free != null ? free.pop() : null;

    if (recycled != null) {
      buffer = recycled;
      this.numFreeBuffers--;
    } else {
      buffer = this.device.createBuffer({size, usage, mappedAtCreation});
      this.numBytesAllocated += size;
    }

    let used = this.usedBuffers.get(key);
    if (used == null) {
      used = [];
      this.usedBuffers.set(key, used);
    }
    used.push(buffer);
    this.numUsedBuffers++;
    this.numBytesUsed += size;

    return buffer;
  }

  releaseBuffer(buffer: GPUBuffer, reuse = true) {
    if (this.usedBuffers.size === 0 && this.freeBuffers.size === 0) {
      return;
    }

    const size = buffer.size;
    const usage = buffer.usage;
    const key = getBufferKey(size, usage);
    const used = this.usedBuffers.get(key);
    if (used == null) {
      throw new Error('Cannot find the buffer in buffer manager');
    }
    const index = used.indexOf(buffer);
    if (index < 0) {
      throw new Error('Cannot find the buffer in buffer manager');
    }
    const last = used.pop();
    if (last == null) {
      throw new Error('Buffer manager has an inconsistent used buffer list');
    }
    if (index < used.length) {
      used[index] = last;
    }
    this.numUsedBuffers--;
    this.numBytesUsed -= size;

    if (!reuse) {
      buffer.destroy();
      this.numBytesAllocated -= size;
      return;
    }
    let free = this.freeBuffers.get(key);
    if (free == null) {
      free = [];
      this.freeBuffers.set(key, free);
    }
    free.push(buffer);
    this.numFreeBuffers++;
  }

  getNumUsedBuffers(): number {
    return this.numUsedBuffers;
  }

  getNumFreeBuffers(): number {
    return this.numFreeBuffers;
  }

  dispose() {
    this.freeBuffers.forEach((buffers, key) => {
      buffers.forEach(buffer => {
        buffer.destroy();
      });
    });

    this.usedBuffers.forEach((buffers, key) => {
      buffers.forEach(buffer => {
        buffer.destroy();
      });
    });

    this.freeBuffers = new Map<string, GPUBuffer[]>();
    this.usedBuffers = new Map<string, GPUBuffer[]>();
    this.numUsedBuffers = 0;
    this.numFreeBuffers = 0;
    this.numBytesUsed = 0;
    this.numBytesAllocated = 0;
  }
}

function getBufferKey(size: number, usage: GPUBufferUsageFlags) {
  return `${size}_${usage}`;
}
