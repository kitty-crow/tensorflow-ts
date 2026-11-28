/**
 * @license
 * Copyright 2022 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/**
 * LruCache: A mapping from the String to T. If the number of the entries is
 * exceeding the `maxEntries`, the LruCache will delete the least recently
 * used entry.
 */

export class LruCache<T> {
  private cache: Map<string, T>;
  private maxEntries: number;

  constructor(maxEntries?: number) {
    this.maxEntries = maxEntries || 100;
    this.cache = new Map<string, T>();
  }

  /**
   * Get the entry for the key and mark it as used recently.
   */
  public get(key: string): T {
    let entry: T;
    if (this.cache.has(key)) {
      entry = this.cache.get(key);
      this.cache.delete(key);
      this.cache.set(key, entry);
    }
    return entry;
  }

  /**
   * Put the entry into the cache. If the key already existed, mark the key as
   * used recently.
   */
  public put(key: string, value: T): void {
    const alreadyPresent = this.cache.has(key);
    if (alreadyPresent) {
      this.cache.delete(key);
    }
    if (!alreadyPresent && this.cache.size >= this.maxEntries) {
      this.evictOldest();
    }
    this.cache.set(key, value);
  }

  /** Evict the oldest present key without using an unchecked iterator result. */
  private evictOldest(): void {
    for (const key of this.cache.keys()) {
      this.cache.delete(key);
      break;
    }
  }

  /**
   * Get the MaxEntries of the cache.
   */
  public getMaxEntries(): number {
    return this.maxEntries;
  }

  /**
   * Set the MaxEntries of the cache. If the maxEntries is decreased, reduce
   * entries in the cache.
   */
  public setMaxEntries(maxEntries: number): void {
    if (maxEntries < 0) {
      throw new Error(
          `The maxEntries of LRU caches must be at least 0, but got ${
              maxEntries}.`);
    }

    if (this.maxEntries > maxEntries) {
      for (let i = 0; i < this.maxEntries - maxEntries; i++) {
        this.evictOldest();
      }
    }

    this.maxEntries = maxEntries;
  }
}
