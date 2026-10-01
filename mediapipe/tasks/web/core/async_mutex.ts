/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** A mutex for async code.
 *
 * This class protects a region of code from re-entrancy. This isn't necessary
 * in most cases, but async emscripten code must not be called re-entrantly,
 * and there's no standard solution to ensure this.
 */
export class AsyncMutex {
  private locked = false;
  private readonly queue: Array<() => void> = [];

  private runNext(): void {
    if (!this.locked && this.queue.length > 0) {
      this.locked = true;
      this.queue.shift()!();
    }
  }

  /**
   * Acquires the lock, executes the given function, then releases the lock,
   * returning (asynchronously) the value returned by the function.
   */
  async runExclusive<T>(f: () => Promise<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.queue.push(() => {
        f()
          .then(resolve, reject)
          .finally(() => {
            this.locked = false;
            this.runNext();
          });
      });

      // If the lock isn't held already, this will immediately run the function
      // we just pushed into the queue. Otherwise it will exit, and whoever
      // currently holds the lock will run our function when they release the
      // lock.
      this.runNext();
    });
  }
}
