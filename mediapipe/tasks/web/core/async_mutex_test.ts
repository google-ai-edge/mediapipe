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

import 'jasmine';

import {AsyncMutex} from './async_mutex';

describe('AsyncMutex', () => {
  it('should execute tasks sequentially, not concurrently', async () => {
    const mutex = new AsyncMutex();

    const executionLog: string[] = [];
    let activeTasks = 0;
    let maxTasks = 0;
    const tasks: Array<Promise<void>> = [];

    // Launch all tasks without awaiting them individually.
    for (let i = 0; i < 5; i++) {
      tasks.push(
        mutex.runExclusive(async () => {
          executionLog.push(`task${i}-enter`);
          activeTasks++;
          maxTasks = Math.max(maxTasks, activeTasks);

          if (activeTasks !== 1) {
            throw new Error(`More than one task active at once`);
          }

          // Simulate minimal async work by yielding to the event loop. This
          // allows other async code to execute, giving our other tasks the
          // chance to run if the lock isn't working correctly.
          await new Promise((resolve) => {
            setTimeout(resolve, 0);
          });

          if (activeTasks !== 1) {
            throw new Error(`More than one task active at once after yielding`);
          }

          activeTasks--;
          executionLog.push(`task${i}-exit`);
        }),
      );
    }

    await Promise.all(tasks);

    expect(maxTasks).toEqual(1);
    expect(activeTasks).toEqual(0);
    expect(executionLog).toEqual([
      'task0-enter',
      'task0-exit',
      'task1-enter',
      'task1-exit',
      'task2-enter',
      'task2-exit',
      'task3-enter',
      'task3-exit',
      'task4-enter',
      'task4-exit',
    ]);
  });

  it('should propagate return values correctly through runExclusive', async () => {
    const mutex = new AsyncMutex();

    const task1 = async () => {
      await Promise.resolve(); // Simulate async work
      return 'result1';
    };
    const task2 = async () => {
      await Promise.resolve(); // Simulate async work
      return 42;
    };

    const results = await Promise.all([
      mutex.runExclusive(task1),
      mutex.runExclusive(task2),
    ]);

    expect(results).toEqual(['result1', 42]);
  });

  it('should propagate rejections correctly and allow subsequent tasks to run', async () => {
    const mutex = new AsyncMutex();

    const executionLog: string[] = [];

    const failingTask = mutex.runExclusive(async () => {
      executionLog.push('task1-enter');
      await Promise.resolve();
      executionLog.push('task1-throws');
      throw new Error('Task 1 deliberately failed');
    });
    const succeedingTask = mutex.runExclusive(async () => {
      executionLog.push('task2-enter');
      await Promise.resolve();
      executionLog.push('task2-exit');
      return 'task2-success';
    });

    await expectAsync(failingTask).toBeRejectedWithError(
      'Task 1 deliberately failed',
    );
    expect(await succeedingTask).toEqual('task2-success');

    expect(executionLog).toEqual([
      'task1-enter',
      'task1-throws',
      'task2-enter',
      'task2-exit',
    ]);
  });

  it('should maintain the lock while the task is blocked', async () => {
    const mutex = new AsyncMutex();

    const executionLog: string[] = [];
    let resolveFirstTask: () => void;

    // Create a promise that we can manually resolve
    const firstTaskPromise = new Promise<void>((resolve) => {
      resolveFirstTask = resolve;
    });

    const p1 = mutex.runExclusive(async () => {
      executionLog.push('task1-enter');

      // This will block until resolveFirstTask() is called
      await firstTaskPromise;

      executionLog.push('task1-exit');
    });

    const p2 = mutex.runExclusive(async () => {
      executionLog.push('task2');
    });

    // At this point task1 should have started and acquired the lock, and task2
    // should be queued and waiting.
    expect(executionLog).toEqual(['task1-enter']);

    // Now, complete the first task
    resolveFirstTask!();

    // Wait for both tasks to complete
    await Promise.all([p1, p2]);

    expect(executionLog).toEqual(['task1-enter', 'task1-exit', 'task2']);
  });
});
