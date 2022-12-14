/**
 * Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

import {BaseOptions as BaseOptionsProto} from '../../../tasks/cc/core/proto/base_options_pb';
import {TaskRunner} from '../../../tasks/web/core/task_runner';
import {createSpyWasmModule, SpyWasmModule} from '../../../tasks/web/core/task_runner_test_utils';
import {ErrorListener} from '../../../web/graph_runner/graph_runner';

import {GraphRunnerImageLib} from './task_runner';

class TaskRunnerFake extends TaskRunner {
  protected baseOptions = new BaseOptionsProto();
  private errorListener: ErrorListener|undefined;
  private errors: string[] = [];

  static createFake(): TaskRunnerFake {
    const wasmModule = createSpyWasmModule();
    return new TaskRunnerFake(wasmModule);
  }

  constructor(wasmModuleFake: SpyWasmModule) {
    super(
        wasmModuleFake, /* glCanvas= */ null,
        jasmine.createSpyObj<GraphRunnerImageLib>([
          'setAutoRenderToScreen', 'setGraph', 'finishProcessing',
          'registerModelResourcesGraphService', 'attachErrorListener'
        ]));
    const graphRunner = this.graphRunner as jasmine.SpyObj<GraphRunnerImageLib>;
    expect(graphRunner.registerModelResourcesGraphService).toHaveBeenCalled();
    expect(graphRunner.setAutoRenderToScreen).toHaveBeenCalled();
    graphRunner.attachErrorListener.and.callFake(listener => {
      this.errorListener = listener;
    });
    graphRunner.setGraph.and.callFake(() => {
      this.throwErrors();
    });
    graphRunner.finishProcessing.and.callFake(() => {
      this.throwErrors();
    });
  }

  enqueueError(message: string): void {
    this.errors.push(message);
  }

  override finishProcessing(): void {
    super.finishProcessing();
  }

  override setGraph(graphData: Uint8Array, isBinary: boolean): void {
    super.setGraph(graphData, isBinary);
  }

  private throwErrors(): void {
    expect(this.errorListener).toBeDefined();
    for (const error of this.errors) {
      this.errorListener!(/* errorCode= */ -1, error);
    }
    this.errors = [];
  }
}

describe('TaskRunner', () => {
  it('handles errors during graph update', () => {
    const taskRunner = TaskRunnerFake.createFake();
    taskRunner.enqueueError('Test error');

    expect(() => {
      taskRunner.setGraph(new Uint8Array(0), /* isBinary= */ true);
    }).toThrowError('Test error');
  });

  it('handles errors during graph execution', () => {
    const taskRunner = TaskRunnerFake.createFake();
    taskRunner.setGraph(new Uint8Array(0), /* isBinary= */ true);

    taskRunner.enqueueError('Test error');

    expect(() => {
      taskRunner.finishProcessing();
    }).toThrowError('Test error');
  });

  it('can handle multiple errors', () => {
    const taskRunner = TaskRunnerFake.createFake();
    taskRunner.enqueueError('Test error 1');
    taskRunner.enqueueError('Test error 2');

    expect(() => {
      taskRunner.setGraph(new Uint8Array(0), /* isBinary= */ true);
    }).toThrowError(/Test error 1, Test error 2/);
  });
});
