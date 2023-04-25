/**
 * Copyright 2022 The MediaPipe Authors.
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

// Placeholder for internal dependency on encodeByteArray
import {InferenceCalculatorOptions} from '../../../calculators/tensor/inference_calculator_pb';
import {BaseOptions as BaseOptionsProto} from '../../../tasks/cc/core/proto/base_options_pb';
import {TaskRunner} from '../../../tasks/web/core/task_runner';
import {ErrorListener} from '../../../web/graph_runner/graph_runner';
// Placeholder for internal dependency on trusted resource URL builder

import {CachedGraphRunner} from './task_runner';
import {TaskRunnerOptions} from './task_runner_options.d';

class TaskRunnerFake extends TaskRunner {
  private errorListener: ErrorListener|undefined;
  private errors: string[] = [];

  baseOptions = new BaseOptionsProto();

  static createFake(): TaskRunnerFake {
    return new TaskRunnerFake();
  }

  constructor() {
    super(jasmine.createSpyObj<CachedGraphRunner>([
      'setAutoRenderToScreen', 'setGraph', 'finishProcessing',
      'registerModelResourcesGraphService', 'attachErrorListener'
    ]));
    const graphRunner = this.graphRunner as jasmine.SpyObj<CachedGraphRunner>;
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

  override refreshGraph(): void {}

  override setGraph(graphData: Uint8Array, isBinary: boolean): void {
    super.setGraph(graphData, isBinary);
    expect(this.graphRunner.registerModelResourcesGraphService)
        .toHaveBeenCalled();
  }

  setOptions(options: TaskRunnerOptions): Promise<void> {
    return this.applyOptions(options);
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
  const mockBytes = new Uint8Array([0, 1, 2, 3]);
  const mockBytesResult = {
    modelAsset: {
      fileContent: Buffer.from(mockBytes).toString('base64'),
      fileName: undefined,
      fileDescriptorMeta: undefined,
      filePointerMeta: undefined,
    },
    useStreamMode: false,
    acceleration: {
      xnnpack: undefined,
      gpu: undefined,
      tflite: {},
      nnapi: undefined,
    },
  };
  const mockBytesResultWithGpuDelegate = {
    ...mockBytesResult,
    acceleration: {
      xnnpack: undefined,
      gpu: {
        useAdvancedGpuApi: false,
        api: InferenceCalculatorOptions.Delegate.Gpu.Api.ANY,
        allowPrecisionLoss: true,
        cachedKernelPath: undefined,
        serializedModelDir: undefined,
        modelToken: undefined,
        usage: InferenceCalculatorOptions.Delegate.Gpu.InferenceUsage
                   .SUSTAINED_SPEED,
      },
      tflite: undefined,
      nnapi: undefined,
    },
  };

  let fetchSpy: jasmine.Spy;
  let taskRunner: TaskRunnerFake;
  let fetchStatus: number;

  beforeEach(() => {
    fetchStatus = 200;
    fetchSpy = jasmine.createSpy().and.callFake(async url => {
      return {
        arrayBuffer: () => mockBytes.buffer,
        ok: fetchStatus === 200,
        status: fetchStatus,
      } as unknown as Response;
    });
    global.fetch = fetchSpy;

    taskRunner = TaskRunnerFake.createFake();
  });

  it('handles errors during graph update', () => {
    taskRunner.enqueueError('Test error');

    expect(() => {
      taskRunner.setGraph(new Uint8Array(0), /* isBinary= */ true);
    }).toThrowError('Test error');
  });

  it('handles errors during graph execution', () => {
    taskRunner.setGraph(new Uint8Array(0), /* isBinary= */ true);

    taskRunner.enqueueError('Test error');

    expect(() => {
      taskRunner.finishProcessing();
    }).toThrowError('Test error');
  });

  it('can handle multiple errors', () => {
    taskRunner.enqueueError('Test error 1');
    taskRunner.enqueueError('Test error 2');

    expect(() => {
      taskRunner.setGraph(new Uint8Array(0), /* isBinary= */ true);
    }).toThrowError(/Test error 1, Test error 2/);
  });

  it('clears errors once thrown', () => {
    taskRunner.enqueueError('Test error');

    expect(() => {
      taskRunner.setGraph(new Uint8Array(0), /* isBinary= */ true);
    }).toThrowError(/Test error/);

    expect(() => {
      taskRunner.setGraph(new Uint8Array(0), /* isBinary= */ true);
    }).not.toThrow();
  });

  it('verifies that at least one model asset option is provided', () => {
    expect(() => {
      taskRunner.setOptions({});
    })
        .toThrowError(
            /Either baseOptions.modelAssetPath or baseOptions.modelAssetBuffer must be set/);
  });

  it('verifies that no more than one model asset option is provided', () => {
    expect(() => {
      taskRunner.setOptions({
        baseOptions: {
          modelAssetPath: `foo`,
          modelAssetBuffer: new Uint8Array([])
        }
      });
    })
        .toThrowError(
            /Cannot set both baseOptions.modelAssetPath and baseOptions.modelAssetBuffer/);
  });

  it('doesn\'t require model once it is configured', async () => {
    await taskRunner.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array(mockBytes)}});
    expect(() => {
      taskRunner.setOptions({});
    }).not.toThrowError();
  });

  it('downloads model', async () => {
    await taskRunner.setOptions(
        {baseOptions: {modelAssetPath: `foo`}});

    expect(fetchSpy).toHaveBeenCalled();
    expect(taskRunner.baseOptions.toObject()).toEqual(mockBytesResult);
  });

  it('does not download model when bytes are provided', async () => {
    await taskRunner.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array(mockBytes)}});

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(taskRunner.baseOptions.toObject()).toEqual(mockBytesResult);
  });

  it('changes model synchronously when bytes are provided', () => {
    const resolvedPromise = taskRunner.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array(mockBytes)}});

    // Check that the change has been applied even though we do not await the
    // above Promise
    expect(taskRunner.baseOptions.toObject()).toEqual(mockBytesResult);
    return resolvedPromise;
  });

  it('returns custom error if model download failed', () => {
    fetchStatus = 404;
    return expectAsync(taskRunner.setOptions({
             baseOptions: {modelAssetPath: `notfound.tflite`}
           }))
        .toBeRejectedWithError('Failed to fetch model: notfound.tflite (404)');
  });

  it('can enable CPU delegate', async () => {
    await taskRunner.setOptions({
      baseOptions: {
        modelAssetBuffer: new Uint8Array(mockBytes),
        delegate: 'CPU',
      }
    });
    expect(taskRunner.baseOptions.toObject()).toEqual(mockBytesResult);
  });

  it('can enable GPU delegate', async () => {
    await taskRunner.setOptions({
      baseOptions: {
        modelAssetBuffer: new Uint8Array(mockBytes),
        delegate: 'GPU',
      }
    });
    expect(taskRunner.baseOptions.toObject())
        .toEqual(mockBytesResultWithGpuDelegate);
  });

  it('can reset delegate', async () => {
    await taskRunner.setOptions({
      baseOptions: {
        modelAssetBuffer: new Uint8Array(mockBytes),
        delegate: 'GPU',
      }
    });
    // Clear delegate
    await taskRunner.setOptions({baseOptions: {delegate: undefined}});
    expect(taskRunner.baseOptions.toObject()).toEqual(mockBytesResult);
  });

  it('keeps delegate if not provided', async () => {
    await taskRunner.setOptions({
      baseOptions: {
        modelAssetBuffer: new Uint8Array(mockBytes),
        delegate: 'GPU',
      }
    });
    await taskRunner.setOptions({baseOptions: {}});
    expect(taskRunner.baseOptions.toObject())
        .toEqual(mockBytesResultWithGpuDelegate);
  });
});
