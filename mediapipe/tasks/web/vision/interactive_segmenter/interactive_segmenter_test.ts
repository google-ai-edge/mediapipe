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

// Add NodeJS global DOM mocks for headless unit testing.
if (typeof navigator === 'undefined') {
  // tslint:disable-next-line:no-any
  (globalThis as any).navigator = {userAgent: 'NodeJS'};
}
if (typeof document === 'undefined') {
  // tslint:disable-next-line:no-any
  (globalThis as any).document = {
    createElement: (tag: string) => {
      return {width: 2, height: 2, getContext: () => null};
    },
  };
}
if (typeof window === 'undefined') {
  // tslint:disable-next-line:no-any
  (globalThis as any).window = globalThis;
}

function runGpuTest(): boolean {
  if (typeof navigator !== 'undefined' && navigator.userAgent === 'NodeJS') {
    return true;
  }
  if (typeof document === 'undefined') {
    return false;
  }
  try {
    const canvas = document.createElement('canvas');
    return !!(canvas.getContext('webgl2') || canvas.getContext('webgl'));
  } catch (e) {
    return false;
  }
}

import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {TaskLogger} from '../../../../tasks/web/core/task_logger';
import {TaskRunner} from '../../../../tasks/web/core/task_runner';
import {
  createSpyWasmModule,
  SpyWasmModule,
} from '../../../../tasks/web/core/task_runner_test_utils';
import {MPImageShaderContext} from '../../../../tasks/web/vision/core/image_shader_context';
import {MPMask} from '../../../../tasks/web/vision/core/mask';
import {WasmImage} from '../../../../web/graph_runner/graph_runner_image_lib';
import * as platformUtils from '../../../../web/graph_runner/platform_utils';
// Placeholder for internal dependency on trusted resource URL builder

import {
  BrushMode,
  InteractiveSegmenter,
  InteractiveSegmenterWasmModule,
  Stroke,
} from './interactive_segmenter';

class InteractiveSegmenterFake extends InteractiveSegmenter {
  fakeWasmModule: SpyWasmModule &
    jasmine.SpyObj<InteractiveSegmenterWasmModule>;
  fakeLogger = jasmine.createSpyObj<TaskLogger>('TaskLogger', [
    'logSessionStart',
    'logSessionEnd',
    'recordCpuInputArrival',
    'recordGpuInputArrival',
    'recordInvocationEnd',
    'close',
  ]);

  constructor() {
    super(createSpyWasmModule(), /* glCanvas= */ null);
    // Forces casting the generic Wasm module to mock spy types for testing.
    // tslint:disable-next-line:no-unnecessary-type-assertion
    this.fakeWasmModule = this.graphRunner
      .wasmModule as unknown as SpyWasmModule &
      jasmine.SpyObj<InteractiveSegmenterWasmModule>;

    this.logger = this.fakeLogger;

    this.setupWasmHeap();
    this.mockMallocAndFree();
    this.mockNativeMethods();
  }

  private setupWasmHeap(): void {
    const heap8 = new Uint8Array(10000);
    const heap32 = new Uint32Array(heap8.buffer);
    this.fakeWasmModule.HEAPU8 = heap8;
    this.fakeWasmModule.HEAPU32 = heap32;
  }

  private mockMallocAndFree(): void {
    let nextPtr = 1000;
    this.fakeWasmModule._malloc.and.callFake((size: number) => {
      const ptr = nextPtr;
      nextPtr += 100;
      return ptr;
    });
    this.fakeWasmModule._free.and.callFake((ptr: number) => {});
  }

  private mockNativeMethods(): void {
    this.fakeWasmModule._interactive_segmenter_create = jasmine
      .createSpy<
        InteractiveSegmenterWasmModule['_interactive_segmenter_create']
      >('_create')
      .and.returnValue(123);
    this.fakeWasmModule._interactive_segmenter_set_image = jasmine
      .createSpy<
        InteractiveSegmenterWasmModule['_interactive_segmenter_set_image']
      >('_set_image')
      .and.returnValue(true);
    this.fakeWasmModule._interactive_segmenter_segment = jasmine
      .createSpy<
        InteractiveSegmenterWasmModule['_interactive_segmenter_segment']
      >('_segment')
      .and.callFake(
        (
          handle: number,
          strokes: number,
          strokesSize: number,
          wPtr: number,
          hPtr: number,
          sPtr: number,
        ) => {
          this.fakeWasmModule.HEAPU32[wPtr / 4] = 2;
          this.fakeWasmModule.HEAPU32[hPtr / 4] = 2;
          this.fakeWasmModule.HEAPU32[sPtr / 4] = 16; // 2x2 Float32 mask is 16 bytes
          return 1000;
        },
      );
    this.fakeWasmModule._interactive_segmenter_close =
      jasmine.createSpy<
        InteractiveSegmenterWasmModule['_interactive_segmenter_close']
      >('_close');
  }

  override convertToMPMask(
    wasmImage: WasmImage,
    options: {interpolateValues: boolean; shouldCopyData: boolean},
  ): MPMask {
    return super.convertToMPMask(wasmImage, options);
  }
}

describe('InteractiveSegmenter', () => {
  // Boilerplate helpers to generate test inputs
  function createDummyImage(): ImageData {
    return {
      width: 2,
      height: 2,
      data: new Uint8ClampedArray([
        0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255,
      ]),
    } as unknown as ImageData;
  }

  function createTestStrokes(): Stroke[] {
    return [
      {
        brushMode: BrushMode.POSITIVE,
        point: [{x: 0.5, y: 0.5}],
        isCompleted: true,
      },
    ];
  }

  it('exports BrushMode correctly', () => {
    expect(BrushMode.UNSPECIFIED).toBe(0);
    expect(BrushMode.POSITIVE).toBe(1);
    expect(BrushMode.NEGATIVE).toBe(2);
    expect(BrushMode.LASSO).toBe(3);
  });

  it('initializes options and executes segmentation successfully returning a float32 mask', async () => {
    const segmenter = new InteractiveSegmenterFake();
    await segmenter.setOptions({
      baseOptions: {
        modelAssetBuffer: new Uint8Array([0, 1, 2, 3]),
      },
    });

    expect(segmenter).toBeDefined();
    expect(
      segmenter.fakeWasmModule._interactive_segmenter_create,
    ).toHaveBeenCalled();

    segmenter.setImage(createDummyImage());

    expect(
      segmenter.fakeWasmModule._interactive_segmenter_set_image,
    ).toHaveBeenCalled();

    const result = segmenter.segment(createTestStrokes());

    expect(result).toBeInstanceOf(MPMask);
    expect(result.width).toBe(2);
    expect(result.height).toBe(2);
    expect(result.hasFloat32Array()).toBeTrue();
    expect(result.hasUint8Array()).toBeFalse();

    segmenter.close();

    expect(
      segmenter.fakeWasmModule._interactive_segmenter_close,
    ).toHaveBeenCalled();
  });

  it('throws descriptive error when segmenting prior to initialization', () => {
    const uninitializedFake = new InteractiveSegmenterFake();
    // Overwrite handle to simulate uninitialized state.
    // tslint:disable-next-line:no-any
    (uninitializedFake as any).nativeSegmenterHandle = 0;
    expect(() => uninitializedFake.segment([])).toThrowError(
      'Segmenter is not initialized.',
    );
  });

  it('reloads native engine when options are changed (e.g., delegate change)', async () => {
    const segmenter = new InteractiveSegmenterFake();
    await segmenter.setOptions({
      baseOptions: {
        modelAssetBuffer: new Uint8Array([0, 1, 2, 3]),
        delegate: 'CPU',
      },
    });
    expect(
      segmenter.fakeWasmModule._interactive_segmenter_create,
    ).toHaveBeenCalledTimes(1);

    // Change options to GPU
    await segmenter.setOptions({
      baseOptions: {
        modelAssetBuffer: new Uint8Array([0, 1, 2, 3]),
        delegate: 'GPU',
      },
    });
    // Should close first handle and create a new one
    expect(
      segmenter.fakeWasmModule._interactive_segmenter_close,
    ).toHaveBeenCalledTimes(1);
    expect(
      segmenter.fakeWasmModule._interactive_segmenter_create,
    ).toHaveBeenCalledTimes(2);
  });

  it('frees previous image allocation during setImage to prevent memory leaks', () => {
    const segmenter = new InteractiveSegmenterFake();
    const segmenterAsPrivate = segmenter as unknown as {
      nativeSegmenterHandle: number;
      currentImagePixelPtr: number;
    };
    // Overwrite handle to simulate initialized state.
    segmenterAsPrivate.nativeSegmenterHandle = 123;

    segmenter.setImage(createDummyImage());
    const initialPtr = segmenterAsPrivate.currentImagePixelPtr;
    expect(initialPtr).not.toBe(0);

    // Set a new image
    segmenter.setImage(createDummyImage());

    // It should have freed the initial pointer
    expect(segmenter.fakeWasmModule._free).toHaveBeenCalledWith(initialPtr);
  });

  it('frees native segmenter handle and image allocations on close', () => {
    const segmenter = new InteractiveSegmenterFake();
    const segmenterAsPrivate = segmenter as unknown as {
      nativeSegmenterHandle: number;
      currentImagePixelPtr: number;
    };
    // Overwrite handle to simulate initialized state.
    segmenterAsPrivate.nativeSegmenterHandle = 123;
    segmenter.setImage(createDummyImage());
    const initialPtr = segmenterAsPrivate.currentImagePixelPtr;

    segmenter.close();

    expect(
      segmenter.fakeWasmModule._interactive_segmenter_close,
    ).toHaveBeenCalledWith(123);
    expect(segmenter.fakeWasmModule._free).toHaveBeenCalledWith(initialPtr);
    expect(segmenterAsPrivate.nativeSegmenterHandle).toBe(0);
    expect(segmenterAsPrivate.currentImagePixelPtr).toBe(0);
  });

  it('throws error if channel count is unsupported', () => {
    const segmenter = new InteractiveSegmenterFake();
    const segmenterAsPrivate = segmenter as unknown as {
      nativeSegmenterHandle: number;
    };
    segmenterAsPrivate.nativeSegmenterHandle = 123;

    // 2x2 image with 2 channels (8 bytes) -> unsupported
    const badImage = {
      width: 2,
      height: 2,
      data: new Uint8Array(8),
    } as unknown as ImageData;

    expect(() => {
      segmenter.setImage(badImage);
    }).toThrowError(/Invalid image dimensions or pixel data length/);
  });

  it('initializes GPU-delegated options and executes segmentation with standard CPU ImageData', async () => {
    if (!runGpuTest()) {
      pending('WebGL2 is not supported in this environment.');
      return;
    }
    const segmenter = new InteractiveSegmenterFake();
    await segmenter.setOptions({
      baseOptions: {
        modelAssetBuffer: new Uint8Array([0, 1, 2, 3]),
        delegate: 'GPU',
      },
    });
    const createSpy = segmenter.fakeWasmModule._interactive_segmenter_create;
    expect(createSpy).toHaveBeenCalled();
    const [ptr, length] = createSpy.calls.mostRecent().args as [number, number];
    const bytes = segmenter.fakeWasmModule.HEAPU8.subarray(ptr, ptr + length);
    const baseOptions = BaseOptionsProto.deserializeBinary(bytes);
    expect(baseOptions.getAcceleration()?.hasGpu()).toBeTrue();

    segmenter.setImage(createDummyImage());
    expect(
      segmenter.fakeWasmModule._interactive_segmenter_set_image,
    ).toHaveBeenCalled();

    const result = segmenter.segment(createTestStrokes());
    expect(result).toBeInstanceOf(MPMask);
    expect(result.width).toBe(2);
    expect(result.height).toBe(2);
    expect(result.hasFloat32Array()).toBeTrue();
    expect(result.hasUint8Array()).toBeFalse();

    segmenter.close();
    expect(
      segmenter.fakeWasmModule._interactive_segmenter_close,
    ).toHaveBeenCalled();
  });


  // tslint:disable:no-any
  it('creates a canvas when OffscreenCanvas is not supported', async () => {
    spyOn(platformUtils, 'supportsOffscreenCanvas').and.returnValue(false);
    const createInstanceSpy = spyOn(
      TaskRunner as any,
      'createInstance',
    ).and.resolveTo({} as any);

    await InteractiveSegmenter.createFromOptions(
      {wasmLoaderPath: `wasm.js`, wasmBinaryPath: {} as any},
      {},
    );

    expect(createInstanceSpy).toHaveBeenCalled();
    const canvas = createInstanceSpy.calls.mostRecent().args[1] as any;
    expect(canvas).toBeDefined();
    expect(canvas.getContext).toBeDefined();
  });

  it('does not create a canvas when OffscreenCanvas is supported', async () => {
    spyOn(platformUtils, 'supportsOffscreenCanvas').and.returnValue(true);
    const createInstanceSpy = spyOn(
      TaskRunner as any,
      'createInstance',
    ).and.resolveTo({} as any);

    await InteractiveSegmenter.createFromOptions(
      {wasmLoaderPath: `wasm.js`, wasmBinaryPath: {} as any},
      {},
    );

    expect(createInstanceSpy).toHaveBeenCalled();
    const canvas = createInstanceSpy.calls.mostRecent().args[1];
    expect(canvas).toBeUndefined();
  });
  // tslint:enable:no-any

  it('clones mask when shouldCopyData is true', () => {
    const segmenter = new InteractiveSegmenterFake();
    const wasmImage: WasmImage = {
      data: new Float32Array([0.1, 0.2, 0.3, 0.4]),
      width: 2,
      height: 2,
    };

    const maskNoCopy = segmenter.convertToMPMask(wasmImage, {
      interpolateValues: true,
      shouldCopyData: false,
    });
    const maskCopy = segmenter.convertToMPMask(wasmImage, {
      interpolateValues: true,
      shouldCopyData: true,
    });

    expect(maskNoCopy.getAsFloat32Array()).toBe(wasmImage.data as Float32Array);
    expect(maskCopy.getAsFloat32Array()).not.toBe(
      wasmImage.data as Float32Array,
    );
    expect(maskCopy.getAsFloat32Array()).toEqual(
      wasmImage.data as Float32Array,
    );
  });

  it('convertToMPMask throws error if Float32Array mask has unsupported channel count', () => {
    const segmenter = new InteractiveSegmenterFake();
    const wasmImage: WasmImage = {
      data: new Float32Array([0.1, 0.2, 0.3]), // length 3, pixels = 2 * 2 = 4
      width: 2,
      height: 2,
    };

    expect(() => {
      segmenter.convertToMPMask(wasmImage, {
        interpolateValues: true,
        shouldCopyData: false,
      });
    }).toThrowError(/Unsupported channel count/);
  });

  it('convertToMPMask throws error if Uint8Array mask has unsupported channel count', () => {
    const segmenter = new InteractiveSegmenterFake();
    const wasmImage: WasmImage = {
      data: new Uint8Array([1, 2, 3]), // length 3, pixels = 2 * 2 = 4
      width: 2,
      height: 2,
    };

    expect(() => {
      segmenter.convertToMPMask(wasmImage, {
        interpolateValues: true,
        shouldCopyData: false,
      });
    }).toThrowError(/Unsupported channel count/);
  });

  it('closes shader context on close', () => {
    const segmenter = new InteractiveSegmenterFake();
    const shaderCloseSpy = spyOn(
      MPImageShaderContext.prototype,
      'close',
    ).and.callThrough();

    segmenter.close();

    expect(shaderCloseSpy).toHaveBeenCalled();
  });

  it('logs session start when options are initially configured', async () => {
    const segmenter = new InteractiveSegmenterFake();

    await segmenter.setOptions({
      baseOptions: {
        modelAssetBuffer: new Uint8Array([0, 1, 2, 3]),
        delegate: 'CPU',
      },
    });

    expect(segmenter.fakeLogger.logSessionStart).toHaveBeenCalledTimes(1);
    expect(segmenter.fakeLogger.logSessionEnd).not.toHaveBeenCalled();
  });

  it('logs session end and start when options are updated', async () => {
    const segmenter = new InteractiveSegmenterFake();
    await segmenter.setOptions({
      baseOptions: {
        modelAssetBuffer: new Uint8Array([0, 1, 2, 3]),
        delegate: 'CPU',
      },
    });
    segmenter.fakeLogger.logSessionStart.calls.reset();
    segmenter.fakeLogger.logSessionEnd.calls.reset();

    await segmenter.setOptions({
      baseOptions: {
        modelAssetBuffer: new Uint8Array([0, 1, 2, 3]),
        delegate: 'GPU',
      },
    });

    expect(segmenter.fakeLogger.logSessionEnd).toHaveBeenCalledTimes(1);
    expect(segmenter.fakeLogger.logSessionStart).toHaveBeenCalledTimes(1);
  });

  it('logs input arrival and invocation end during segment()', async () => {
    const segmenter = new InteractiveSegmenterFake();
    await segmenter.setOptions({
      baseOptions: {
        modelAssetBuffer: new Uint8Array([0, 1, 2, 3]),
        delegate: 'GPU',
      },
    });
    segmenter.setImage(createDummyImage());

    segmenter.segment(createTestStrokes());

    expect(segmenter.fakeLogger.recordGpuInputArrival).toHaveBeenCalledWith(0);
    expect(segmenter.fakeLogger.recordCpuInputArrival).not.toHaveBeenCalled();
    expect(segmenter.fakeLogger.recordInvocationEnd).toHaveBeenCalledWith(0);
  });

  it('increments the logged timestamp for subsequent segment() calls', async () => {
    const segmenter = new InteractiveSegmenterFake();
    await segmenter.setOptions({
      baseOptions: {
        modelAssetBuffer: new Uint8Array([0, 1, 2, 3]),
        delegate: 'GPU',
      },
    });
    segmenter.setImage(createDummyImage());
    segmenter.segment(createTestStrokes());

    segmenter.segment(createTestStrokes());

    expect(segmenter.fakeLogger.recordGpuInputArrival).toHaveBeenCalledWith(0);
    expect(segmenter.fakeLogger.recordGpuInputArrival).toHaveBeenCalledWith(1);
    expect(segmenter.fakeLogger.recordInvocationEnd).toHaveBeenCalledWith(0);
    expect(segmenter.fakeLogger.recordInvocationEnd).toHaveBeenCalledWith(1);
  });

  it('logs session end and close when segmenter is closed', async () => {
    const segmenter = new InteractiveSegmenterFake();
    await segmenter.setOptions({
      baseOptions: {
        modelAssetBuffer: new Uint8Array([0, 1, 2, 3]),
        delegate: 'CPU',
      },
    });
    segmenter.fakeLogger.logSessionEnd.calls.reset();

    segmenter.close();

    expect(segmenter.fakeLogger.logSessionEnd).toHaveBeenCalledTimes(1);
    expect(segmenter.fakeLogger.close).toHaveBeenCalledTimes(1);
  });
});
