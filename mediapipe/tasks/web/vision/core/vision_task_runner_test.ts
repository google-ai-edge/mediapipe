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

import {NormalizedRect} from '../../../../framework/formats/rect_pb';
import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {addJasmineCustomFloatEqualityTester} from '../../../../tasks/web/core/task_runner_test_utils';
import {ImageProcessingOptions} from '../../../../tasks/web/vision/core/image_processing_options';
import {ImageSource} from '../../../../web/graph_runner/graph_runner';

import {VisionTaskOptions} from './vision_task_options';
import {VisionGraphRunner, VisionTaskRunner} from './vision_task_runner';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

const IMAGE_STREAM = 'image_in';
const NORM_RECT_STREAM = 'norm_rect';

const IMAGE = {} as unknown as HTMLImageElement;
const TIMESTAMP = 42;

class VisionTaskRunnerFake extends VisionTaskRunner {
  declare graphRunner: VisionGraphRunner;

  baseOptions = new BaseOptionsProto();
  fakeGraphRunner: jasmine.SpyObj<VisionGraphRunner>;
  expectedImageSource?: ImageSource;
  expectedNormalizedRect?: NormalizedRect;

  constructor(roiAllowed = true) {
    super(
      jasmine.createSpyObj<VisionGraphRunner>([
        'addProtoToStream',
        'addGpuBufferAsImageToStream',
        'setAutoRenderToScreen',
        'registerModelResourcesGraphService',
        'finishProcessing',
        'wasmModule',
      ]),
      IMAGE_STREAM,
      NORM_RECT_STREAM,
      roiAllowed,
    );

    this.fakeGraphRunner = this
      .graphRunner as unknown as jasmine.SpyObj<VisionGraphRunner>;

    (this.graphRunner.addProtoToStream as jasmine.Spy).and.callFake(
      (serializedData, type, streamName, timestamp) => {
        expect(type).toBe('mediapipe.NormalizedRect');
        expect(streamName).toBe(NORM_RECT_STREAM);
        expect(timestamp).toBe(TIMESTAMP);

        const actualNormalizedRect =
          NormalizedRect.deserializeBinary(serializedData);
        expect(actualNormalizedRect.toObject()).toEqual(
          this.expectedNormalizedRect!.toObject(),
        );
      },
    );

    (this.graphRunner.addGpuBufferAsImageToStream as jasmine.Spy).and.callFake(
      (imageSource, streamName, timestamp) => {
        expect(streamName).toBe(IMAGE_STREAM);
        expect(timestamp).toBe(TIMESTAMP);
        expect(imageSource).toBe(this.expectedImageSource!);
      },
    );

    // SetOptions with a modelAssetBuffer runs synchronously
    void this.setOptions({baseOptions: {modelAssetBuffer: new Uint8Array([])}});
  }

  protected override refreshGraph(): void {}

  override setOptions(options: VisionTaskOptions): Promise<void> {
    return this.applyOptions(options);
  }

  override processImageData(
    image: ImageSource,
    imageProcessingOptions: ImageProcessingOptions | undefined,
  ): void {
    super.processImageData(image, imageProcessingOptions);
  }

  override processVideoData(
    imageFrame: ImageSource,
    imageProcessingOptions: ImageProcessingOptions | undefined,
    timestamp: number,
  ): void {
    super.processVideoData(imageFrame, imageProcessingOptions, timestamp);
  }

  expectNormalizedRect(
    xCenter: number,
    yCenter: number,
    width: number,
    height: number,
  ): void {
    const rect = new NormalizedRect();
    rect.setXCenter(xCenter);
    rect.setYCenter(yCenter);
    rect.setWidth(width);
    rect.setHeight(height);
    this.expectedNormalizedRect = rect;
  }

  expectImage(imageSource: ImageSource): void {
    this.expectedImageSource = imageSource;
  }
}

describe('VisionTaskRunner', () => {
  beforeEach(() => {
    addJasmineCustomFloatEqualityTester();
  });

  it('can enable image mode', async () => {
    const visionTaskRunner = new VisionTaskRunnerFake();
    await visionTaskRunner.setOptions({runningMode: 'IMAGE'});
    expect(visionTaskRunner.baseOptions.toObject()).toEqual(
      jasmine.objectContaining({useStreamMode: false}),
    );
  });

  it('can enable video mode', async () => {
    const visionTaskRunner = new VisionTaskRunnerFake();
    await visionTaskRunner.setOptions({runningMode: 'VIDEO'});
    expect(visionTaskRunner.baseOptions.toObject()).toEqual(
      jasmine.objectContaining({useStreamMode: true}),
    );
  });

  it('can clear running mode', async () => {
    const visionTaskRunner = new VisionTaskRunnerFake();
    await visionTaskRunner.setOptions({runningMode: 'VIDEO'});

    // Clear running mode
    await visionTaskRunner.setOptions({
      runningMode: /* imageProcessingOptions= */ undefined,
    });
    expect(visionTaskRunner.baseOptions.toObject()).toEqual(
      jasmine.objectContaining({useStreamMode: false}),
    );
  });

  it('cannot process images with video mode', async () => {
    const visionTaskRunner = new VisionTaskRunnerFake();
    await visionTaskRunner.setOptions({runningMode: 'VIDEO'});
    expect(() => {
      visionTaskRunner.processImageData(
        IMAGE,
        /* imageProcessingOptions= */ undefined,
      );
    }).toThrowError(/Task is not initialized with image mode./);
  });

  it('cannot process video with image mode', async () => {
    const visionTaskRunner = new VisionTaskRunnerFake();
    // Use default for `useStreamMode`
    expect(() => {
      visionTaskRunner.processVideoData(
        IMAGE,
        /* imageProcessingOptions= */ undefined,
        TIMESTAMP,
      );
    }).toThrowError(/Task is not initialized with video mode./);

    // Explicitly set to image mode
    await visionTaskRunner.setOptions({runningMode: 'IMAGE'});
    expect(() => {
      visionTaskRunner.processVideoData(
        IMAGE,
        /* imageProcessingOptions= */ undefined,
        TIMESTAMP,
      );
    }).toThrowError(/Task is not initialized with video mode./);
  });

  it('validates that the canvas cannot be changed', async () => {
    if (typeof OffscreenCanvas === 'undefined') {
      console.log('Test is not supported under Node.');
      return;
    }

    const visionTaskRunner = new VisionTaskRunnerFake();
    const canvas = new OffscreenCanvas(1, 1);
    visionTaskRunner.graphRunner.wasmModule.canvas = canvas;
    expect(() => {
      visionTaskRunner.setOptions({canvas});
    }).not.toThrow();

    expect(() => {
      visionTaskRunner.setOptions({canvas: new OffscreenCanvas(2, 2)});
    }).toThrowError(/You must create a new task to reset the canvas./);
  });

  it('validates that an undefined canvas leaves the graph unmodified', async () => {
    if (typeof OffscreenCanvas === 'undefined') {
      console.log('Test is not supported under Node.');
      return;
    }

    const visionTaskRunner = new VisionTaskRunnerFake();
    const canvas = new OffscreenCanvas(1, 1);
    visionTaskRunner.graphRunner.wasmModule.canvas = canvas;

    await visionTaskRunner.setOptions({canvas});
    await visionTaskRunner.setOptions({canvas: undefined});

    expect(visionTaskRunner.graphRunner.wasmModule.canvas).toBe(canvas);
  });

  it('sends packets to graph', async () => {
    const visionTaskRunner = new VisionTaskRunnerFake();
    await visionTaskRunner.setOptions({runningMode: 'VIDEO'});

    visionTaskRunner.expectImage(IMAGE);
    visionTaskRunner.expectNormalizedRect(0.5, 0.5, 1, 1);
    visionTaskRunner.processVideoData(
      IMAGE,
      /* imageProcessingOptions= */ undefined,
      TIMESTAMP,
    );
  });

  it('sends packets to graph with image processing options', async () => {
    const visionTaskRunner = new VisionTaskRunnerFake();
    await visionTaskRunner.setOptions({runningMode: 'VIDEO'});

    visionTaskRunner.expectImage(IMAGE);
    visionTaskRunner.expectNormalizedRect(0.3, 0.6, 0.2, 0.4);
    visionTaskRunner.processVideoData(
      IMAGE,
      {regionOfInterest: {left: 0.2, right: 0.4, top: 0.4, bottom: 0.8}},
      TIMESTAMP,
    );
  });

  describe('validates processing options', () => {
    it('with left > right', () => {
      const visionTaskRunner = new VisionTaskRunnerFake();
      expect(() => {
        visionTaskRunner.processImageData(IMAGE, {
          regionOfInterest: {
            left: 0.2,
            right: 0.1,
            top: 0.1,
            bottom: 0.2,
          },
        });
      }).toThrowError('Expected RectF with left < right and top < bottom.');
    });

    it('with top > bottom', () => {
      const visionTaskRunner = new VisionTaskRunnerFake();
      expect(() => {
        visionTaskRunner.processImageData(IMAGE, {
          regionOfInterest: {
            left: 0.1,
            right: 0.2,
            top: 0.2,
            bottom: 0.1,
          },
        });
      }).toThrowError('Expected RectF with left < right and top < bottom.');
    });

    it('with out of range values', () => {
      const visionTaskRunner = new VisionTaskRunnerFake();
      expect(() => {
        visionTaskRunner.processImageData(IMAGE, {
          regionOfInterest: {
            left: 0.1,
            right: 1.1,
            top: 0.1,
            bottom: 0.2,
          },
        });
      }).toThrowError('Expected RectF values to be in [0,1].');
    });

    it('without region of interest support', () => {
      const visionTaskRunner = new VisionTaskRunnerFake(
        /* roiAllowed= */ false,
      );
      expect(() => {
        visionTaskRunner.processImageData(IMAGE, {
          regionOfInterest: {
            left: 0.1,
            right: 0.2,
            top: 0.1,
            bottom: 0.2,
          },
        });
      }).toThrowError("This task doesn't support region-of-interest.");
    });

    it('with non-90 degree rotation', () => {
      const visionTaskRunner = new VisionTaskRunnerFake();
      expect(() => {
        visionTaskRunner.processImageData(IMAGE, {rotationDegrees: 42});
      }).toThrowError('Expected rotation to be a multiple of 90°.');
    });
  });
});
