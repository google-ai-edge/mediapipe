/**
 * Copyright 2023 The MediaPipe Authors.
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
import {CalculatorGraphConfig} from '../../../../framework/calculator_pb';
import {addJasmineCustomFloatEqualityTester, createSpyWasmModule, MediapipeTasksFake, SpyWasmModule, verifyGraph} from '../../../../tasks/web/core/task_runner_test_utils';
import {MPMask} from '../../../../tasks/web/vision/core/mask';
import {RenderData as RenderDataProto} from '../../../../util/render_data_pb';
import {WasmImage} from '../../../../web/graph_runner/graph_runner_image_lib';

import {InteractiveSegmenter, RegionOfInterest} from './interactive_segmenter';


const KEYPOINT: RegionOfInterest = {
  keypoint: {x: 0.1, y: 0.2}
};

const SCRIBBLE: RegionOfInterest = {
  scribble: [{x: 0.1, y: 0.2}, {x: 0.3, y: 0.4}]
};

class InteractiveSegmenterFake extends InteractiveSegmenter implements
    MediapipeTasksFake {
  calculatorName =
      'mediapipe.tasks.vision.interactive_segmenter.InteractiveSegmenterGraph';
  attachListenerSpies: jasmine.Spy[] = [];
  graph: CalculatorGraphConfig|undefined;

  fakeWasmModule: SpyWasmModule;
  categoryMaskListener:
      ((images: WasmImage, timestamp: number) => void)|undefined;
  confidenceMasksListener:
      ((images: WasmImage[], timestamp: number) => void)|undefined;
  qualityScoresListener:
      ((data: number[], timestamp: number) => void)|undefined;
  lastRoi?: RenderDataProto;

  constructor() {
    super(createSpyWasmModule(), /* glCanvas= */ null);
    this.fakeWasmModule =
        this.graphRunner.wasmModule as unknown as SpyWasmModule;

    this.attachListenerSpies[0] = spyOn(this.graphRunner, 'attachImageListener')
                                      .and.callFake((stream, listener) => {
                                        expect(stream).toEqual('category_mask');
                                        this.categoryMaskListener = listener;
                                      });
    this.attachListenerSpies[1] =
        spyOn(this.graphRunner, 'attachImageVectorListener')
            .and.callFake((stream, listener) => {
              expect(stream).toEqual('confidence_masks');
              this.confidenceMasksListener = listener;
            });
    this.attachListenerSpies[2] =
        spyOn(this.graphRunner, 'attachFloatVectorListener')
            .and.callFake((stream, listener) => {
              expect(stream).toEqual('quality_scores');
              this.qualityScoresListener = listener;
            });
    spyOn(this.graphRunner, 'setGraph').and.callFake(binaryGraph => {
      this.graph = CalculatorGraphConfig.deserializeBinary(binaryGraph);
    });
    spyOn(this.graphRunner, 'addGpuBufferAsImageToStream');

    spyOn(this.graphRunner, 'addProtoToStream')
        .and.callFake((data, protoName, stream) => {
          if (stream === 'roi_in') {
            expect(protoName).toEqual('mediapipe.RenderData');
            this.lastRoi = RenderDataProto.deserializeBinary(data);
          }
        });
  }
}

describe('InteractiveSegmenter', () => {
  let interactiveSegmenter: InteractiveSegmenterFake;

  beforeEach(async () => {
    addJasmineCustomFloatEqualityTester();
    interactiveSegmenter = new InteractiveSegmenterFake();
    await interactiveSegmenter.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array([])}});
  });

  afterEach(() => {
    interactiveSegmenter.close();
  });

  it('initializes graph', async () => {
    verifyGraph(interactiveSegmenter);

    // Verify default options
    expect(interactiveSegmenter.categoryMaskListener).not.toBeDefined();
    expect(interactiveSegmenter.confidenceMasksListener).toBeDefined();
  });

  it('reloads graph when settings are changed', async () => {
    await interactiveSegmenter.setOptions(
        {outputConfidenceMasks: true, outputCategoryMask: false});
    expect(interactiveSegmenter.categoryMaskListener).not.toBeDefined();
    expect(interactiveSegmenter.confidenceMasksListener).toBeDefined();

    await interactiveSegmenter.setOptions(
        {outputConfidenceMasks: false, outputCategoryMask: true});
    expect(interactiveSegmenter.categoryMaskListener).toBeDefined();
  });

  it('can use custom models', async () => {
    const newModel = new Uint8Array([0, 1, 2, 3, 4]);
    const newModelBase64 = Buffer.from(newModel).toString('base64');
    await interactiveSegmenter.setOptions({
      baseOptions: {
        modelAssetBuffer: newModel,
      }
    });

    verifyGraph(
        interactiveSegmenter,
        /* expectedCalculatorOptions= */ undefined,
        /* expectedBaseOptions= */
        [
          'modelAsset', {
            fileContent: newModelBase64,
            fileName: undefined,
            fileDescriptorMeta: undefined,
            filePointerMeta: undefined
          }
        ]);
  });

  it('doesn\'t support region of interest', () => {
    expect(() => {
      interactiveSegmenter.segment(
          {} as HTMLImageElement, KEYPOINT,
          {regionOfInterest: {left: 0, right: 0, top: 0, bottom: 0}}, () => {});
    }).toThrowError('This task doesn\'t support region-of-interest.');
  });

  it('sends region-of-interest with keypoint', (done) => {
    interactiveSegmenter.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      expect(interactiveSegmenter.lastRoi).toBeDefined();
      expect(interactiveSegmenter.lastRoi!.toObject().renderAnnotationsList![0])
          .toEqual(jasmine.objectContaining({
            color: {r: 255, b: undefined, g: undefined},
            point: {x: 0.1, y: 0.2, normalized: true},
          }));
      done();
    });

    interactiveSegmenter.segment({} as HTMLImageElement, KEYPOINT, () => {});
  });

  it('sends region-of-interest with scribble', (done) => {
    interactiveSegmenter.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      expect(interactiveSegmenter.lastRoi).toBeDefined();
      expect(interactiveSegmenter.lastRoi!.toObject().renderAnnotationsList![0])
          .toEqual(jasmine.objectContaining({
            color: {r: 255, b: undefined, g: undefined},
            scribble: {
              pointList: [
                {x: 0.1, y: 0.2, normalized: true},
                {x: 0.3, y: 0.4, normalized: true}
              ]
            },
          }));
      done();
    });

    interactiveSegmenter.segment({} as HTMLImageElement, SCRIBBLE, () => {});
  });

  it('supports category mask', async () => {
    const mask = new Uint8Array([1, 2, 3, 4]);

    await interactiveSegmenter.setOptions(
        {outputCategoryMask: true, outputConfidenceMasks: false});

    // Pass the test data to our listener
    interactiveSegmenter.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      expect(interactiveSegmenter.categoryMaskListener).toBeDefined();
      interactiveSegmenter.categoryMaskListener!
          ({data: mask, width: 2, height: 2},
           /* timestamp= */ 1337);
    });

    // Invoke the image segmenter
    return new Promise<void>(resolve => {
      interactiveSegmenter.segment({} as HTMLImageElement, KEYPOINT, result => {
        expect(interactiveSegmenter.fakeWasmModule._waitUntilIdle)
            .toHaveBeenCalled();
        expect(result.categoryMask).toBeInstanceOf(MPMask);
        expect(result.categoryMask!.width).toEqual(2);
        expect(result.categoryMask!.height).toEqual(2);
        expect(result.confidenceMasks).not.toBeDefined();
        resolve();
      });
    });
  });

  it('supports confidence masks', async () => {
    const mask1 = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    const mask2 = new Float32Array([0.5, 0.6, 0.7, 0.8]);

    await interactiveSegmenter.setOptions(
        {outputCategoryMask: false, outputConfidenceMasks: true});

    // Pass the test data to our listener
    interactiveSegmenter.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      expect(interactiveSegmenter.confidenceMasksListener).toBeDefined();
      interactiveSegmenter.confidenceMasksListener!(
          [
            {data: mask1, width: 2, height: 2},
            {data: mask2, width: 2, height: 2},
          ],
          1337);
    });
    return new Promise<void>(resolve => {
      // Invoke the image segmenter
      interactiveSegmenter.segment({} as HTMLImageElement, KEYPOINT, result => {
        expect(interactiveSegmenter.fakeWasmModule._waitUntilIdle)
            .toHaveBeenCalled();
        expect(result.categoryMask).not.toBeDefined();

        expect(result.confidenceMasks![0]).toBeInstanceOf(MPMask);
        expect(result.confidenceMasks![0].width).toEqual(2);
        expect(result.confidenceMasks![0].height).toEqual(2);

        expect(result.confidenceMasks![1]).toBeInstanceOf(MPMask);
        resolve();
      });
    });
  });

  it('supports combined category and confidence masks', async () => {
    const categoryMask = new Uint8Array([1]);
    const confidenceMask1 = new Float32Array([0.0]);
    const confidenceMask2 = new Float32Array([1.0]);

    await interactiveSegmenter.setOptions(
        {outputCategoryMask: true, outputConfidenceMasks: true});

    // Pass the test data to our listener
    interactiveSegmenter.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      expect(interactiveSegmenter.categoryMaskListener).toBeDefined();
      expect(interactiveSegmenter.confidenceMasksListener).toBeDefined();
      interactiveSegmenter.categoryMaskListener!
          ({data: categoryMask, width: 1, height: 1}, 1337);
      interactiveSegmenter.confidenceMasksListener!(
          [
            {data: confidenceMask1, width: 1, height: 1},
            {data: confidenceMask2, width: 1, height: 1},
          ],
          1337);
    });

    return new Promise<void>(resolve => {
      // Invoke the image segmenter
      interactiveSegmenter.segment(
          {} as HTMLImageElement, KEYPOINT, result => {
            expect(interactiveSegmenter.fakeWasmModule._waitUntilIdle)
                .toHaveBeenCalled();
            expect(result.categoryMask).toBeInstanceOf(MPMask);
            expect(result.categoryMask!.width).toEqual(1);
            expect(result.categoryMask!.height).toEqual(1);

            expect(result.confidenceMasks![0]).toBeInstanceOf(MPMask);
            expect(result.confidenceMasks![1]).toBeInstanceOf(MPMask);
            resolve();
          });
    });
  });

  it('invokes listener after masks are available', async () => {
    const categoryMask = new Uint8Array([1]);
    const confidenceMask = new Float32Array([0.0]);
    const qualityScores = [1.0];
    let listenerCalled = false;

    await interactiveSegmenter.setOptions(
        {outputCategoryMask: true, outputConfidenceMasks: true});

    // Pass the test data to our listener
    interactiveSegmenter.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      expect(listenerCalled).toBeFalse();
      interactiveSegmenter.categoryMaskListener!
          ({data: categoryMask, width: 1, height: 1}, 1337);
      expect(listenerCalled).toBeFalse();
      interactiveSegmenter.confidenceMasksListener!(
          [
            {data: confidenceMask, width: 1, height: 1},
          ],
          1337);
      expect(listenerCalled).toBeFalse();
      interactiveSegmenter.qualityScoresListener!(qualityScores, 1337);
      expect(listenerCalled).toBeFalse();
    });

    return new Promise<void>(resolve => {
      interactiveSegmenter.segment({} as HTMLImageElement, KEYPOINT, result => {
        listenerCalled = true;
        expect(result.categoryMask).toBeInstanceOf(MPMask);
        expect(result.confidenceMasks![0]).toBeInstanceOf(MPMask);
        expect(result.qualityScores).toEqual(qualityScores);
        resolve();
      });
    });
  });

  it('returns result', () => {
    const confidenceMask = new Float32Array([0.0]);

    // Pass the test data to our listener
    interactiveSegmenter.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      interactiveSegmenter.confidenceMasksListener!(
          [
            {data: confidenceMask, width: 1, height: 1},
          ],
          1337);
    });

    const result =
        interactiveSegmenter.segment({} as HTMLImageElement, KEYPOINT);
    expect(result.confidenceMasks![0]).toBeInstanceOf(MPMask);
    result.close();
  });
});
