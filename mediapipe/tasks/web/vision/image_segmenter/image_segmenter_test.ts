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
import {CalculatorGraphConfig} from '../../../../framework/calculator_pb';
import {addJasmineCustomFloatEqualityTester, createSpyWasmModule, MediapipeTasksFake, SpyWasmModule, verifyGraph} from '../../../../tasks/web/core/task_runner_test_utils';
import {MPMask} from '../../../../tasks/web/vision/core/mask';
import {WasmImage} from '../../../../web/graph_runner/graph_runner_image_lib';

import {ImageSegmenter} from './image_segmenter';
import {ImageSegmenterOptions} from './image_segmenter_options';

class ImageSegmenterFake extends ImageSegmenter implements MediapipeTasksFake {
  calculatorName = 'mediapipe.tasks.vision.image_segmenter.ImageSegmenterGraph';
  attachListenerSpies: jasmine.Spy[] = [];
  graph: CalculatorGraphConfig|undefined;

  fakeWasmModule: SpyWasmModule;
  categoryMaskListener:
      ((images: WasmImage, timestamp: number) => void)|undefined;
  confidenceMasksListener:
      ((images: WasmImage[], timestamp: number) => void)|undefined;
  qualityScoresListener:
      ((data: number[], timestamp: number) => void)|undefined;

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
  }
}

describe('ImageSegmenter', () => {
  let imageSegmenter: ImageSegmenterFake;

  beforeEach(async () => {
    addJasmineCustomFloatEqualityTester();
    imageSegmenter = new ImageSegmenterFake();
    await imageSegmenter.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array([])}});
  });

  afterEach(() => {
    imageSegmenter.close();
  });

  it('initializes graph', async () => {
    verifyGraph(imageSegmenter);

    // Verify default options
    expect(imageSegmenter.categoryMaskListener).not.toBeDefined();
    expect(imageSegmenter.confidenceMasksListener).toBeDefined();
  });

  it('reloads graph when settings are changed', async () => {
    await imageSegmenter.setOptions({displayNamesLocale: 'en'});
    verifyGraph(imageSegmenter, ['displayNamesLocale', 'en']);

    await imageSegmenter.setOptions({displayNamesLocale: 'de'});
    verifyGraph(imageSegmenter, ['displayNamesLocale', 'de']);
  });

  it('can use custom models', async () => {
    const newModel = new Uint8Array([0, 1, 2, 3, 4]);
    const newModelBase64 = Buffer.from(newModel).toString('base64');
    await imageSegmenter.setOptions({
      baseOptions: {
        modelAssetBuffer: newModel,
      }
    });

    verifyGraph(
        imageSegmenter,
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

  it('merges options', async () => {
    await imageSegmenter.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array([])}});
    await imageSegmenter.setOptions({displayNamesLocale: 'en'});
    verifyGraph(
        imageSegmenter, [['baseOptions', 'modelAsset', 'fileContent'], '']);
    verifyGraph(imageSegmenter, ['displayNamesLocale', 'en']);
  });

  describe('setOptions()', () => {
    interface TestCase {
      optionName: keyof ImageSegmenterOptions;
      fieldPath: string[];
      userValue: unknown;
      graphValue: unknown;
      defaultValue: unknown;
    }

    const testCases: TestCase[] = [{
      optionName: 'displayNamesLocale',
      fieldPath: ['displayNamesLocale'],
      userValue: 'en',
      graphValue: 'en',
      defaultValue: 'en'
    }];

    for (const testCase of testCases) {
      it(`can set ${testCase.optionName}`, async () => {
        await imageSegmenter.setOptions(
            {[testCase.optionName]: testCase.userValue});
        verifyGraph(imageSegmenter, [testCase.fieldPath, testCase.graphValue]);
      });

      it(`can clear ${testCase.optionName}`, async () => {
        await imageSegmenter.setOptions(
            {[testCase.optionName]: testCase.userValue});
        verifyGraph(imageSegmenter, [testCase.fieldPath, testCase.graphValue]);
        await imageSegmenter.setOptions({[testCase.optionName]: undefined});
        verifyGraph(
            imageSegmenter, [testCase.fieldPath, testCase.defaultValue]);
      });
    }
  });

  it('doesn\'t support region of interest', () => {
    expect(() => {
      imageSegmenter.segment(
          {} as HTMLImageElement,
          {regionOfInterest: {left: 0, right: 0, top: 0, bottom: 0}}, () => {});
    }).toThrowError('This task doesn\'t support region-of-interest.');
  });

  it('supports category mask', async () => {
    const mask = new Uint8Array([1, 2, 3, 4]);

    await imageSegmenter.setOptions(
        {outputCategoryMask: true, outputConfidenceMasks: false});

    // Pass the test data to our listener
    imageSegmenter.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      expect(imageSegmenter.categoryMaskListener).toBeDefined();
      imageSegmenter.categoryMaskListener!
          ({data: mask, width: 2, height: 2},
           /* timestamp= */ 1337);
    });

    // Invoke the image segmenter

    return new Promise<void>(resolve => {
      imageSegmenter.segment({} as HTMLImageElement, result => {
        expect(imageSegmenter.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();
        expect(result.categoryMask).toBeInstanceOf(MPMask);
        expect(result.confidenceMasks).not.toBeDefined();
        expect(result.categoryMask!.width).toEqual(2);
        expect(result.categoryMask!.height).toEqual(2);
        resolve();
      });
    });
  });

  it('supports confidence masks', async () => {
    const mask1 = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    const mask2 = new Float32Array([0.5, 0.6, 0.7, 0.8]);

    await imageSegmenter.setOptions(
        {outputCategoryMask: false, outputConfidenceMasks: true});

    // Pass the test data to our listener
    imageSegmenter.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      expect(imageSegmenter.confidenceMasksListener).toBeDefined();
      imageSegmenter.confidenceMasksListener!(
          [
            {data: mask1, width: 2, height: 2},
            {data: mask2, width: 2, height: 2},
          ],
          1337);
    });

    return new Promise<void>(resolve => {
      // Invoke the image segmenter
      imageSegmenter.segment({} as HTMLImageElement, result => {
        expect(imageSegmenter.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();
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

    await imageSegmenter.setOptions(
        {outputCategoryMask: true, outputConfidenceMasks: true});

    // Pass the test data to our listener
    imageSegmenter.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      expect(imageSegmenter.categoryMaskListener).toBeDefined();
      expect(imageSegmenter.confidenceMasksListener).toBeDefined();
      imageSegmenter.categoryMaskListener!
          ({data: categoryMask, width: 1, height: 1}, 1337);
      imageSegmenter.confidenceMasksListener!(
          [
            {data: confidenceMask1, width: 1, height: 1},
            {data: confidenceMask2, width: 1, height: 1},
          ],
          1337);
    });

    return new Promise<void>(resolve => {
      // Invoke the image segmenter
      imageSegmenter.segment({} as HTMLImageElement, result => {
        expect(imageSegmenter.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();
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

    await imageSegmenter.setOptions(
        {outputCategoryMask: true, outputConfidenceMasks: true});

    // Pass the test data to our listener
    imageSegmenter.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      expect(listenerCalled).toBeFalse();
      imageSegmenter.categoryMaskListener!
          ({data: categoryMask, width: 1, height: 1}, 1337);
      expect(listenerCalled).toBeFalse();
      imageSegmenter.confidenceMasksListener!(
          [
            {data: confidenceMask, width: 1, height: 1},
          ],
          1337);
      expect(listenerCalled).toBeFalse();
      imageSegmenter.qualityScoresListener!(qualityScores, 1337);
      expect(listenerCalled).toBeFalse();
    });

    return new Promise<void>(resolve => {
      imageSegmenter.segment({} as HTMLImageElement, result => {
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
    imageSegmenter.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      imageSegmenter.confidenceMasksListener!(
          [
            {data: confidenceMask, width: 1, height: 1},
          ],
          1337);
    });

    const result = imageSegmenter.segment({} as HTMLImageElement);
    expect(result.confidenceMasks![0]).toBeInstanceOf(MPMask);
    result.close();
  });
});
