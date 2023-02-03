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

// Placeholder for internal dependency on encodeByteArray
import {CalculatorGraphConfig} from '../../../../framework/calculator_pb';
import {addJasmineCustomFloatEqualityTester, createSpyWasmModule, MediapipeTasksFake, SpyWasmModule, verifyGraph, verifyListenersRegistered} from '../../../../tasks/web/core/task_runner_test_utils';
import {WasmImage} from '../../../../web/graph_runner/graph_runner_image_lib';

import {ImageSegmenter} from './image_segmenter';
import {ImageSegmenterOptions} from './image_segmenter_options';

class ImageSegmenterFake extends ImageSegmenter implements MediapipeTasksFake {
  calculatorName = 'mediapipe.tasks.vision.image_segmenter.ImageSegmenterGraph';
  attachListenerSpies: jasmine.Spy[] = [];
  graph: CalculatorGraphConfig|undefined;

  fakeWasmModule: SpyWasmModule;
  imageVectorListener:
      ((images: WasmImage[], timestamp: number) => void)|undefined;

  constructor() {
    super(createSpyWasmModule(), /* glCanvas= */ null);
    this.fakeWasmModule =
        this.graphRunner.wasmModule as unknown as SpyWasmModule;

    this.attachListenerSpies[0] =
        spyOn(this.graphRunner, 'attachImageVectorListener')
            .and.callFake((stream, listener) => {
              expect(stream).toEqual('segmented_masks');
              this.imageVectorListener = listener;
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

  it('initializes graph', async () => {
    verifyGraph(imageSegmenter);
    verifyListenersRegistered(imageSegmenter);
  });

  it('reloads graph when settings are changed', async () => {
    await imageSegmenter.setOptions({displayNamesLocale: 'en'});
    verifyGraph(imageSegmenter, ['displayNamesLocale', 'en']);
    verifyListenersRegistered(imageSegmenter);

    await imageSegmenter.setOptions({displayNamesLocale: 'de'});
    verifyGraph(imageSegmenter, ['displayNamesLocale', 'de']);
    verifyListenersRegistered(imageSegmenter);
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
    await imageSegmenter.setOptions({outputType: 'CATEGORY_MASK'});
    await imageSegmenter.setOptions({displayNamesLocale: 'en'});
    verifyGraph(imageSegmenter, [['segmenterOptions', 'outputType'], 1]);
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

    const testCases: TestCase[] = [
      {
        optionName: 'displayNamesLocale',
        fieldPath: ['displayNamesLocale'],
        userValue: 'en',
        graphValue: 'en',
        defaultValue: 'en'
      },
      {
        optionName: 'outputType',
        fieldPath: ['segmenterOptions', 'outputType'],
        userValue: 'CONFIDENCE_MASK',
        graphValue: 2,
        defaultValue: 1
      },
    ];

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

  it('supports category masks', (done) => {
    const mask = new Uint8Array([1, 2, 3, 4]);

    // Pass the test data to our listener
    imageSegmenter.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      verifyListenersRegistered(imageSegmenter);
      imageSegmenter.imageVectorListener!(
          [
            {data: mask, width: 2, height: 2},
          ],
          /* timestamp= */ 1337);
    });

    // Invoke the image segmenter
    imageSegmenter.segment({} as HTMLImageElement, (masks, width, height) => {
      expect(imageSegmenter.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();
      expect(masks).toHaveSize(1);
      expect(masks[0]).toEqual(mask);
      expect(width).toEqual(2);
      expect(height).toEqual(2);
      done();
    });
  });

  it('supports confidence masks', async () => {
    const mask1 = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    const mask2 = new Float32Array([0.5, 0.6, 0.7, 0.8]);

    await imageSegmenter.setOptions({outputType: 'CONFIDENCE_MASK'});

    // Pass the test data to our listener
    imageSegmenter.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      verifyListenersRegistered(imageSegmenter);
      imageSegmenter.imageVectorListener!(
          [
            {data: mask1, width: 2, height: 2},
            {data: mask2, width: 2, height: 2},
          ],
          1337);
    });

    return new Promise<void>(resolve => {
      // Invoke the image segmenter
      imageSegmenter.segment({} as HTMLImageElement, (masks, width, height) => {
        expect(imageSegmenter.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();
        expect(masks).toHaveSize(2);
        expect(masks[0]).toEqual(mask1);
        expect(masks[1]).toEqual(mask2);
        expect(width).toEqual(2);
        expect(height).toEqual(2);
        resolve();
      });
    });
  });
});
