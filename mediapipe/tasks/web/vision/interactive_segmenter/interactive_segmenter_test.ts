/**
 * Copyright 2023 The MediaPipe Authors. All Rights Reserved.
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
import {RenderData as RenderDataProto} from '../../../../util/render_data_pb';
import {WasmImage} from '../../../../web/graph_runner/graph_runner_image_lib';

import {InteractiveSegmenter, RegionOfInterest} from './interactive_segmenter';


const ROI: RegionOfInterest = {
  keypoint: {x: 0.1, y: 0.2}
};

class InteractiveSegmenterFake extends InteractiveSegmenter implements
    MediapipeTasksFake {
  calculatorName =
      'mediapipe.tasks.vision.interactive_segmenter.InteractiveSegmenterGraph';
  attachListenerSpies: jasmine.Spy[] = [];
  graph: CalculatorGraphConfig|undefined;

  fakeWasmModule: SpyWasmModule;
  imageVectorListener:
      ((images: WasmImage[], timestamp: number) => void)|undefined;
  lastRoi?: RenderDataProto;

  constructor() {
    super(createSpyWasmModule(), /* glCanvas= */ null);
    this.fakeWasmModule =
        this.graphRunner.wasmModule as unknown as SpyWasmModule;

    this.attachListenerSpies[0] =
        spyOn(this.graphRunner, 'attachImageVectorListener')
            .and.callFake((stream, listener) => {
              expect(stream).toEqual('image_out');
              this.imageVectorListener = listener;
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

  it('initializes graph', async () => {
    verifyGraph(interactiveSegmenter);
    verifyListenersRegistered(interactiveSegmenter);
  });

  it('reloads graph when settings are changed', async () => {
    await interactiveSegmenter.setOptions({outputType: 'CATEGORY_MASK'});
    verifyGraph(interactiveSegmenter, [['segmenterOptions', 'outputType'], 1]);
    verifyListenersRegistered(interactiveSegmenter);

    await interactiveSegmenter.setOptions({outputType: 'CONFIDENCE_MASK'});
    verifyGraph(interactiveSegmenter, [['segmenterOptions', 'outputType'], 2]);
    verifyListenersRegistered(interactiveSegmenter);
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


  describe('setOptions()', () => {
    const fieldPath = ['segmenterOptions', 'outputType'];

    it(`can set outputType`, async () => {
      await interactiveSegmenter.setOptions({outputType: 'CONFIDENCE_MASK'});
      verifyGraph(interactiveSegmenter, [fieldPath, 2]);
    });

    it(`can clear outputType`, async () => {
      await interactiveSegmenter.setOptions({outputType: 'CONFIDENCE_MASK'});
      verifyGraph(interactiveSegmenter, [fieldPath, 2]);
      await interactiveSegmenter.setOptions({outputType: undefined});
      verifyGraph(interactiveSegmenter, [fieldPath, 1]);
    });
  });

  it('doesn\'t support region of interest', () => {
    expect(() => {
      interactiveSegmenter.segment(
          {} as HTMLImageElement, ROI,
          {regionOfInterest: {left: 0, right: 0, top: 0, bottom: 0}}, () => {});
    }).toThrowError('This task doesn\'t support region-of-interest.');
  });

  it('sends region-of-interest', (done) => {
    interactiveSegmenter.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      expect(interactiveSegmenter.lastRoi).toBeDefined();
      expect(interactiveSegmenter.lastRoi!.toObject().renderAnnotationsList![0])
          .toEqual(jasmine.objectContaining({
            color: {r: 255, b: undefined, g: undefined},
          }));
      done();
    });

    interactiveSegmenter.segment({} as HTMLImageElement, ROI, () => {});
  });

  it('supports category masks', (done) => {
    const mask = new Uint8ClampedArray([1, 2, 3, 4]);

    // Pass the test data to our listener
    interactiveSegmenter.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      verifyListenersRegistered(interactiveSegmenter);
      interactiveSegmenter.imageVectorListener!(
          [
            {data: mask, width: 2, height: 2},
          ],
          /* timestamp= */ 1337);
    });

    // Invoke the image segmenter
    interactiveSegmenter.segment(
        {} as HTMLImageElement, ROI, (masks, width, height) => {
          expect(interactiveSegmenter.fakeWasmModule._waitUntilIdle)
              .toHaveBeenCalled();
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

    await interactiveSegmenter.setOptions({outputType: 'CONFIDENCE_MASK'});

    // Pass the test data to our listener
    interactiveSegmenter.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      verifyListenersRegistered(interactiveSegmenter);
      interactiveSegmenter.imageVectorListener!(
          [
            {data: mask1, width: 2, height: 2},
            {data: mask2, width: 2, height: 2},
          ],
          1337);
    });

    return new Promise<void>(resolve => {
      // Invoke the image segmenter
      interactiveSegmenter.segment(
          {} as HTMLImageElement, ROI, (masks, width, height) => {
            expect(interactiveSegmenter.fakeWasmModule._waitUntilIdle)
                .toHaveBeenCalled();
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
