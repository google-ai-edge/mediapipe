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
import {Detection as DetectionProto} from '../../../../framework/formats/detection_pb';
import {LocationData} from '../../../../framework/formats/location_data_pb';
import {addJasmineCustomFloatEqualityTester, createSpyWasmModule, MediapipeTasksFake, SpyWasmModule, verifyGraph, verifyListenersRegistered} from '../../../../tasks/web/core/task_runner_test_utils';

import {FaceDetector} from './face_detector';
import {FaceDetectorOptions} from './face_detector_options';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

class FaceDetectorFake extends FaceDetector implements MediapipeTasksFake {
  lastSampleRate: number|undefined;
  calculatorName = 'mediapipe.tasks.vision.face_detector.FaceDetectorGraph';
  attachListenerSpies: jasmine.Spy[] = [];
  graph: CalculatorGraphConfig|undefined;

  fakeWasmModule: SpyWasmModule;
  protoListener:
      ((binaryProtos: Uint8Array[], timestamp: number) => void)|undefined;

  constructor() {
    super(createSpyWasmModule(), /* glCanvas= */ null);
    this.fakeWasmModule =
        this.graphRunner.wasmModule as unknown as SpyWasmModule;

    this.attachListenerSpies[0] =
        spyOn(this.graphRunner, 'attachProtoVectorListener')
            .and.callFake((stream, listener) => {
              expect(stream).toEqual('detections');
              this.protoListener = listener;
            });
    spyOn(this.graphRunner, 'setGraph').and.callFake(binaryGraph => {
      this.graph = CalculatorGraphConfig.deserializeBinary(binaryGraph);
    });
    spyOn(this.graphRunner, 'addGpuBufferAsImageToStream');
  }
}

describe('FaceDetector', () => {
  let faceDetector: FaceDetectorFake;

  beforeEach(async () => {
    addJasmineCustomFloatEqualityTester();
    faceDetector = new FaceDetectorFake();
    await faceDetector.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array([])}});
  });

  afterEach(() => {
    faceDetector.close();
  });

  it('initializes graph', async () => {
    verifyGraph(faceDetector);
    verifyListenersRegistered(faceDetector);
  });

  it('reloads graph when settings are changed', async () => {
    await faceDetector.setOptions({minDetectionConfidence: 0.1});
    verifyGraph(faceDetector, ['minDetectionConfidence', 0.1]);
    verifyListenersRegistered(faceDetector);

    await faceDetector.setOptions({minDetectionConfidence: 0.2});
    verifyGraph(faceDetector, ['minDetectionConfidence', 0.2]);
    verifyListenersRegistered(faceDetector);
  });

  it('can use custom models', async () => {
    const newModel = new Uint8Array([0, 1, 2, 3, 4]);
    const newModelBase64 = Buffer.from(newModel).toString('base64');
    await faceDetector.setOptions({
      baseOptions: {
        modelAssetBuffer: newModel,
      }
    });

    verifyGraph(
        faceDetector,
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
    await faceDetector.setOptions({minDetectionConfidence: 0.1});
    await faceDetector.setOptions({minSuppressionThreshold: 0.2});
    verifyGraph(faceDetector, ['minDetectionConfidence', 0.1]);
    verifyGraph(faceDetector, ['minSuppressionThreshold', 0.2]);
  });

  describe('setOptions()', () => {
    interface TestCase {
      optionName: keyof FaceDetectorOptions;
      protoName: string;
      customValue: unknown;
      defaultValue: unknown;
    }

    const testCases: TestCase[] = [
      {
        optionName: 'minDetectionConfidence',
        protoName: 'minDetectionConfidence',
        customValue: 0.1,
        defaultValue: 0.5
      },
      {
        optionName: 'minSuppressionThreshold',
        protoName: 'minSuppressionThreshold',
        customValue: 0.2,
        defaultValue: 0.3
      },
    ];

    for (const testCase of testCases) {
      it(`can set ${testCase.optionName}`, async () => {
        await faceDetector.setOptions(
            {[testCase.optionName]: testCase.customValue});
        verifyGraph(faceDetector, [testCase.protoName, testCase.customValue]);
      });

      it(`can clear ${testCase.optionName}`, async () => {
        await faceDetector.setOptions(
            {[testCase.optionName]: testCase.customValue});
        verifyGraph(faceDetector, [testCase.protoName, testCase.customValue]);
        await faceDetector.setOptions({[testCase.optionName]: undefined});
        verifyGraph(faceDetector, [testCase.protoName, testCase.defaultValue]);
      });
    }
  });

  it('doesn\'t support region of interest', () => {
    expect(() => {
      faceDetector.detect(
          {} as HTMLImageElement,
          {regionOfInterest: {left: 0, right: 0, top: 0, bottom: 0}});
    }).toThrowError('This task doesn\'t support region-of-interest.');
  });

  it('transforms results', async () => {
    const detection = new DetectionProto();
    detection.addScore(0.1);
    const locationData = new LocationData();
    const boundingBox = new LocationData.BoundingBox();
    locationData.setBoundingBox(boundingBox);
    detection.setLocationData(locationData);

    const binaryProto = detection.serializeBinary();

    // Pass the test data to our listener
    faceDetector.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      verifyListenersRegistered(faceDetector);
      faceDetector.protoListener!([binaryProto], 1337);
    });

    // Invoke the face detector
    const {detections} = faceDetector.detect({} as HTMLImageElement);

    expect(faceDetector.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();
    expect(detections.length).toEqual(1);
    expect(detections[0]).toEqual({
      categories: [{
        score: 0.1,
        index: -1,
        categoryName: '',
        displayName: '',
      }],
      boundingBox: {originX: 0, originY: 0, width: 0, height: 0, angle: 0},
      keypoints: []
    });
  });
});
