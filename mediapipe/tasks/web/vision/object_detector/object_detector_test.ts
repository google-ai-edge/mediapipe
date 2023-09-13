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

import {ObjectDetector} from './object_detector';
import {ObjectDetectorOptions} from './object_detector_options';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

class ObjectDetectorFake extends ObjectDetector implements MediapipeTasksFake {
  lastSampleRate: number|undefined;
  calculatorName = 'mediapipe.tasks.vision.ObjectDetectorGraph';
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

describe('ObjectDetector', () => {
  let objectDetector: ObjectDetectorFake;

  beforeEach(async () => {
    addJasmineCustomFloatEqualityTester();
    objectDetector = new ObjectDetectorFake();
    await objectDetector.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array([])}});
  });

  afterEach(() => {
    objectDetector.close();
  });

  it('initializes graph', async () => {
    verifyGraph(objectDetector);
    verifyListenersRegistered(objectDetector);
  });

  it('reloads graph when settings are changed', async () => {
    await objectDetector.setOptions({maxResults: 1});
    verifyGraph(objectDetector, ['maxResults', 1]);
    verifyListenersRegistered(objectDetector);

    await objectDetector.setOptions({maxResults: 5});
    verifyGraph(objectDetector, ['maxResults', 5]);
    verifyListenersRegistered(objectDetector);
  });

  it('can use custom models', async () => {
    const newModel = new Uint8Array([0, 1, 2, 3, 4]);
    const newModelBase64 = Buffer.from(newModel).toString('base64');
    await objectDetector.setOptions({
      baseOptions: {
        modelAssetBuffer: newModel,
      }
    });

    verifyGraph(
        objectDetector,
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
    await objectDetector.setOptions({maxResults: 1});
    await objectDetector.setOptions({displayNamesLocale: 'en'});
    verifyGraph(objectDetector, ['maxResults', 1]);
    verifyGraph(objectDetector, ['displayNamesLocale', 'en']);
  });

  describe('setOptions()', () => {
    interface TestCase {
      optionName: keyof ObjectDetectorOptions;
      protoName: string;
      customValue: unknown;
      defaultValue: unknown;
    }

    const testCases: TestCase[] = [
      {
        optionName: 'maxResults',
        protoName: 'maxResults',
        customValue: 5,
        defaultValue: -1
      },
      {
        optionName: 'displayNamesLocale',
        protoName: 'displayNamesLocale',
        customValue: 'en',
        defaultValue: 'en'
      },
      {
        optionName: 'scoreThreshold',
        protoName: 'scoreThreshold',
        customValue: 0.1,
        defaultValue: undefined
      },
      {
        optionName: 'categoryAllowlist',
        protoName: 'categoryAllowlistList',
        customValue: ['foo'],
        defaultValue: []
      },
      {
        optionName: 'categoryDenylist',
        protoName: 'categoryDenylistList',
        customValue: ['bar'],
        defaultValue: []
      },
    ];

    for (const testCase of testCases) {
      it(`can set ${testCase.optionName}`, async () => {
        await objectDetector.setOptions(
            {[testCase.optionName]: testCase.customValue});
        verifyGraph(objectDetector, [testCase.protoName, testCase.customValue]);
      });

      it(`can clear ${testCase.optionName}`, async () => {
        await objectDetector.setOptions(
            {[testCase.optionName]: testCase.customValue});
        verifyGraph(objectDetector, [testCase.protoName, testCase.customValue]);
        await objectDetector.setOptions({[testCase.optionName]: undefined});
        verifyGraph(
            objectDetector, [testCase.protoName, testCase.defaultValue]);
      });
    }
  });

  it('doesn\'t support region of interest', () => {
    expect(() => {
      objectDetector.detect(
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
    objectDetector.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      verifyListenersRegistered(objectDetector);
      objectDetector.protoListener!([binaryProto], 1337);
    });

    // Invoke the object detector
    const {detections} = objectDetector.detect({} as HTMLImageElement);

    expect(objectDetector.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();
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
