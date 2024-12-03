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

import {CalculatorGraphConfig} from '../../../../framework/calculator_pb';
import {Classification, ClassificationList} from '../../../../framework/formats/classification_pb';
import {createLandmarks, createWorldLandmarks} from '../../../../tasks/web/components/processors/landmark_result_test_lib';
import {addJasmineCustomFloatEqualityTester, createSpyWasmModule, MediapipeTasksFake, SpyWasmModule, verifyGraph, verifyListenersRegistered} from '../../../../tasks/web/core/task_runner_test_utils';
import {VisionGraphRunner} from '../../../../tasks/web/vision/core/vision_task_runner';

import {HandLandmarker} from './hand_landmarker';
import {HandLandmarkerOptions} from './hand_landmarker_options';


// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

type ProtoListener = ((binaryProtos: Uint8Array[], timestamp: number) => void);

function createHandedness(): ClassificationList {
  const handsProto = new ClassificationList();
  const classification = new Classification();
  classification.setScore(0.1);
  classification.setIndex(1);
  classification.setLabel('handedness_label');
  classification.setDisplayName('handedness_display_name');
  handsProto.addClassification(classification);
  return handsProto;
}

class HandLandmarkerFake extends HandLandmarker implements MediapipeTasksFake {
  calculatorName = 'mediapipe.tasks.vision.hand_landmarker.HandLandmarkerGraph';
  attachListenerSpies: jasmine.Spy[] = [];
  graph: CalculatorGraphConfig|undefined;
  fakeWasmModule: SpyWasmModule;
  listeners = new Map<string, ProtoListener>();

  constructor() {
    super(createSpyWasmModule(), /* glCanvas= */ null);
    this.fakeWasmModule =
        this.graphRunner.wasmModule as unknown as SpyWasmModule;

    this.attachListenerSpies[0] =
        spyOn(this.graphRunner, 'attachProtoVectorListener')
            .and.callFake((stream, listener) => {
              expect(stream).toMatch(
                  /(hand_landmarks|world_hand_landmarks|handedness|hand_hands)/);
              this.listeners.set(stream, listener);
            });

    spyOn(this.graphRunner, 'setGraph').and.callFake(binaryGraph => {
      this.graph = CalculatorGraphConfig.deserializeBinary(binaryGraph);
    });
    spyOn(this.graphRunner, 'addGpuBufferAsImageToStream');
    spyOn(this.graphRunner, 'addProtoToStream');
  }

  getGraphRunner(): VisionGraphRunner {
    return this.graphRunner;
  }
}

describe('HandLandmarker', () => {
  let handLandmarker: HandLandmarkerFake;

  beforeEach(async () => {
    addJasmineCustomFloatEqualityTester();
    handLandmarker = new HandLandmarkerFake();
    await handLandmarker.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array([])}});
  });

  afterEach(() => {
    handLandmarker.close();
  });

  it('initializes graph', async () => {
    verifyGraph(handLandmarker);
    verifyListenersRegistered(handLandmarker);
  });

  it('reloads graph when settings are changed', async () => {
    verifyListenersRegistered(handLandmarker);

    await handLandmarker.setOptions({numHands: 1});
    verifyGraph(handLandmarker, [['handDetectorGraphOptions', 'numHands'], 1]);
    verifyListenersRegistered(handLandmarker);

    await handLandmarker.setOptions({numHands: 5});
    verifyGraph(handLandmarker, [['handDetectorGraphOptions', 'numHands'], 5]);
    verifyListenersRegistered(handLandmarker);
  });

  it('merges options', async () => {
    await handLandmarker.setOptions({numHands: 1});
    await handLandmarker.setOptions({minHandDetectionConfidence: 0.5});
    verifyGraph(handLandmarker, [
      'handDetectorGraphOptions',
      {numHands: 1, baseOptions: undefined, minDetectionConfidence: 0.5}
    ]);
  });

  describe('setOptions()', () => {
    interface TestCase {
      optionPath: [keyof HandLandmarkerOptions, ...string[]];
      fieldPath: string[];
      customValue: unknown;
      defaultValue: unknown;
    }

    const testCases: TestCase[] = [
      {
        optionPath: ['numHands'],
        fieldPath: ['handDetectorGraphOptions', 'numHands'],
        customValue: 5,
        defaultValue: 1
      },
      {
        optionPath: ['minHandDetectionConfidence'],
        fieldPath: ['handDetectorGraphOptions', 'minDetectionConfidence'],
        customValue: 0.1,
        defaultValue: 0.5
      },
      {
        optionPath: ['minHandPresenceConfidence'],
        fieldPath:
            ['handLandmarksDetectorGraphOptions', 'minDetectionConfidence'],
        customValue: 0.2,
        defaultValue: 0.5
      },
      {
        optionPath: ['minTrackingConfidence'],
        fieldPath: ['minTrackingConfidence'],
        customValue: 0.3,
        defaultValue: 0.5
      },
    ];

    /** Creates an options object that can be passed to setOptions() */
    function createOptions(
        path: string[], value: unknown): HandLandmarkerOptions {
      const options: Record<string, unknown> = {};
      let currentLevel = options;
      for (const element of path.slice(0, -1)) {
        currentLevel[element] = {};
        currentLevel = currentLevel[element] as Record<string, unknown>;
      }
      currentLevel[path[path.length - 1]] = value;
      return options;
    }

    for (const testCase of testCases) {
      it(`uses default value for ${testCase.optionPath[0]}`, async () => {
        verifyGraph(
            handLandmarker, [testCase.fieldPath, testCase.defaultValue]);
      });

      it(`can set ${testCase.optionPath[0]}`, async () => {
        await handLandmarker.setOptions(
            createOptions(testCase.optionPath, testCase.customValue));
        verifyGraph(handLandmarker, [testCase.fieldPath, testCase.customValue]);
      });

      it(`can clear ${testCase.optionPath[0]}`, async () => {
        await handLandmarker.setOptions(
            createOptions(testCase.optionPath, testCase.customValue));
        verifyGraph(handLandmarker, [testCase.fieldPath, testCase.customValue]);

        await handLandmarker.setOptions(
            createOptions(testCase.optionPath, undefined));
        verifyGraph(
            handLandmarker, [testCase.fieldPath, testCase.defaultValue]);
      });
    }
  });

  it('doesn\'t support region of interest', () => {
    expect(() => {
      handLandmarker.detect(
          {} as HTMLImageElement,
          {regionOfInterest: {left: 0, right: 0, top: 0, bottom: 0}});
    }).toThrowError('This task doesn\'t support region-of-interest.');
  });

  it('transforms results', async () => {
    const landmarksProto = [createLandmarks().serializeBinary()];
    const worldLandmarksProto = [createWorldLandmarks().serializeBinary()];
    const handednessProto = [createHandedness().serializeBinary()];

    // Pass the test data to our listener
    handLandmarker.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      verifyListenersRegistered(handLandmarker);
      handLandmarker.listeners.get('hand_landmarks')!(landmarksProto, 1337);
      handLandmarker.listeners.get('world_hand_landmarks')!
          (worldLandmarksProto, 1337);
      handLandmarker.listeners.get('handedness')!(handednessProto, 1337);
    });

    // Invoke the hand landmarker
    const landmarks = handLandmarker.detect({} as HTMLImageElement);
    expect(handLandmarker.getGraphRunner().addProtoToStream)
        .toHaveBeenCalledTimes(1);
    expect(handLandmarker.getGraphRunner().addGpuBufferAsImageToStream)
        .toHaveBeenCalledTimes(1);
    expect(handLandmarker.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();

    expect(landmarks).toEqual({
      'landmarks': [[{'x': 0, 'y': 0, 'z': 0, 'visibility': 0}]],
      'worldLandmarks': [[{'x': 0, 'y': 0, 'z': 0, 'visibility': 0}]],
      'handedness': [[{
        'score': 0.1,
        'index': 1,
        'categoryName': 'handedness_label',
        'displayName': 'handedness_display_name'
      }]],
      'handednesses': [[{
        'score': 0.1,
        'index': 1,
        'categoryName': 'handedness_label',
        'displayName': 'handedness_display_name'
      }]]
    });
  });

  it('clears results between invoations', async () => {
    const landmarks = [createLandmarks().serializeBinary()];
    const worldLandmarks = [createWorldLandmarks().serializeBinary()];
    const handedness = [createHandedness().serializeBinary()];

    // Pass the test data to our listener
    handLandmarker.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      handLandmarker.listeners.get('hand_landmarks')!(landmarks, 1337);
      handLandmarker.listeners.get('world_hand_landmarks')!
          (worldLandmarks, 1337);
      handLandmarker.listeners.get('handedness')!(handedness, 1337);
    });

    // Invoke the hand landmarker twice
    const landmarks1 = handLandmarker.detect({} as HTMLImageElement);
    const landmarks2 = handLandmarker.detect({} as HTMLImageElement);

    // Verify that hands2 is not a concatenation of all previously returned
    // hands.
    expect(landmarks1).toEqual(landmarks2);
  });
});
