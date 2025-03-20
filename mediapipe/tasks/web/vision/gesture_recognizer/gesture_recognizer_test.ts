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
import {Landmark, LandmarkList, NormalizedLandmark, NormalizedLandmarkList} from '../../../../framework/formats/landmark_pb';
import {addJasmineCustomFloatEqualityTester, createSpyWasmModule, MediapipeTasksFake, SpyWasmModule, verifyGraph, verifyListenersRegistered} from '../../../../tasks/web/core/task_runner_test_utils';
import {VisionGraphRunner} from '../../../../tasks/web/vision/core/vision_task_runner';

import {GestureRecognizer, GestureRecognizerOptions} from './gesture_recognizer';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

type ProtoListener = ((binaryProtos: Uint8Array[], timestamp: number) => void);

function createHandedness(): Uint8Array[] {
  const handsProto = new ClassificationList();
  const classification = new Classification();
  classification.setScore(0.1);
  classification.setIndex(1);
  classification.setLabel('handedness_label');
  classification.setDisplayName('handedness_display_name');
  handsProto.addClassification(classification);
  return [handsProto.serializeBinary()];
}

function createGestures(): Uint8Array[] {
  const gesturesProto = new ClassificationList();
  const classification = new Classification();
  classification.setScore(0.2);
  classification.setIndex(2);
  classification.setLabel('gesture_label');
  classification.setDisplayName('gesture_display_name');
  gesturesProto.addClassification(classification);
  return [gesturesProto.serializeBinary()];
}

function createLandmarks(): Uint8Array[] {
  const handLandmarksProto = new NormalizedLandmarkList();
  const landmark = new NormalizedLandmark();
  landmark.setX(0.3);
  landmark.setY(0.4);
  landmark.setZ(0.5);
  handLandmarksProto.addLandmark(landmark);
  return [handLandmarksProto.serializeBinary()];
}

function createWorldLandmarks(): Uint8Array[] {
  const handLandmarksProto = new LandmarkList();
  const landmark = new Landmark();
  landmark.setX(21);
  landmark.setY(22);
  landmark.setZ(23);
  handLandmarksProto.addLandmark(landmark);
  return [handLandmarksProto.serializeBinary()];
}

class GestureRecognizerFake extends GestureRecognizer implements
    MediapipeTasksFake {
  calculatorName =
      'mediapipe.tasks.vision.gesture_recognizer.GestureRecognizerGraph';
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
                  /(hand_landmarks|world_hand_landmarks|handedness|hand_gestures)/);
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

describe('GestureRecognizer', () => {
  let gestureRecognizer: GestureRecognizerFake;

  beforeEach(async () => {
    addJasmineCustomFloatEqualityTester();
    gestureRecognizer = new GestureRecognizerFake();
    await gestureRecognizer.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array([])}});
  });

  afterEach(() => {
    gestureRecognizer.close();
  });

  it('initializes graph', async () => {
    verifyGraph(gestureRecognizer);
    verifyListenersRegistered(gestureRecognizer);
  });

  it('reloads graph when settings are changed', async () => {
    await gestureRecognizer.setOptions({numHands: 1});
    verifyGraph(gestureRecognizer, [
      ['handLandmarkerGraphOptions', 'handDetectorGraphOptions', 'numHands'], 1
    ]);
    verifyListenersRegistered(gestureRecognizer);

    await gestureRecognizer.setOptions({numHands: 5});
    verifyGraph(gestureRecognizer, [
      ['handLandmarkerGraphOptions', 'handDetectorGraphOptions', 'numHands'], 5
    ]);
    verifyListenersRegistered(gestureRecognizer);
  });

  it('merges options', async () => {
    await gestureRecognizer.setOptions({numHands: 1});
    await gestureRecognizer.setOptions({minHandDetectionConfidence: 0.5});
    verifyGraph(gestureRecognizer, [
      ['handLandmarkerGraphOptions', 'handDetectorGraphOptions', 'numHands'], 1
    ]);
    verifyGraph(gestureRecognizer, [
      [
        'handLandmarkerGraphOptions', 'handDetectorGraphOptions',
        'minDetectionConfidence'
      ],
      0.5
    ]);
  });

  it('does not reset default values when not specified', async () => {
    await gestureRecognizer.setOptions({minHandDetectionConfidence: 0.5});
    await gestureRecognizer.setOptions({});
    verifyGraph(gestureRecognizer, [
      [
        'handLandmarkerGraphOptions', 'handDetectorGraphOptions',
        'minDetectionConfidence'
      ],
      0.5
    ]);
  });

  describe('setOptions()', () => {
    interface TestCase {
      optionPath: [keyof GestureRecognizerOptions, ...string[]];
      fieldPath: string[];
      customValue: unknown;
      defaultValue: unknown;
    }

    const testCases: TestCase[] = [
      {
        optionPath: ['numHands'],
        fieldPath: [
          'handLandmarkerGraphOptions', 'handDetectorGraphOptions', 'numHands'
        ],
        customValue: 5,
        defaultValue: 1
      },
      {
        optionPath: ['minHandDetectionConfidence'],
        fieldPath: [
          'handLandmarkerGraphOptions', 'handDetectorGraphOptions',
          'minDetectionConfidence'
        ],
        customValue: 0.1,
        defaultValue: 0.5
      },
      {
        optionPath: ['minHandPresenceConfidence'],
        fieldPath: [
          'handLandmarkerGraphOptions', 'handLandmarksDetectorGraphOptions',
          'minDetectionConfidence'
        ],
        customValue: 0.2,
        defaultValue: 0.5
      },
      {
        optionPath: ['minTrackingConfidence'],
        fieldPath: ['handLandmarkerGraphOptions', 'minTrackingConfidence'],
        customValue: 0.3,
        defaultValue: 0.5
      },
      {
        optionPath: ['cannedGesturesClassifierOptions', 'scoreThreshold'],
        fieldPath: [
          'handGestureRecognizerGraphOptions',
          'cannedGestureClassifierGraphOptions', 'classifierOptions',
          'scoreThreshold'
        ],
        customValue: 0.4,
        defaultValue: undefined
      },
      {
        optionPath: ['customGesturesClassifierOptions', 'scoreThreshold'],
        fieldPath: [
          'handGestureRecognizerGraphOptions',
          'customGestureClassifierGraphOptions', 'classifierOptions',
          'scoreThreshold'
        ],
        customValue: 0.5,
        defaultValue: undefined,
      },
    ];

    /** Creates an options object that can be passed to setOptions() */
    function createOptions(
        path: string[], value: unknown): GestureRecognizerOptions {
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
            gestureRecognizer, [testCase.fieldPath, testCase.defaultValue]);
      });

      it(`can set ${testCase.optionPath[0]}`, async () => {
        await gestureRecognizer.setOptions(
            createOptions(testCase.optionPath, testCase.customValue));
        verifyGraph(
            gestureRecognizer, [testCase.fieldPath, testCase.customValue]);
      });

      it(`can clear ${testCase.optionPath[0]}`, async () => {
        await gestureRecognizer.setOptions(
            createOptions(testCase.optionPath, testCase.customValue));
        verifyGraph(
            gestureRecognizer, [testCase.fieldPath, testCase.customValue]);

        await gestureRecognizer.setOptions(
            createOptions(testCase.optionPath, undefined));
        verifyGraph(
            gestureRecognizer, [testCase.fieldPath, testCase.defaultValue]);
      });
    }
  });

  it('doesn\'t support region of interest', () => {
    expect(() => {
      gestureRecognizer.recognize(
          {} as HTMLImageElement,
          {regionOfInterest: {left: 0, right: 0, top: 0, bottom: 0}});
    }).toThrowError('This task doesn\'t support region-of-interest.');
  });

  it('transforms results', async () => {
    // Pass the test data to our listener
    gestureRecognizer.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      verifyListenersRegistered(gestureRecognizer);
      gestureRecognizer.listeners.get('hand_landmarks')!
          (createLandmarks(), 1337);
      gestureRecognizer.listeners.get('world_hand_landmarks')!
          (createWorldLandmarks(), 1337);
      gestureRecognizer.listeners.get('handedness')!(createHandedness(), 1337);
      gestureRecognizer.listeners.get('hand_gestures')!(createGestures(), 1337);
    });

    // Invoke the gesture recognizer
    const gestures = gestureRecognizer.recognize({} as HTMLImageElement);
    expect(gestureRecognizer.getGraphRunner().addProtoToStream)
        .toHaveBeenCalledTimes(1);
    expect(gestureRecognizer.getGraphRunner().addGpuBufferAsImageToStream)
        .toHaveBeenCalledTimes(1);
    expect(gestureRecognizer.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();

    expect(gestures).toEqual({
      'gestures': [[{
        'score': 0.2,
        'index': -1,
        'categoryName': 'gesture_label',
        'displayName': 'gesture_display_name'
      }]],
      'landmarks': [[{'x': 0.3, 'y': 0.4, 'z': 0.5, 'visibility': 0}]],
      'worldLandmarks': [[{'x': 21, 'y': 22, 'z': 23, 'visibility': 0}]],
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
    // Pass the test data to our listener
    gestureRecognizer.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      gestureRecognizer.listeners.get('hand_landmarks')!
          (createLandmarks(), 1337);
      gestureRecognizer.listeners.get('world_hand_landmarks')!
          (createWorldLandmarks(), 1337);
      gestureRecognizer.listeners.get('handedness')!(createHandedness(), 1337);
      gestureRecognizer.listeners.get('hand_gestures')!(createGestures(), 1337);
    });

    // Invoke the gesture recognizer twice
    const gestures1 = gestureRecognizer.recognize({} as HTMLImageElement);
    const gestures2 = gestureRecognizer.recognize({} as HTMLImageElement);

    // Verify that gestures2 is not a concatenation of all previously returned
    // gestures.
    expect(gestures2).toEqual(gestures1);
  });

  it('returns empty results when no gestures are detected', async () => {
    // Pass the test data to our listener
    gestureRecognizer.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      verifyListenersRegistered(gestureRecognizer);
      gestureRecognizer.listeners.get('hand_landmarks')!
          (createLandmarks(), 1337);
      gestureRecognizer.listeners.get('world_hand_landmarks')!
          (createWorldLandmarks(), 1337);
      gestureRecognizer.listeners.get('handedness')!(createHandedness(), 1337);
      gestureRecognizer.listeners.get('hand_gestures')!([], 1337);
    });

    // Invoke the gesture recognizer
    const gestures = gestureRecognizer.recognize({} as HTMLImageElement);
    expect(gestures).toEqual({
      'gestures': [],
      'landmarks': [],
      'worldLandmarks': [],
      'handedness': [],
      'handednesses': []
    });
  });
});
