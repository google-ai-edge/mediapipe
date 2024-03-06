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

import {CalculatorGraphConfig} from '../../../../framework/calculator_pb';
import {Classification, ClassificationList} from '../../../../framework/formats/classification_pb';
import {HolisticLandmarkerGraphOptions} from '../../../../tasks/cc/vision/holistic_landmarker/proto/holistic_landmarker_graph_options_pb';
import {createLandmarks, createWorldLandmarks} from '../../../../tasks/web/components/processors/landmark_result_test_lib';
import {addJasmineCustomFloatEqualityTester, createSpyWasmModule, Deserializer, MediapipeTasksFake, SpyWasmModule, verifyGraph, verifyListenersRegistered} from '../../../../tasks/web/core/task_runner_test_utils';
import {VisionGraphRunner} from '../../../../tasks/web/vision/core/vision_task_runner';

import {HolisticLandmarker} from './holistic_landmarker';
import {HolisticLandmarkerOptions} from './holistic_landmarker_options';


// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

type ProtoListener = ((binaryProtos: Uint8Array, timestamp: number) => void);
const holisticLandmarkerDeserializer =
    (binaryProto =>
         HolisticLandmarkerGraphOptions.deserializeBinary(binaryProto)
             .toObject()) as Deserializer;

function createBlendshapes(): ClassificationList {
  const blendshapesProto = new ClassificationList();
  const classification = new Classification();
  classification.setScore(0.1);
  classification.setIndex(1);
  classification.setLabel('face_label');
  classification.setDisplayName('face_display_name');
  blendshapesProto.addClassification(classification);
  return blendshapesProto;
}

class HolisticLandmarkerFake extends HolisticLandmarker implements
    MediapipeTasksFake {
  calculatorName =
      'mediapipe.tasks.vision.holistic_landmarker.HolisticLandmarkerGraph';
  attachListenerSpies: jasmine.Spy[] = [];
  graph: CalculatorGraphConfig|undefined;
  fakeWasmModule: SpyWasmModule;
  listeners = new Map<string, ProtoListener>();

  constructor() {
    super(createSpyWasmModule(), /* glCanvas= */ null);
    this.fakeWasmModule =
        this.graphRunner.wasmModule as unknown as SpyWasmModule;

    this.attachListenerSpies[0] =
        spyOn(this.graphRunner, 'attachProtoListener')
            .and.callFake((stream, listener) => {
              expect(stream).toMatch(
                  /(pose_landmarks|pose_world_landmarks|pose_segmentation_mask|face_landmarks|extra_blendshapes|left_hand_landmarks|left_hand_world_landmarks|right_hand_landmarks|right_hand_world_landmarks)/);
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

describe('HolisticLandmarker', () => {
  let holisticLandmarker: HolisticLandmarkerFake;

  beforeEach(async () => {
    addJasmineCustomFloatEqualityTester();
    holisticLandmarker = new HolisticLandmarkerFake();
    await holisticLandmarker.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array([])}});
  });

  afterEach(() => {
    holisticLandmarker.close();
  });

  it('initializes graph', async () => {
    verifyGraph(holisticLandmarker);
    verifyGraph(
        holisticLandmarker, undefined, undefined,
        holisticLandmarkerDeserializer);
  });

  it('reloads graph when settings are changed', async () => {
    verifyListenersRegistered(holisticLandmarker);

    await holisticLandmarker.setOptions({minFaceDetectionConfidence: 0.6});
    verifyGraph(
        holisticLandmarker,
        [['faceDetectorGraphOptions', 'minDetectionConfidence'], 0.6],
        undefined, holisticLandmarkerDeserializer);
    verifyListenersRegistered(holisticLandmarker);

    await holisticLandmarker.setOptions({minFaceDetectionConfidence: 0.7});
    verifyGraph(
        holisticLandmarker,
        [['faceDetectorGraphOptions', 'minDetectionConfidence'], 0.7],
        undefined, holisticLandmarkerDeserializer);
    verifyListenersRegistered(holisticLandmarker);
  });

  it('merges options', async () => {
    await holisticLandmarker.setOptions({minFaceDetectionConfidence: 0.5});
    await holisticLandmarker.setOptions({minFaceSuppressionThreshold: 0.5});
    await holisticLandmarker.setOptions({minFacePresenceConfidence: 0.5});
    await holisticLandmarker.setOptions({minPoseDetectionConfidence: 0.5});
    await holisticLandmarker.setOptions({minPoseSuppressionThreshold: 0.5});
    await holisticLandmarker.setOptions({minPosePresenceConfidence: 0.5});
    await holisticLandmarker.setOptions({minHandLandmarksConfidence: 0.5});

    verifyGraph(
        holisticLandmarker,
        [
          'faceDetectorGraphOptions', {
            baseOptions: undefined,
            minDetectionConfidence: 0.5,
            minSuppressionThreshold: 0.5,
            numFaces: undefined
          }
        ],
        undefined, holisticLandmarkerDeserializer);
    verifyGraph(
        holisticLandmarker,
        [
          'faceLandmarksDetectorGraphOptions', {
            baseOptions: undefined,
            minDetectionConfidence: 0.5,
            smoothLandmarks: undefined,
            faceBlendshapesGraphOptions: undefined
          }
        ],
        undefined, holisticLandmarkerDeserializer);
    verifyGraph(
        holisticLandmarker,
        [
          'poseDetectorGraphOptions', {
            baseOptions: undefined,
            minDetectionConfidence: 0.5,
            minSuppressionThreshold: 0.5,
            numPoses: undefined
          }
        ],
        undefined, holisticLandmarkerDeserializer);
    verifyGraph(
        holisticLandmarker,
        [
          'poseLandmarksDetectorGraphOptions', {
            baseOptions: undefined,
            minDetectionConfidence: 0.5,
            smoothLandmarks: undefined
          }
        ],
        undefined, holisticLandmarkerDeserializer);
    verifyGraph(
        holisticLandmarker,
        [
          'handLandmarksDetectorGraphOptions',
          {baseOptions: undefined, minDetectionConfidence: 0.5}
        ],
        undefined, holisticLandmarkerDeserializer);
  });

  describe('setOptions()', () => {
    interface TestCase {
      optionPath: [keyof HolisticLandmarkerOptions, ...string[]];
      fieldPath: string[];
      customValue: unknown;
      defaultValue: unknown;
    }

    const testCases: TestCase[] = [
      {
        optionPath: ['minFaceDetectionConfidence'],
        fieldPath: ['faceDetectorGraphOptions', 'minDetectionConfidence'],
        customValue: 0.1,
        defaultValue: 0.5
      },
      {
        optionPath: ['minFaceSuppressionThreshold'],
        fieldPath: ['faceDetectorGraphOptions', 'minSuppressionThreshold'],
        customValue: 0.2,
        defaultValue: 0.3
      },
      {
        optionPath: ['minFacePresenceConfidence'],
        fieldPath:
            ['faceLandmarksDetectorGraphOptions', 'minDetectionConfidence'],
        customValue: 0.2,
        defaultValue: 0.5
      },
      {
        optionPath: ['minPoseDetectionConfidence'],
        fieldPath: ['poseDetectorGraphOptions', 'minDetectionConfidence'],
        customValue: 0.1,
        defaultValue: 0.5
      },
      {
        optionPath: ['minPoseSuppressionThreshold'],
        fieldPath: ['poseDetectorGraphOptions', 'minSuppressionThreshold'],
        customValue: 0.2,
        defaultValue: 0.3
      },
      {
        optionPath: ['minPosePresenceConfidence'],
        fieldPath:
            ['poseLandmarksDetectorGraphOptions', 'minDetectionConfidence'],
        customValue: 0.2,
        defaultValue: 0.5
      },
      {
        optionPath: ['minHandLandmarksConfidence'],
        fieldPath:
            ['handLandmarksDetectorGraphOptions', 'minDetectionConfidence'],
        customValue: 0.1,
        defaultValue: 0.5
      },
    ];

    /** Creates an options object that can be passed to setOptions() */
    function createOptions(
        path: string[], value: unknown): HolisticLandmarkerOptions {
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
            holisticLandmarker, [testCase.fieldPath, testCase.defaultValue],
            undefined, holisticLandmarkerDeserializer);
      });

      it(`can set ${testCase.optionPath[0]}`, async () => {
        await holisticLandmarker.setOptions(
            createOptions(testCase.optionPath, testCase.customValue));
        verifyGraph(
            holisticLandmarker, [testCase.fieldPath, testCase.customValue],
            undefined, holisticLandmarkerDeserializer);
      });

      it(`can clear ${testCase.optionPath[0]}`, async () => {
        await holisticLandmarker.setOptions(
            createOptions(testCase.optionPath, testCase.customValue));
        verifyGraph(
            holisticLandmarker, [testCase.fieldPath, testCase.customValue],
            undefined, holisticLandmarkerDeserializer);

        await holisticLandmarker.setOptions(
            createOptions(testCase.optionPath, undefined));
        verifyGraph(
            holisticLandmarker, [testCase.fieldPath, testCase.defaultValue],
            undefined, holisticLandmarkerDeserializer);
      });
    }
  });

  it('supports outputFaceBlendshapes', async () => {
    const stream = 'extra_blendshapes';
    await holisticLandmarker.setOptions({});
    expect(holisticLandmarker.graph!.getOutputStreamList())
        .not.toContain(stream);

    await holisticLandmarker.setOptions({outputFaceBlendshapes: false});
    expect(holisticLandmarker.graph!.getOutputStreamList())
        .not.toContain(stream);

    await holisticLandmarker.setOptions({outputFaceBlendshapes: true});
    expect(holisticLandmarker.graph!.getOutputStreamList()).toContain(stream);
  });

  it('transforms results', async () => {
    const faceLandmarksProto = createLandmarks().serializeBinary();
    const blendshapesProto = createBlendshapes().serializeBinary();

    const poseLandmarksProto = createLandmarks().serializeBinary();
    const poseWorldLandmarksProto = createWorldLandmarks().serializeBinary();

    const leftHandLandmarksProto = createLandmarks().serializeBinary();
    const leftHandWorldLandmarksProto =
        createWorldLandmarks().serializeBinary();
    const rightHandLandmarksProto = createLandmarks().serializeBinary();
    const rightHandWorldLandmarksProto =
        createWorldLandmarks().serializeBinary();

    await holisticLandmarker.setOptions(
        {outputFaceBlendshapes: true, outputPoseSegmentationMasks: false});

    // Pass the test data to our listener
    holisticLandmarker.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      verifyListenersRegistered(holisticLandmarker);
      holisticLandmarker.listeners.get('face_landmarks')!
          (faceLandmarksProto, 1337);
      holisticLandmarker.listeners.get('extra_blendshapes')!
          (blendshapesProto, 1337);

      holisticLandmarker.listeners.get('pose_landmarks')!
          (poseLandmarksProto, 1337);
      holisticLandmarker.listeners.get('pose_world_landmarks')!
          (poseWorldLandmarksProto, 1337);

      holisticLandmarker.listeners.get('left_hand_landmarks')!
          (leftHandLandmarksProto, 1337);
      holisticLandmarker.listeners.get('left_hand_world_landmarks')!
          (leftHandWorldLandmarksProto, 1337);

      holisticLandmarker.listeners.get('right_hand_landmarks')!
          (rightHandLandmarksProto, 1337);
      holisticLandmarker.listeners.get('right_hand_world_landmarks')!
          (rightHandWorldLandmarksProto, 1337);
    });

    // Invoke the holistic landmarker
    const landmarks = holisticLandmarker.detect({} as HTMLImageElement);
    expect(holisticLandmarker.getGraphRunner().addGpuBufferAsImageToStream)
        .toHaveBeenCalledTimes(1);
    expect(holisticLandmarker.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();

    expect(landmarks).toEqual({
      faceLandmarks: [[{x: 0, y: 0, z: 0, visibility: 0}]],
      faceBlendshapes: [{
        categories: [{
          index: 1,
          score: 0.1,
          categoryName: 'face_label',
          displayName: 'face_display_name'
        }],
        headIndex: -1,
        headName: ''
      }],
      poseLandmarks: [[{x: 0, y: 0, z: 0, visibility: 0}]],
      poseWorldLandmarks: [[{x: 0, y: 0, z: 0, visibility: 0}]],
      poseSegmentationMasks: [],
      leftHandLandmarks: [[{x: 0, y: 0, z: 0, visibility: 0}]],
      leftHandWorldLandmarks: [[{x: 0, y: 0, z: 0, visibility: 0}]],
      rightHandLandmarks: [[{x: 0, y: 0, z: 0, visibility: 0}]],
      rightHandWorldLandmarks: [[{x: 0, y: 0, z: 0, visibility: 0}]]
    });
  });

  it('clears results between invoations', async () => {
    const faceLandmarksProto = createLandmarks().serializeBinary();
    const poseLandmarksProto = createLandmarks().serializeBinary();
    const poseWorldLandmarksProto = createWorldLandmarks().serializeBinary();
    const leftHandLandmarksProto = createLandmarks().serializeBinary();
    const leftHandWorldLandmarksProto =
        createWorldLandmarks().serializeBinary();
    const rightHandLandmarksProto = createLandmarks().serializeBinary();
    const rightHandWorldLandmarksProto =
        createWorldLandmarks().serializeBinary();

    // Pass the test data to our listener
    holisticLandmarker.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      holisticLandmarker.listeners.get('face_landmarks')!
          (faceLandmarksProto, 1337);
      holisticLandmarker.listeners.get('pose_landmarks')!
          (poseLandmarksProto, 1337);
      holisticLandmarker.listeners.get('pose_world_landmarks')!
          (poseWorldLandmarksProto, 1337);
      holisticLandmarker.listeners.get('left_hand_landmarks')!
          (leftHandLandmarksProto, 1337);
      holisticLandmarker.listeners.get('left_hand_world_landmarks')!
          (leftHandWorldLandmarksProto, 1337);
      holisticLandmarker.listeners.get('right_hand_landmarks')!
          (rightHandLandmarksProto, 1337);
      holisticLandmarker.listeners.get('right_hand_world_landmarks')!
          (rightHandWorldLandmarksProto, 1337);
    });

    // Invoke the holistic landmarker twice
    const landmarks1 = holisticLandmarker.detect({} as HTMLImageElement);
    const landmarks2 = holisticLandmarker.detect({} as HTMLImageElement);

    // Verify that landmarks2 is not a concatenation of all previously returned
    // hands.
    expect(landmarks1).toEqual(landmarks2);
  });
});
