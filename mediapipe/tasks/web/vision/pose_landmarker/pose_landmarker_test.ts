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
import {createLandmarks, createWorldLandmarks} from '../../../../tasks/web/components/processors/landmark_result_test_lib';
import {addJasmineCustomFloatEqualityTester, createSpyWasmModule, MediapipeTasksFake, SpyWasmModule, verifyGraph} from '../../../../tasks/web/core/task_runner_test_utils';
import {MPMask} from '../../../../tasks/web/vision/core/mask';
import {VisionGraphRunner} from '../../../../tasks/web/vision/core/vision_task_runner';

import {PoseLandmarker} from './pose_landmarker';
import {PoseLandmarkerOptions} from './pose_landmarker_options';
import {PoseLandmarkerResult} from './pose_landmarker_result';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

type PacketListener = (data: unknown, timestamp: number) => void;

class PoseLandmarkerFake extends PoseLandmarker implements MediapipeTasksFake {
  calculatorName = 'mediapipe.tasks.vision.pose_landmarker.PoseLandmarkerGraph';
  attachListenerSpies: jasmine.Spy[] = [];
  graph: CalculatorGraphConfig|undefined;
  fakeWasmModule: SpyWasmModule;
  listeners = new Map<string, PacketListener>();

  constructor() {
    super(createSpyWasmModule(), /* glCanvas= */ null);
    this.fakeWasmModule =
        this.graphRunner.wasmModule as unknown as SpyWasmModule;

    this.attachListenerSpies[0] =
        spyOn(this.graphRunner, 'attachProtoVectorListener')
            .and.callFake((stream, listener) => {
              expect(stream).toMatch(/(normalized_landmarks|world_landmarks)/);
              this.listeners.set(stream, listener as PacketListener);
            });
    this.attachListenerSpies[1] =
        spyOn(this.graphRunner, 'attachImageVectorListener')
            .and.callFake((stream, listener) => {
              expect(stream).toEqual('segmentation_masks');
              this.listeners.set(stream, listener as PacketListener);
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

describe('PoseLandmarker', () => {
  let poseLandmarker: PoseLandmarkerFake;

  beforeEach(async () => {
    addJasmineCustomFloatEqualityTester();
    poseLandmarker = new PoseLandmarkerFake();
    await poseLandmarker.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array([])}});
  });

  it('initializes graph', async () => {
    verifyGraph(poseLandmarker);
    expect(poseLandmarker.listeners).toHaveSize(2);
  });

  it('reloads graph when settings are changed', async () => {
    await poseLandmarker.setOptions({numPoses: 1});
    verifyGraph(poseLandmarker, [['poseDetectorGraphOptions', 'numPoses'], 1]);
    expect(poseLandmarker.listeners).toHaveSize(2);

    await poseLandmarker.setOptions({numPoses: 5});
    verifyGraph(poseLandmarker, [['poseDetectorGraphOptions', 'numPoses'], 5]);
    expect(poseLandmarker.listeners).toHaveSize(2);
  });

  it('registers listener for segmentation masks', async () => {
    expect(poseLandmarker.listeners).toHaveSize(2);
    await poseLandmarker.setOptions({outputSegmentationMasks: true});
    expect(poseLandmarker.listeners).toHaveSize(3);
  });

  it('merges options', async () => {
    await poseLandmarker.setOptions({numPoses: 2});
    await poseLandmarker.setOptions({minPoseDetectionConfidence: 0.1});
    verifyGraph(poseLandmarker, [
      'poseDetectorGraphOptions', {
        numPoses: 2,
        baseOptions: undefined,
        minDetectionConfidence: 0.1,
        minSuppressionThreshold: 0.5
      }
    ]);
  });

  describe('setOptions()', () => {
    interface TestCase {
      optionPath: [keyof PoseLandmarkerOptions, ...string[]];
      fieldPath: string[];
      customValue: unknown;
      defaultValue: unknown;
    }

    const testCases: TestCase[] = [
      {
        optionPath: ['numPoses'],
        fieldPath: ['poseDetectorGraphOptions', 'numPoses'],
        customValue: 5,
        defaultValue: 1
      },
      {
        optionPath: ['minPoseDetectionConfidence'],
        fieldPath: ['poseDetectorGraphOptions', 'minDetectionConfidence'],
        customValue: 0.1,
        defaultValue: 0.5
      },
      {
        optionPath: ['minPosePresenceConfidence'],
        fieldPath:
            ['poseLandmarksDetectorGraphOptions', 'minDetectionConfidence'],
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
        path: string[], value: unknown): PoseLandmarkerOptions {
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
            poseLandmarker, [testCase.fieldPath, testCase.defaultValue]);
      });

      it(`can set ${testCase.optionPath[0]}`, async () => {
        await poseLandmarker.setOptions(
            createOptions(testCase.optionPath, testCase.customValue));
        verifyGraph(poseLandmarker, [testCase.fieldPath, testCase.customValue]);
      });

      it(`can clear ${testCase.optionPath[0]}`, async () => {
        await poseLandmarker.setOptions(
            createOptions(testCase.optionPath, testCase.customValue));
        verifyGraph(poseLandmarker, [testCase.fieldPath, testCase.customValue]);

        await poseLandmarker.setOptions(
            createOptions(testCase.optionPath, undefined));
        verifyGraph(
            poseLandmarker, [testCase.fieldPath, testCase.defaultValue]);
      });
    }
  });

  it('doesn\'t support region of interest', () => {
    expect(() => {
      poseLandmarker.detect(
          {} as HTMLImageElement,
          {regionOfInterest: {left: 0, right: 0, top: 0, bottom: 0}}, () => {});
    }).toThrowError('This task doesn\'t support region-of-interest.');
  });

  it('transforms results', (done) => {
    const landmarksProto = [createLandmarks().serializeBinary()];
    const worldLandmarksProto = [createWorldLandmarks().serializeBinary()];
    const masks = [
      {data: new Float32Array([0, 1, 2, 3]), width: 2, height: 2},
    ];

    poseLandmarker.setOptions({outputSegmentationMasks: true});

    // Pass the test data to our listener
    poseLandmarker.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      poseLandmarker.listeners.get('normalized_landmarks')!
          (landmarksProto, 1337);
      poseLandmarker.listeners.get('world_landmarks')!
          (worldLandmarksProto, 1337);
      poseLandmarker.listeners.get('segmentation_masks')!(masks, 1337);
    });

    // Invoke the pose landmarker
    poseLandmarker.detect({} as HTMLImageElement, result => {
      expect(poseLandmarker.getGraphRunner().addProtoToStream)
          .toHaveBeenCalledTimes(1);
      expect(poseLandmarker.getGraphRunner().addGpuBufferAsImageToStream)
          .toHaveBeenCalledTimes(1);
      expect(poseLandmarker.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();

      expect(result.landmarks).toEqual([
        [{'x': 0, 'y': 0, 'z': 0, 'visibility': 0}]
      ]);
      expect(result.worldLandmarks).toEqual([
        [{'x': 0, 'y': 0, 'z': 0, 'visibility': 0}]
      ]);
      expect(result.segmentationMasks![0]).toBeInstanceOf(MPMask);
      done();
    });
  });

  it('clears results between invoations', async () => {
    const landmarksProto = [createLandmarks().serializeBinary()];
    const worldLandmarksProto = [createWorldLandmarks().serializeBinary()];

    // Pass the test data to our listener
    poseLandmarker.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      poseLandmarker.listeners.get('normalized_landmarks')!
          (landmarksProto, 1337);
      poseLandmarker.listeners.get('world_landmarks')!
          (worldLandmarksProto, 1337);
    });

    // Invoke the pose landmarker twice
    let landmarks1: PoseLandmarkerResult|undefined;
    poseLandmarker.detect({} as HTMLImageElement, result => {
      landmarks1 = result;
    });

    let landmarks2: PoseLandmarkerResult|undefined;
    poseLandmarker.detect({} as HTMLImageElement, result => {
      landmarks2 = result;
    });

    // Verify that poses2 is not a concatenation of all previously returned
    // poses.
    expect(landmarks1).toBeDefined();
    expect(landmarks1).toEqual(landmarks2);
  });

  it('supports multiple poses', (done) => {
    const landmarksProto = [
      createLandmarks(0.1, 0.2, 0.3).serializeBinary(),
      createLandmarks(0.4, 0.5, 0.6).serializeBinary()
    ];
    const worldLandmarksProto = [
      createWorldLandmarks(1, 2, 3).serializeBinary(),
      createWorldLandmarks(4, 5, 6).serializeBinary()
    ];

    poseLandmarker.setOptions({numPoses: 1});

    // Pass the test data to our listener
    poseLandmarker.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      poseLandmarker.listeners.get('normalized_landmarks')!
          (landmarksProto, 1337);
      poseLandmarker.listeners.get('world_landmarks')!
          (worldLandmarksProto, 1337);
    });

    // Invoke the pose landmarker
    poseLandmarker.detect({} as HTMLImageElement, result => {
      expect(result.landmarks).toEqual([
        [{'x': 0.1, 'y': 0.2, 'z': 0.3, 'visibility': 0}],
        [{'x': 0.4, 'y': 0.5, 'z': 0.6, 'visibility': 0}]
      ]);
      expect(result.worldLandmarks).toEqual([
        [{'x': 1, 'y': 2, 'z': 3, 'visibility': 0}],
        [{'x': 4, 'y': 5, 'z': 6, 'visibility': 0}]
      ]);
      done();
    });
  });

  it('invokes listener after masks are available', (done) => {
    const landmarksProto = [createLandmarks().serializeBinary()];
    const worldLandmarksProto = [createWorldLandmarks().serializeBinary()];
    const masks = [
      {data: new Float32Array([0, 1, 2, 3]), width: 2, height: 2},
    ];
    let listenerCalled = false;


    poseLandmarker.setOptions({outputSegmentationMasks: true});

    // Pass the test data to our listener
    poseLandmarker.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      expect(listenerCalled).toBeFalse();
      poseLandmarker.listeners.get('normalized_landmarks')!
          (landmarksProto, 1337);
      expect(listenerCalled).toBeFalse();
      poseLandmarker.listeners.get('world_landmarks')!
          (worldLandmarksProto, 1337);
      expect(listenerCalled).toBeFalse();
      expect(listenerCalled).toBeFalse();
      poseLandmarker.listeners.get('segmentation_masks')!(masks, 1337);
    });

    // Invoke the pose landmarker
    poseLandmarker.detect({} as HTMLImageElement, () => {
      listenerCalled = true;
      done();
    });
  });

  it('returns result', () => {
    const landmarksProto = [createLandmarks().serializeBinary()];
    const worldLandmarksProto = [createWorldLandmarks().serializeBinary()];

    // Pass the test data to our listener
    poseLandmarker.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      poseLandmarker.listeners.get('normalized_landmarks')!
          (landmarksProto, 1337);
      poseLandmarker.listeners.get('world_landmarks')!
          (worldLandmarksProto, 1337);
    });

    // Invoke the pose landmarker
    const result = poseLandmarker.detect({} as HTMLImageElement);
    expect(poseLandmarker.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();
    expect(result.landmarks).toEqual([
      [{'x': 0, 'y': 0, 'z': 0, 'visibility': 0}]
    ]);
    expect(result.worldLandmarks).toEqual([
      [{'x': 0, 'y': 0, 'z': 0, 'visibility': 0}]
    ]);
    result.close();
  });
});
