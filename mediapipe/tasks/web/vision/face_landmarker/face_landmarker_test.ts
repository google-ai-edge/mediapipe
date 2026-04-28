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
import {MatrixData as MatrixDataProto} from '../../../../framework/formats/matrix_data_pb';
import {FaceGeometry as FaceGeometryProto} from '../../../../tasks/cc/vision/face_geometry/proto/face_geometry_pb';
import {createLandmarks} from '../../../../tasks/web/components/processors/landmark_result_test_lib';
import {addJasmineCustomFloatEqualityTester, createSpyWasmModule, MediapipeTasksFake, SpyWasmModule, verifyGraph, verifyListenersRegistered} from '../../../../tasks/web/core/task_runner_test_utils';
import {VisionGraphRunner} from '../../../../tasks/web/vision/core/vision_task_runner';

import {FaceLandmarker} from './face_landmarker';
import {FaceLandmarkerOptions} from './face_landmarker_options';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

type ProtoListener = ((binaryProtos: Uint8Array[], timestamp: number) => void);

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

function createFacialTransformationMatrixes(): FaceGeometryProto {
  const faceGeometryProto = new FaceGeometryProto();
  const posteTransformationMatrix = new MatrixDataProto();
  posteTransformationMatrix.setRows(1);
  posteTransformationMatrix.setCols(1);
  posteTransformationMatrix.setPackedDataList([1.0]);
  faceGeometryProto.setPoseTransformMatrix(posteTransformationMatrix);
  return faceGeometryProto;
}

class FaceLandmarkerFake extends FaceLandmarker implements MediapipeTasksFake {
  calculatorName = 'mediapipe.tasks.vision.face_landmarker.FaceLandmarkerGraph';
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
                  /(face_landmarks|blendshapes|face_geometry)/);
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

describe('FaceLandmarker', () => {
  let faceLandmarker: FaceLandmarkerFake;

  beforeEach(async () => {
    addJasmineCustomFloatEqualityTester();
    faceLandmarker = new FaceLandmarkerFake();
    await faceLandmarker.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array([])}});
  });

  afterEach(() => {
    faceLandmarker.close();
  });

  it('initializes graph', async () => {
    verifyGraph(faceLandmarker);
    verifyListenersRegistered(faceLandmarker);
  });

  it('reloads graph when settings are changed', async () => {
    verifyListenersRegistered(faceLandmarker);

    await faceLandmarker.setOptions({numFaces: 1});
    verifyGraph(faceLandmarker, [['faceDetectorGraphOptions', 'numFaces'], 1]);
    verifyListenersRegistered(faceLandmarker);

    await faceLandmarker.setOptions({numFaces: 5});
    verifyGraph(faceLandmarker, [['faceDetectorGraphOptions', 'numFaces'], 5]);
    verifyListenersRegistered(faceLandmarker);
  });

  it('merges options', async () => {
    await faceLandmarker.setOptions({numFaces: 1});
    await faceLandmarker.setOptions({minFaceDetectionConfidence: 0.5});
    verifyGraph(faceLandmarker, [
      'faceDetectorGraphOptions', {
        numFaces: 1,
        baseOptions: undefined,
        minDetectionConfidence: 0.5,
        minSuppressionThreshold: 0.5
      }
    ]);
  });

  describe('setOptions()', () => {
    interface TestCase {
      optionPath: [keyof FaceLandmarkerOptions, ...string[]];
      fieldPath: string[];
      customValue: unknown;
      defaultValue: unknown;
    }

    const testCases: TestCase[] = [
      {
        optionPath: ['numFaces'],
        fieldPath: ['faceDetectorGraphOptions', 'numFaces'],
        customValue: 5,
        defaultValue: 1
      },
      {
        optionPath: ['minFaceDetectionConfidence'],
        fieldPath: ['faceDetectorGraphOptions', 'minDetectionConfidence'],
        customValue: 0.1,
        defaultValue: 0.5
      },
      {
        optionPath: ['minFacePresenceConfidence'],
        fieldPath:
            ['faceLandmarksDetectorGraphOptions', 'minDetectionConfidence'],
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
        path: string[], value: unknown): FaceLandmarkerOptions {
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
            faceLandmarker, [testCase.fieldPath, testCase.defaultValue]);
      });

      it(`can set ${testCase.optionPath[0]}`, async () => {
        await faceLandmarker.setOptions(
            createOptions(testCase.optionPath, testCase.customValue));
        verifyGraph(faceLandmarker, [testCase.fieldPath, testCase.customValue]);
      });

      it(`can clear ${testCase.optionPath[0]}`, async () => {
        await faceLandmarker.setOptions(
            createOptions(testCase.optionPath, testCase.customValue));
        verifyGraph(faceLandmarker, [testCase.fieldPath, testCase.customValue]);

        await faceLandmarker.setOptions(
            createOptions(testCase.optionPath, undefined));
        verifyGraph(
            faceLandmarker, [testCase.fieldPath, testCase.defaultValue]);
      });
    }

    it('supports outputFaceBlendshapes', async () => {
      const stream = 'blendshapes';
      await faceLandmarker.setOptions({});
      expect(faceLandmarker.graph!.getOutputStreamList()).not.toContain(stream);

      await faceLandmarker.setOptions({outputFaceBlendshapes: false});
      expect(faceLandmarker.graph!.getOutputStreamList()).not.toContain(stream);

      await faceLandmarker.setOptions({outputFaceBlendshapes: true});
      expect(faceLandmarker.graph!.getOutputStreamList()).toContain(stream);
    });

    it('supports outputFacialTransformationMatrixes', async () => {
      const stream = 'face_geometry';
      await faceLandmarker.setOptions({});
      expect(faceLandmarker.graph!.getOutputStreamList()).not.toContain(stream);

      await faceLandmarker.setOptions(
          {outputFacialTransformationMatrixes: false});
      expect(faceLandmarker.graph!.getOutputStreamList()).not.toContain(stream);

      await faceLandmarker.setOptions(
          {outputFacialTransformationMatrixes: true});
      expect(faceLandmarker.graph!.getOutputStreamList()).toContain(stream);
    });
  });

  it('doesn\'t support region of interest', () => {
    expect(() => {
      faceLandmarker.detect(
          {} as HTMLImageElement,
          {regionOfInterest: {left: 0, right: 0, top: 0, bottom: 0}});
    }).toThrowError('This task doesn\'t support region-of-interest.');
  });

  it('transforms results', async () => {
    const landmarksProto = [createLandmarks().serializeBinary()];
    const blendshapesProto = [createBlendshapes().serializeBinary()];
    const faceGeometryProto =
        [createFacialTransformationMatrixes().serializeBinary()];

    // Pass the test data to our listener
    faceLandmarker.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      verifyListenersRegistered(faceLandmarker);
      faceLandmarker.listeners.get('face_landmarks')!(landmarksProto, 1337);
      faceLandmarker.listeners.get('blendshapes')!(blendshapesProto, 1337);
      faceLandmarker.listeners.get('face_geometry')!(faceGeometryProto, 1337);
    });

    await faceLandmarker.setOptions({
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true
    });

    // Invoke the face landmarker
    const landmarks = faceLandmarker.detect({} as HTMLImageElement);
    expect(faceLandmarker.getGraphRunner().addProtoToStream)
        .toHaveBeenCalledTimes(1);
    expect(faceLandmarker.getGraphRunner().addGpuBufferAsImageToStream)
        .toHaveBeenCalledTimes(1);
    expect(faceLandmarker.fakeWasmModule._waitUntilIdle).toHaveBeenCalled();

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
      facialTransformationMatrixes: [({rows: 1, columns: 1, data: [1]})]
    });
  });

  it('clears results between invoations', async () => {
    const landmarksProto = [createLandmarks().serializeBinary()];
    const blendshapesProto = [createBlendshapes().serializeBinary()];
    const faceGeometryProto =
        [createFacialTransformationMatrixes().serializeBinary()];

    // Pass the test data to our listener
    faceLandmarker.fakeWasmModule._waitUntilIdle.and.callFake(() => {
      faceLandmarker.listeners.get('face_landmarks')!(landmarksProto, 1337);
      faceLandmarker.listeners.get('blendshapes')!(blendshapesProto, 1337);
      faceLandmarker.listeners.get('face_geometry')!(faceGeometryProto, 1337);
    });

    await faceLandmarker.setOptions({
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true
    });

    // Invoke the face landmarker twice
    const landmarks1 = faceLandmarker.detect({} as HTMLImageElement);
    const landmarks2 = faceLandmarker.detect({} as HTMLImageElement);

    // Verify that faces2 is not a concatenation of all previously returned
    // faces.
    expect(landmarks1).toEqual(landmarks2);
  });
});
