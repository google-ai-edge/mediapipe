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

import {CalculatorGraphConfig} from '../../../../framework/calculator_pb';
import {CalculatorOptions} from '../../../../framework/calculator_options_pb';
import {ClassificationList as ClassificationListProto} from '../../../../framework/formats/classification_pb';
import {NormalizedLandmarkList as NormalizedLandmarkListProto} from '../../../../framework/formats/landmark_pb';
import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {FaceDetectorGraphOptions} from '../../../../tasks/cc/vision/face_detector/proto/face_detector_graph_options_pb';
import {FaceGeometry as FaceGeometryProto} from '../../../../tasks/cc/vision/face_geometry/proto/face_geometry_pb';
import {FaceLandmarkerGraphOptions} from '../../../../tasks/cc/vision/face_landmarker/proto/face_landmarker_graph_options_pb';
import {FaceLandmarksDetectorGraphOptions} from '../../../../tasks/cc/vision/face_landmarker/proto/face_landmarks_detector_graph_options_pb';
import {convertFromClassifications} from '../../../../tasks/web/components/processors/classifier_result';
import {convertToLandmarks} from '../../../../tasks/web/components/processors/landmark_result';
import {WasmFileset} from '../../../../tasks/web/core/wasm_fileset';
import {ImageProcessingOptions} from '../../../../tasks/web/vision/core/image_processing_options';
import {
  VisionGraphRunner,
  VisionTaskRunner,
} from '../../../../tasks/web/vision/core/vision_task_runner';
import {
  ImageSource,
  WasmModule,
} from '../../../../web/graph_runner/graph_runner';
// Placeholder for internal dependency on trusted resource url

import {FaceLandmarkerOptions} from './face_landmarker_options';
import {FaceLandmarkerResult} from './face_landmarker_result';
import {
  FACE_LANDMARKS_CONTOURS,
  FACE_LANDMARKS_FACE_OVAL,
  FACE_LANDMARKS_LEFT_EYE,
  FACE_LANDMARKS_LEFT_EYEBROW,
  FACE_LANDMARKS_LEFT_IRIS,
  FACE_LANDMARKS_LIPS,
  FACE_LANDMARKS_RIGHT_EYE,
  FACE_LANDMARKS_RIGHT_EYEBROW,
  FACE_LANDMARKS_RIGHT_IRIS,
  FACE_LANDMARKS_TESSELATION,
} from './face_landmarks_connections';

export * from './face_landmarker_options';
export * from './face_landmarker_result';
export {type ImageSource};

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

const IMAGE_STREAM = 'image_in';
const NORM_RECT_STREAM = 'norm_rect';
const LANDMARKS_STREAM = 'face_landmarks';
const BLENDSHAPES_STREAM = 'blendshapes';
const FACE_GEOMETRY_STREAM = 'face_geometry';
const FACE_LANDMARKER_GRAPH =
  'mediapipe.tasks.vision.face_landmarker.FaceLandmarkerGraph';

const DEFAULT_NUM_FACES = 1;
const DEFAULT_SCORE_THRESHOLD = 0.5;

/**
 * Performs face landmarks detection on images.
 *
 * This API expects a pre-trained face landmarker model asset bundle.
 */
export class FaceLandmarker extends VisionTaskRunner {
  private result: FaceLandmarkerResult = {
    faceLandmarks: [],
    faceBlendshapes: [],
    facialTransformationMatrixes: [],
  };
  private outputFaceBlendshapes = false;
  private outputFacialTransformationMatrixes = false;

  private readonly options: FaceLandmarkerGraphOptions;
  private readonly faceLandmarksDetectorGraphOptions: FaceLandmarksDetectorGraphOptions;
  private readonly faceDetectorGraphOptions: FaceDetectorGraphOptions;

  /**
   * Initializes the Wasm runtime and creates a new `FaceLandmarker` from the
   * provided options.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param faceLandmarkerOptions The options for the FaceLandmarker.
   *     Note that either a path to the model asset or a model buffer needs to
   *     be provided (via `baseOptions`).
   */
  static createFromOptions(
    wasmFileset: WasmFileset,
    faceLandmarkerOptions: FaceLandmarkerOptions,
  ): Promise<FaceLandmarker> {
    return VisionTaskRunner.createVisionInstance(
      FaceLandmarker,
      wasmFileset,
      faceLandmarkerOptions,
    );
  }

  /**
   * Initializes the Wasm runtime and creates a new `FaceLandmarker` based on
   * the provided model asset buffer.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param modelAssetBuffer An array or a stream containing a binary
   *    representation of the model.
   */
  static createFromModelBuffer(
    wasmFileset: WasmFileset,
    modelAssetBuffer: Uint8Array | ReadableStreamDefaultReader,
  ): Promise<FaceLandmarker> {
    return VisionTaskRunner.createVisionInstance(FaceLandmarker, wasmFileset, {
      baseOptions: {modelAssetBuffer},
    });
  }

  /**
   * Initializes the Wasm runtime and creates a new `FaceLandmarker` based on
   * the path to the model asset.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   */
  static createFromModelPath(
    wasmFileset: WasmFileset,
    modelAssetPath: string,
  ): Promise<FaceLandmarker> {
    return VisionTaskRunner.createVisionInstance(FaceLandmarker, wasmFileset, {
      baseOptions: {modelAssetPath},
    });
  }

  /**
   * Landmark connections to draw the connection between a face's lips.
   * @export
   * @nocollapse
   */
  static FACE_LANDMARKS_LIPS = FACE_LANDMARKS_LIPS;

  /**
   * Landmark connections to draw the connection between a face's left eye.
   * @export
   * @nocollapse
   */
  static FACE_LANDMARKS_LEFT_EYE = FACE_LANDMARKS_LEFT_EYE;

  /**
   * Landmark connections to draw the connection between a face's left eyebrow.
   * @export
   * @nocollapse
   */
  static FACE_LANDMARKS_LEFT_EYEBROW = FACE_LANDMARKS_LEFT_EYEBROW;

  /**
   * Landmark connections to draw the connection between a face's left iris.
   * @export
   * @nocollapse
   */
  static FACE_LANDMARKS_LEFT_IRIS = FACE_LANDMARKS_LEFT_IRIS;

  /**
   * Landmark connections to draw the connection between a face's right eye.
   * @export
   * @nocollapse
   */
  static FACE_LANDMARKS_RIGHT_EYE = FACE_LANDMARKS_RIGHT_EYE;

  /**
   * Landmark connections to draw the connection between a face's right
   * eyebrow.
   * @export
   * @nocollapse
   */
  static FACE_LANDMARKS_RIGHT_EYEBROW = FACE_LANDMARKS_RIGHT_EYEBROW;

  /**
   * Landmark connections to draw the connection between a face's right iris.
   * @export
   * @nocollapse
   */
  static FACE_LANDMARKS_RIGHT_IRIS = FACE_LANDMARKS_RIGHT_IRIS;

  /**
   * Landmark connections to draw the face's oval.
   * @export
   * @nocollapse
   */
  static FACE_LANDMARKS_FACE_OVAL = FACE_LANDMARKS_FACE_OVAL;

  /**
   * Landmark connections to draw the face's contour.
   * @export
   * @nocollapse
   */
  static FACE_LANDMARKS_CONTOURS = FACE_LANDMARKS_CONTOURS;

  /**
   * Landmark connections to draw the face's tesselation.
   * @export
   * @nocollapse
   */
  static FACE_LANDMARKS_TESSELATION = FACE_LANDMARKS_TESSELATION;

  /** @hideconstructor */
  constructor(
    wasmModule: WasmModule,
    glCanvas?: HTMLCanvasElement | OffscreenCanvas | null,
  ) {
    super(
      new VisionGraphRunner(wasmModule, glCanvas),
      IMAGE_STREAM,
      NORM_RECT_STREAM,
      /* roiAllowed= */ false,
    );

    this.options = new FaceLandmarkerGraphOptions();
    this.options.setBaseOptions(new BaseOptionsProto());
    this.faceLandmarksDetectorGraphOptions =
      new FaceLandmarksDetectorGraphOptions();
    this.options.setFaceLandmarksDetectorGraphOptions(
      this.faceLandmarksDetectorGraphOptions,
    );
    this.faceDetectorGraphOptions = new FaceDetectorGraphOptions();
    this.options.setFaceDetectorGraphOptions(this.faceDetectorGraphOptions);

    this.initDefaults();
  }

  protected override get baseOptions(): BaseOptionsProto {
    return this.options.getBaseOptions()!;
  }

  protected override set baseOptions(proto: BaseOptionsProto) {
    this.options.setBaseOptions(proto);
  }

  /**
   * Sets new options for this `FaceLandmarker`.
   *
   * Calling `setOptions()` with a subset of options only affects those options.
   * You can reset an option back to its default value by explicitly setting it
   * to `undefined`.
   *
   * @export
   * @param options The options for the face landmarker.
   */
  override setOptions(options: FaceLandmarkerOptions): Promise<void> {
    // Configure face detector options.
    if ('numFaces' in options) {
      this.faceDetectorGraphOptions.setNumFaces(
        options.numFaces ?? DEFAULT_NUM_FACES,
      );
    }
    if ('minFaceDetectionConfidence' in options) {
      this.faceDetectorGraphOptions.setMinDetectionConfidence(
        options.minFaceDetectionConfidence ?? DEFAULT_SCORE_THRESHOLD,
      );
    }

    // Configure face landmark detector options.
    if ('minTrackingConfidence' in options) {
      this.options.setMinTrackingConfidence(
        options.minTrackingConfidence ?? DEFAULT_SCORE_THRESHOLD,
      );
    }
    if ('minFacePresenceConfidence' in options) {
      this.faceLandmarksDetectorGraphOptions.setMinDetectionConfidence(
        options.minFacePresenceConfidence ?? DEFAULT_SCORE_THRESHOLD,
      );
    }

    if ('outputFaceBlendshapes' in options) {
      this.outputFaceBlendshapes = !!options.outputFaceBlendshapes;
    }

    if ('outputFacialTransformationMatrixes' in options) {
      this.outputFacialTransformationMatrixes =
        !!options.outputFacialTransformationMatrixes;
    }

    return this.applyOptions(options);
  }

  /**
   * Performs face landmarks detection on the provided single image and waits
   * synchronously for the response. Only use this method when the
   * FaceLandmarker is created with running mode `image`.
   *
   * @export
   * @param image An image to process.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @return The detected face landmarks.
   */
  detect(
    image: ImageSource,
    imageProcessingOptions?: ImageProcessingOptions,
  ): FaceLandmarkerResult {
    this.resetResults();
    this.processImageData(image, imageProcessingOptions);
    return this.result;
  }

  /**
   * Performs face landmarks detection on the provided video frame and waits
   * synchronously for the response. Only use this method when the
   * FaceLandmarker is created with running mode `video`.
   *
   * @export
   * @param videoFrame A video frame to process.
   * @param timestamp The timestamp of the current frame, in ms.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @return The detected face landmarks.
   */
  detectForVideo(
    videoFrame: ImageSource,
    timestamp: number,
    imageProcessingOptions?: ImageProcessingOptions,
  ): FaceLandmarkerResult {
    this.resetResults();
    this.processVideoData(videoFrame, imageProcessingOptions, timestamp);
    return this.result;
  }

  private resetResults(): void {
    this.result = {
      faceLandmarks: [],
      faceBlendshapes: [],
      facialTransformationMatrixes: [],
    };
  }

  /** Sets the default values for the graph. */
  private initDefaults(): void {
    this.faceDetectorGraphOptions.setNumFaces(DEFAULT_NUM_FACES);
    this.faceDetectorGraphOptions.setMinDetectionConfidence(
      DEFAULT_SCORE_THRESHOLD,
    );
    this.faceLandmarksDetectorGraphOptions.setMinDetectionConfidence(
      DEFAULT_SCORE_THRESHOLD,
    );
    this.options.setMinTrackingConfidence(DEFAULT_SCORE_THRESHOLD);
  }

  /** Adds new face landmark from the given proto. */
  private addJsLandmarks(data: Uint8Array[]): void {
    for (const binaryProto of data) {
      const faceLandmarksProto =
        NormalizedLandmarkListProto.deserializeBinary(binaryProto);
      this.result.faceLandmarks.push(convertToLandmarks(faceLandmarksProto));
    }
  }

  /** Adds new blendshapes from the given proto. */
  private addBlenshape(data: Uint8Array[]): void {
    if (!this.outputFaceBlendshapes) {
      return;
    }

    for (const binaryProto of data) {
      const classificationList =
        ClassificationListProto.deserializeBinary(binaryProto);
      this.result.faceBlendshapes.push(
        convertFromClassifications(
          classificationList.getClassificationList() ?? [],
        ),
      );
    }
  }

  /** Adds new transformation matrixes from the given proto. */
  private addFacialTransformationMatrixes(data: Uint8Array[]): void {
    if (!this.outputFacialTransformationMatrixes) {
      return;
    }

    for (const binaryProto of data) {
      const faceGeometryProto =
        FaceGeometryProto.deserializeBinary(binaryProto);
      const poseTransformMatrix = faceGeometryProto.getPoseTransformMatrix();
      if (poseTransformMatrix) {
        this.result.facialTransformationMatrixes.push({
          rows: poseTransformMatrix.getRows() ?? 0,
          columns: poseTransformMatrix.getCols() ?? 0,
          data: poseTransformMatrix.getPackedDataList().slice() ?? [],
        });
      }
    }
  }

  /** Updates the MediaPipe graph configuration. */
  protected override refreshGraph(): void {
    const graphConfig = new CalculatorGraphConfig();
    graphConfig.addInputStream(IMAGE_STREAM);
    graphConfig.addInputStream(NORM_RECT_STREAM);
    graphConfig.addOutputStream(LANDMARKS_STREAM);

    const calculatorOptions = new CalculatorOptions();
    calculatorOptions.setExtension(
      FaceLandmarkerGraphOptions.ext,
      this.options,
    );

    const landmarkerNode = new CalculatorGraphConfig.Node();
    landmarkerNode.setCalculator(FACE_LANDMARKER_GRAPH);
    landmarkerNode.addInputStream('IMAGE:' + IMAGE_STREAM);
    landmarkerNode.addInputStream('NORM_RECT:' + NORM_RECT_STREAM);
    landmarkerNode.addOutputStream('NORM_LANDMARKS:' + LANDMARKS_STREAM);
    landmarkerNode.setOptions(calculatorOptions);

    graphConfig.addNode(landmarkerNode);

    this.graphRunner.attachProtoVectorListener(
      LANDMARKS_STREAM,
      (binaryProto, timestamp) => {
        this.addJsLandmarks(binaryProto);
        this.setLatestOutputTimestamp(timestamp);
      },
    );
    this.graphRunner.attachEmptyPacketListener(
      LANDMARKS_STREAM,
      (timestamp) => {
        this.setLatestOutputTimestamp(timestamp);
      },
    );

    if (this.outputFaceBlendshapes) {
      graphConfig.addOutputStream(BLENDSHAPES_STREAM);
      landmarkerNode.addOutputStream('BLENDSHAPES:' + BLENDSHAPES_STREAM);
      this.graphRunner.attachProtoVectorListener(
        BLENDSHAPES_STREAM,
        (binaryProto, timestamp) => {
          this.addBlenshape(binaryProto);
          this.setLatestOutputTimestamp(timestamp);
        },
      );
      this.graphRunner.attachEmptyPacketListener(
        BLENDSHAPES_STREAM,
        (timestamp) => {
          this.setLatestOutputTimestamp(timestamp);
        },
      );
    }

    if (this.outputFacialTransformationMatrixes) {
      graphConfig.addOutputStream(FACE_GEOMETRY_STREAM);
      landmarkerNode.addOutputStream('FACE_GEOMETRY:' + FACE_GEOMETRY_STREAM);

      this.graphRunner.attachProtoVectorListener(
        FACE_GEOMETRY_STREAM,
        (binaryProto, timestamp) => {
          this.addFacialTransformationMatrixes(binaryProto);
          this.setLatestOutputTimestamp(timestamp);
        },
      );
      this.graphRunner.attachEmptyPacketListener(
        FACE_GEOMETRY_STREAM,
        (timestamp) => {
          this.setLatestOutputTimestamp(timestamp);
        },
      );
    }

    const binaryGraph = graphConfig.serializeBinary();
    this.setGraph(new Uint8Array(binaryGraph), /* isBinary= */ true);
  }
}


