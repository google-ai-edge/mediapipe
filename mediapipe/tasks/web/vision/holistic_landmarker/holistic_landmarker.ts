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

import {Any} from 'google-protobuf/google/protobuf/any_pb';
import {CalculatorGraphConfig} from '../../../../framework/calculator_pb';
import {CalculatorOptions} from '../../../../framework/calculator_options_pb';
import {ClassificationList as ClassificationListProto} from '../../../../framework/formats/classification_pb';
import {
  LandmarkList,
  NormalizedLandmarkList,
} from '../../../../framework/formats/landmark_pb';
import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {FaceDetectorGraphOptions} from '../../../../tasks/cc/vision/face_detector/proto/face_detector_graph_options_pb';
import {FaceLandmarksDetectorGraphOptions} from '../../../../tasks/cc/vision/face_landmarker/proto/face_landmarks_detector_graph_options_pb';
import {HandLandmarksDetectorGraphOptions} from '../../../../tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options_pb';
import {HandRoiRefinementGraphOptions} from '../../../../tasks/cc/vision/hand_landmarker/proto/hand_roi_refinement_graph_options_pb';
import {HolisticLandmarkerGraphOptions} from '../../../../tasks/cc/vision/holistic_landmarker/proto/holistic_landmarker_graph_options_pb';
import {PoseDetectorGraphOptions} from '../../../../tasks/cc/vision/pose_detector/proto/pose_detector_graph_options_pb';
import {PoseLandmarksDetectorGraphOptions} from '../../../../tasks/cc/vision/pose_landmarker/proto/pose_landmarks_detector_graph_options_pb';
import {Classifications} from '../../../../tasks/web/components/containers/classification_result';
import {
  Landmark,
  NormalizedLandmark,
} from '../../../../tasks/web/components/containers/landmark';
import {convertFromClassifications} from '../../../../tasks/web/components/processors/classifier_result';
import {
  convertToLandmarks,
  convertToWorldLandmarks,
} from '../../../../tasks/web/components/processors/landmark_result';
import {WasmFileset} from '../../../../tasks/web/core/wasm_fileset';
import {ImageProcessingOptions} from '../../../../tasks/web/vision/core/image_processing_options';
import {
  VisionGraphRunner,
  VisionTaskRunner,
} from '../../../../tasks/web/vision/core/vision_task_runner';
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
} from '../../../../tasks/web/vision/face_landmarker/face_landmarks_connections';
import {HAND_CONNECTIONS} from '../../../../tasks/web/vision/hand_landmarker/hand_landmarks_connections';
import {POSE_CONNECTIONS} from '../../../../tasks/web/vision/pose_landmarker/pose_landmarks_connections';
import {
  ImageSource,
  WasmModule,
} from '../../../../web/graph_runner/graph_runner';
// Placeholder for internal dependency on trusted resource url

import {HolisticLandmarkerOptions} from './holistic_landmarker_options';
import {HolisticLandmarkerResult} from './holistic_landmarker_result';

export * from './holistic_landmarker_options';
export * from './holistic_landmarker_result';
export {type ImageSource};

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

const IMAGE_STREAM = 'input_frames_image';

const POSE_LANDMARKS_STREAM = 'pose_landmarks';
const POSE_WORLD_LANDMARKS_STREAM = 'pose_world_landmarks';
const POSE_SEGMENTATION_MASK_STREAM = 'pose_segmentation_mask';
const FACE_LANDMARKS_STREAM = 'face_landmarks';
const FACE_BLENDSHAPES_STREAM = 'extra_blendshapes';
const LEFT_HAND_LANDMARKS_STREAM = 'left_hand_landmarks';
const LEFT_HAND_WORLD_LANDMARKS_STREAM = 'left_hand_world_landmarks';
const RIGHT_HAND_LANDMARKS_STREAM = 'right_hand_landmarks';
const RIGHT_HAND_WORLD_LANDMARKS_STREAM = 'right_hand_world_landmarks';

const HOLISTIC_LANDMARKER_GRAPH =
  'mediapipe.tasks.vision.holistic_landmarker.HolisticLandmarkerGraph';

const DEFAULT_SUPRESSION_THRESHOLD = 0.3;
const DEFAULT_SCORE_THRESHOLD = 0.5;

/**
 * A callback that receives the result from the holistic landmarker detection.
 * The returned result are only valid for the duration of the callback. If
 * asynchronous processing is needed, the masks need to be copied before the
 * callback returns.
 */
export type HolisticLandmarkerCallback = (
  result: HolisticLandmarkerResult,
) => void;

/** Performs holistic landmarks detection on images. */
export class HolisticLandmarker extends VisionTaskRunner {
  private result: HolisticLandmarkerResult = {
    faceLandmarks: [],
    faceBlendshapes: [],
    poseLandmarks: [],
    poseWorldLandmarks: [],
    poseSegmentationMasks: [],
    leftHandLandmarks: [],
    leftHandWorldLandmarks: [],
    rightHandLandmarks: [],
    rightHandWorldLandmarks: [],
  };
  private outputFaceBlendshapes = false;
  private outputPoseSegmentationMasks = false;
  private userCallback?: HolisticLandmarkerCallback;

  private readonly options: HolisticLandmarkerGraphOptions;
  private readonly handLandmarksDetectorGraphOptions: HandLandmarksDetectorGraphOptions;
  private readonly handRoiRefinementGraphOptions: HandRoiRefinementGraphOptions;
  private readonly faceDetectorGraphOptions: FaceDetectorGraphOptions;
  private readonly faceLandmarksDetectorGraphOptions: FaceLandmarksDetectorGraphOptions;
  private readonly poseDetectorGraphOptions: PoseDetectorGraphOptions;
  private readonly poseLandmarksDetectorGraphOptions: PoseLandmarksDetectorGraphOptions;

  /**
   * An array containing the pairs of hand landmark indices to be rendered with
   * connections.
   * @export
   * @nocollapse
   */
  static HAND_CONNECTIONS = HAND_CONNECTIONS;

  /**
   * An array containing the pairs of pose landmark indices to be rendered with
   * connections.
   * @export
   * @nocollapse
   */
  static POSE_CONNECTIONS = POSE_CONNECTIONS;

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

  /**
   * Initializes the Wasm runtime and creates a new `HolisticLandmarker` from
   * the provided options.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param holisticLandmarkerOptions The options for the HolisticLandmarker.
   *     Note that either a path to the model asset or a model buffer needs to
   *     be provided (via `baseOptions`).
   */
  static createFromOptions(
    wasmFileset: WasmFileset,
    holisticLandmarkerOptions: HolisticLandmarkerOptions,
  ): Promise<HolisticLandmarker> {
    return VisionTaskRunner.createVisionInstance(
      HolisticLandmarker,
      wasmFileset,
      holisticLandmarkerOptions,
    );
  }

  /**
   * Initializes the Wasm runtime and creates a new `HolisticLandmarker` based
   * on the provided model asset buffer.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param modelAssetBuffer An array or a stream containing a binary
   *    representation of the model.
   */
  static createFromModelBuffer(
    wasmFileset: WasmFileset,
    modelAssetBuffer: Uint8Array | ReadableStreamDefaultReader,
  ): Promise<HolisticLandmarker> {
    return VisionTaskRunner.createVisionInstance(
      HolisticLandmarker,
      wasmFileset,
      {baseOptions: {modelAssetBuffer}},
    );
  }

  /**
   * Initializes the Wasm runtime and creates a new `HolisticLandmarker` based
   * on the path to the model asset.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   */
  static createFromModelPath(
    wasmFileset: WasmFileset,
    modelAssetPath: string,
  ): Promise<HolisticLandmarker> {
    return VisionTaskRunner.createVisionInstance(
      HolisticLandmarker,
      wasmFileset,
      {baseOptions: {modelAssetPath}},
    );
  }

  /** @hideconstructor */
  constructor(
    wasmModule: WasmModule,
    glCanvas?: HTMLCanvasElement | OffscreenCanvas | null,
  ) {
    super(
      new VisionGraphRunner(wasmModule, glCanvas),
      IMAGE_STREAM,
      /* normRectStream= */ null,
      /* roiAllowed= */ false,
    );

    this.options = new HolisticLandmarkerGraphOptions();
    this.options.setBaseOptions(new BaseOptionsProto());
    this.handLandmarksDetectorGraphOptions =
      new HandLandmarksDetectorGraphOptions();
    this.options.setHandLandmarksDetectorGraphOptions(
      this.handLandmarksDetectorGraphOptions,
    );
    this.handRoiRefinementGraphOptions = new HandRoiRefinementGraphOptions();
    this.options.setHandRoiRefinementGraphOptions(
      this.handRoiRefinementGraphOptions,
    );
    this.faceDetectorGraphOptions = new FaceDetectorGraphOptions();
    this.options.setFaceDetectorGraphOptions(this.faceDetectorGraphOptions);
    this.faceLandmarksDetectorGraphOptions =
      new FaceLandmarksDetectorGraphOptions();
    this.options.setFaceLandmarksDetectorGraphOptions(
      this.faceLandmarksDetectorGraphOptions,
    );
    this.poseDetectorGraphOptions = new PoseDetectorGraphOptions();
    this.options.setPoseDetectorGraphOptions(this.poseDetectorGraphOptions);
    this.poseLandmarksDetectorGraphOptions =
      new PoseLandmarksDetectorGraphOptions();
    this.options.setPoseLandmarksDetectorGraphOptions(
      this.poseLandmarksDetectorGraphOptions,
    );

    this.initDefaults();
  }

  protected override get baseOptions(): BaseOptionsProto {
    return this.options.getBaseOptions()!;
  }

  protected override set baseOptions(proto: BaseOptionsProto) {
    this.options.setBaseOptions(proto);
  }

  /**
   * Sets new options for this `HolisticLandmarker`.
   *
   * Calling `setOptions()` with a subset of options only affects those options.
   * You can reset an option back to its default value by explicitly setting it
   * to `undefined`.
   *
   * @export
   * @param options The options for the holistic landmarker.
   */
  override setOptions(options: HolisticLandmarkerOptions): Promise<void> {
    // Configure face detector options.
    if ('minFaceDetectionConfidence' in options) {
      this.faceDetectorGraphOptions.setMinDetectionConfidence(
        options.minFaceDetectionConfidence ?? DEFAULT_SCORE_THRESHOLD,
      );
    }
    if ('minFaceSuppressionThreshold' in options) {
      this.faceDetectorGraphOptions.setMinSuppressionThreshold(
        options.minFaceSuppressionThreshold ?? DEFAULT_SUPRESSION_THRESHOLD,
      );
    }

    // Configure face landmark detector options.
    if ('minFacePresenceConfidence' in options) {
      this.faceLandmarksDetectorGraphOptions.setMinDetectionConfidence(
        options.minFacePresenceConfidence ?? DEFAULT_SCORE_THRESHOLD,
      );
    }
    if ('outputFaceBlendshapes' in options) {
      this.outputFaceBlendshapes = !!options.outputFaceBlendshapes;
    }

    // Configure pose detector options.
    if ('minPoseDetectionConfidence' in options) {
      this.poseDetectorGraphOptions.setMinDetectionConfidence(
        options.minPoseDetectionConfidence ?? DEFAULT_SCORE_THRESHOLD,
      );
    }
    if ('minPoseSuppressionThreshold' in options) {
      this.poseDetectorGraphOptions.setMinSuppressionThreshold(
        options.minPoseSuppressionThreshold ?? DEFAULT_SUPRESSION_THRESHOLD,
      );
    }

    // Configure pose landmark detector options.
    if ('minPosePresenceConfidence' in options) {
      this.poseLandmarksDetectorGraphOptions.setMinDetectionConfidence(
        options.minPosePresenceConfidence ?? DEFAULT_SCORE_THRESHOLD,
      );
    }
    if ('outputPoseSegmentationMasks' in options) {
      this.outputPoseSegmentationMasks = !!options.outputPoseSegmentationMasks;
    }

    // Configure hand detector options.
    if ('minHandLandmarksConfidence' in options) {
      this.handLandmarksDetectorGraphOptions.setMinDetectionConfidence(
        options.minHandLandmarksConfidence ?? DEFAULT_SCORE_THRESHOLD,
      );
    }
    return this.applyOptions(options);
  }

  /**
   * Performs holistic landmarks detection on the provided single image and
   * invokes the callback with the response. The method returns synchronously
   * once the callback returns. Only use this method when the HolisticLandmarker
   * is created with running mode `image`.
   *
   * @export
   * @param image An image to process.
   * @param callback The callback that is invoked with the result. The
   *    lifetime of the returned masks is only guaranteed for the duration of
   *    the callback.
   */
  detect(image: ImageSource, callback: HolisticLandmarkerCallback): void;
  /**
   * Performs holistic landmarks detection on the provided single image and
   * invokes the callback with the response. The method returns synchronously
   * once the callback returns. Only use this method when the HolisticLandmarker
   * is created with running mode `image`.
   *
   * @export
   * @param image An image to process.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @param callback The callback that is invoked with the result. The
   *    lifetime of the returned masks is only guaranteed for the duration of
   *    the callback.
   */
  detect(
    image: ImageSource,
    imageProcessingOptions: ImageProcessingOptions,
    callback: HolisticLandmarkerCallback,
  ): void;
  /**
   * Performs holistic landmarks detection on the provided single image and
   * waits synchronously for the response. This method creates a copy of the
   * resulting masks and should not be used in high-throughput applications.
   * Only use this method when the HolisticLandmarker is created with running
   * mode `image`.
   *
   * @export
   * @param image An image to process.
   * @return The landmarker result. Any masks are copied to avoid lifetime
   *     limits.
   * @return The detected pose landmarks.
   */
  detect(image: ImageSource): HolisticLandmarkerResult;
  /**
   * Performs holistic landmarks detection on the provided single image and
   * waits synchronously for the response. This method creates a copy of the
   * resulting masks and should not be used in high-throughput applications.
   * Only use this method when the HolisticLandmarker is created with running
   * mode `image`.
   *
   * @export
   * @param image An image to process.
   * @return The landmarker result. Any masks are copied to avoid lifetime
   *     limits.
   * @return The detected pose landmarks.
   */
  detect(
    image: ImageSource,
    imageProcessingOptions: ImageProcessingOptions,
  ): HolisticLandmarkerResult;
  /** @export */
  detect(
    image: ImageSource,
    imageProcessingOptionsOrCallback?:
      | ImageProcessingOptions
      | HolisticLandmarkerCallback,
    callback?: HolisticLandmarkerCallback,
  ): HolisticLandmarkerResult | void {
    const imageProcessingOptions =
      typeof imageProcessingOptionsOrCallback !== 'function'
        ? imageProcessingOptionsOrCallback
        : {};
    this.userCallback =
      typeof imageProcessingOptionsOrCallback === 'function'
        ? imageProcessingOptionsOrCallback
        : callback!;

    this.resetResults();
    this.processImageData(image, imageProcessingOptions);
    return this.processResults();
  }

  /**
   * Performs holistic landmarks detection on the provided video frame and
   * invokes the callback with the response. The method returns synchronously
   * once the callback returns. Only use this method when the HolisticLandmarker
   * is created with running mode `video`.
   *
   * @export
   * @param videoFrame A video frame to process.
   * @param timestamp The timestamp of the current frame, in ms.
   * @param callback The callback that is invoked with the result. The
   *    lifetime of the returned masks is only guaranteed for the duration of
   *    the callback.
   */
  detectForVideo(
    videoFrame: ImageSource,
    timestamp: number,
    callback: HolisticLandmarkerCallback,
  ): void;
  /**
   * Performs holistic landmarks detection on the provided video frame and
   * invokes the callback with the response. The method returns synchronously
   * once the callback returns. Only use this method when the holisticLandmarker
   * is created with running mode `video`.
   *
   * @export
   * @param videoFrame A video frame to process.
   * @param timestamp The timestamp of the current frame, in ms.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @param callback The callback that is invoked with the result. The
   *    lifetime of the returned masks is only guaranteed for the duration of
   *    the callback.
   */
  detectForVideo(
    videoFrame: ImageSource,
    timestamp: number,
    imageProcessingOptions: ImageProcessingOptions,
    callback: HolisticLandmarkerCallback,
  ): void;
  /**
   * Performs holistic landmarks detection on the provided video frame and
   * returns the result. This method creates a copy of the resulting masks and
   * should not be used in high-throughput applications. Only use this method
   * when the HolisticLandmarker is created with running mode `video`.
   *
   * @export
   * @param videoFrame A video frame to process.
   * @param timestamp The timestamp of the current frame, in ms.
   * @return The landmarker result. Any masks are copied to extend the
   *     lifetime of the returned data.
   */
  detectForVideo(
    videoFrame: ImageSource,
    timestamp: number,
  ): HolisticLandmarkerResult;
  /**
   * Performs holistic landmarks detection on the provided video frame and waits
   * synchronously for the response. Only use this method when the
   * HolisticLandmarker is created with running mode `video`.
   *
   * @export
   * @param videoFrame A video frame to process.
   * @param timestamp The timestamp of the current frame, in ms.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @return The detected holistic landmarks.
   */
  detectForVideo(
    videoFrame: ImageSource,
    timestamp: number,
    imageProcessingOptions: ImageProcessingOptions,
  ): HolisticLandmarkerResult;
  /** @export */
  detectForVideo(
    videoFrame: ImageSource,
    timestamp: number,
    imageProcessingOptionsOrCallback?:
      | ImageProcessingOptions
      | HolisticLandmarkerCallback,
    callback?: HolisticLandmarkerCallback,
  ): HolisticLandmarkerResult | void {
    const imageProcessingOptions =
      typeof imageProcessingOptionsOrCallback !== 'function'
        ? imageProcessingOptionsOrCallback
        : {};
    this.userCallback =
      typeof imageProcessingOptionsOrCallback === 'function'
        ? imageProcessingOptionsOrCallback
        : callback;

    this.resetResults();
    this.processVideoData(videoFrame, imageProcessingOptions, timestamp);
    return this.processResults();
  }

  private resetResults(): void {
    this.result = {
      faceLandmarks: [],
      faceBlendshapes: [],
      poseLandmarks: [],
      poseWorldLandmarks: [],
      poseSegmentationMasks: [],
      leftHandLandmarks: [],
      leftHandWorldLandmarks: [],
      rightHandLandmarks: [],
      rightHandWorldLandmarks: [],
    };
  }

  private processResults(): HolisticLandmarkerResult | void {
    try {
      if (this.userCallback) {
        this.userCallback(this.result);
      } else {
        return this.result;
      }
    } finally {
      // Free the image memory, now that we've finished our callback.
      this.freeKeepaliveStreams();
    }
  }

  /** Sets the default values for the graph. */
  private initDefaults(): void {
    this.faceDetectorGraphOptions.setMinDetectionConfidence(
      DEFAULT_SCORE_THRESHOLD,
    );
    this.faceDetectorGraphOptions.setMinSuppressionThreshold(
      DEFAULT_SUPRESSION_THRESHOLD,
    );

    this.faceLandmarksDetectorGraphOptions.setMinDetectionConfidence(
      DEFAULT_SCORE_THRESHOLD,
    );

    this.poseDetectorGraphOptions.setMinDetectionConfidence(
      DEFAULT_SCORE_THRESHOLD,
    );
    this.poseDetectorGraphOptions.setMinSuppressionThreshold(
      DEFAULT_SUPRESSION_THRESHOLD,
    );

    this.poseLandmarksDetectorGraphOptions.setMinDetectionConfidence(
      DEFAULT_SCORE_THRESHOLD,
    );

    this.handLandmarksDetectorGraphOptions.setMinDetectionConfidence(
      DEFAULT_SCORE_THRESHOLD,
    );
  }

  /** Converts raw data into a landmark, and adds it to our landmarks list. */
  private addJsLandmarks(
    data: Uint8Array,
    outputList: NormalizedLandmark[][],
  ): void {
    const landmarksProto = NormalizedLandmarkList.deserializeBinary(data);
    outputList.push(convertToLandmarks(landmarksProto));
  }

  /**
   * Converts raw data into a world landmark, and adds it to our worldLandmarks
   * list.
   */
  private addJsWorldLandmarks(
    data: Uint8Array,
    outputList: Landmark[][],
  ): void {
    const worldLandmarksProto = LandmarkList.deserializeBinary(data);
    outputList.push(convertToWorldLandmarks(worldLandmarksProto));
  }

  /** Adds new blendshapes from the given proto. */
  private addBlenshape(data: Uint8Array, outputList: Classifications[]): void {
    if (!this.outputFaceBlendshapes) {
      return;
    }
    const classificationList = ClassificationListProto.deserializeBinary(data);
    outputList.push(
      convertFromClassifications(
        classificationList.getClassificationList() ?? [],
      ),
    );
  }

  /** Updates the MediaPipe graph configuration. */
  protected override refreshGraph(): void {
    const graphConfig = new CalculatorGraphConfig();

    graphConfig.addInputStream(IMAGE_STREAM);
    graphConfig.addOutputStream(POSE_LANDMARKS_STREAM);
    graphConfig.addOutputStream(POSE_WORLD_LANDMARKS_STREAM);
    graphConfig.addOutputStream(FACE_LANDMARKS_STREAM);
    graphConfig.addOutputStream(LEFT_HAND_LANDMARKS_STREAM);
    graphConfig.addOutputStream(LEFT_HAND_WORLD_LANDMARKS_STREAM);
    graphConfig.addOutputStream(RIGHT_HAND_LANDMARKS_STREAM);
    graphConfig.addOutputStream(RIGHT_HAND_WORLD_LANDMARKS_STREAM);

    const calculatorOptions = new CalculatorOptions();
    const optionsProto = new Any();
    optionsProto.setTypeUrl(
      'type.googleapis.com/mediapipe.tasks.vision.holistic_landmarker.proto.HolisticLandmarkerGraphOptions',
    );
    optionsProto.setValue(this.options.serializeBinary());

    const landmarkerNode = new CalculatorGraphConfig.Node();
    landmarkerNode.setCalculator(HOLISTIC_LANDMARKER_GRAPH);
    landmarkerNode.addNodeOptions(optionsProto);

    landmarkerNode.addInputStream('IMAGE:' + IMAGE_STREAM);
    landmarkerNode.addOutputStream('POSE_LANDMARKS:' + POSE_LANDMARKS_STREAM);
    landmarkerNode.addOutputStream(
      'POSE_WORLD_LANDMARKS:' + POSE_WORLD_LANDMARKS_STREAM,
    );
    landmarkerNode.addOutputStream('FACE_LANDMARKS:' + FACE_LANDMARKS_STREAM);
    landmarkerNode.addOutputStream(
      'LEFT_HAND_LANDMARKS:' + LEFT_HAND_LANDMARKS_STREAM,
    );
    landmarkerNode.addOutputStream(
      'LEFT_HAND_WORLD_LANDMARKS:' + LEFT_HAND_WORLD_LANDMARKS_STREAM,
    );
    landmarkerNode.addOutputStream(
      'RIGHT_HAND_LANDMARKS:' + RIGHT_HAND_LANDMARKS_STREAM,
    );
    landmarkerNode.addOutputStream(
      'RIGHT_HAND_WORLD_LANDMARKS:' + RIGHT_HAND_WORLD_LANDMARKS_STREAM,
    );
    landmarkerNode.setOptions(calculatorOptions);

    graphConfig.addNode(landmarkerNode);
    // We only need to keep alive the image stream, since the protos are being
    // deep-copied anyways via serialization+deserialization.
    this.addKeepaliveNode(graphConfig);

    this.graphRunner.attachProtoListener(
      POSE_LANDMARKS_STREAM,
      (binaryProto, timestamp) => {
        this.addJsLandmarks(binaryProto, this.result.poseLandmarks);
        this.setLatestOutputTimestamp(timestamp);
      },
    );
    this.graphRunner.attachEmptyPacketListener(
      POSE_LANDMARKS_STREAM,
      (timestamp) => {
        this.setLatestOutputTimestamp(timestamp);
      },
    );

    this.graphRunner.attachProtoListener(
      POSE_WORLD_LANDMARKS_STREAM,
      (binaryProto, timestamp) => {
        this.addJsWorldLandmarks(binaryProto, this.result.poseWorldLandmarks);
        this.setLatestOutputTimestamp(timestamp);
      },
    );
    this.graphRunner.attachEmptyPacketListener(
      POSE_WORLD_LANDMARKS_STREAM,
      (timestamp) => {
        this.setLatestOutputTimestamp(timestamp);
      },
    );

    if (this.outputPoseSegmentationMasks) {
      landmarkerNode.addOutputStream(
        'POSE_SEGMENTATION_MASK:' + POSE_SEGMENTATION_MASK_STREAM,
      );
      this.keepStreamAlive(POSE_SEGMENTATION_MASK_STREAM);

      this.graphRunner.attachImageListener(
        POSE_SEGMENTATION_MASK_STREAM,
        (mask, timestamp) => {
          this.result.poseSegmentationMasks = [
            this.convertToMPMask(
              mask,
              /* interpolateValues= */ true,
              /* shouldCopyData= */ !this.userCallback,
            ),
          ];
          this.setLatestOutputTimestamp(timestamp);
        },
      );
      this.graphRunner.attachEmptyPacketListener(
        POSE_SEGMENTATION_MASK_STREAM,
        (timestamp) => {
          this.result.poseSegmentationMasks = [];
          this.setLatestOutputTimestamp(timestamp);
        },
      );
    }

    this.graphRunner.attachProtoListener(
      FACE_LANDMARKS_STREAM,
      (binaryProto, timestamp) => {
        this.addJsLandmarks(binaryProto, this.result.faceLandmarks);
        this.setLatestOutputTimestamp(timestamp);
      },
    );
    this.graphRunner.attachEmptyPacketListener(
      FACE_LANDMARKS_STREAM,
      (timestamp) => {
        this.setLatestOutputTimestamp(timestamp);
      },
    );

    if (this.outputFaceBlendshapes) {
      graphConfig.addOutputStream(FACE_BLENDSHAPES_STREAM);
      landmarkerNode.addOutputStream(
        'FACE_BLENDSHAPES:' + FACE_BLENDSHAPES_STREAM,
      );
      this.graphRunner.attachProtoListener(
        FACE_BLENDSHAPES_STREAM,
        (binaryProto, timestamp) => {
          this.addBlenshape(binaryProto, this.result.faceBlendshapes);
          this.setLatestOutputTimestamp(timestamp);
        },
      );
      this.graphRunner.attachEmptyPacketListener(
        FACE_BLENDSHAPES_STREAM,
        (timestamp) => {
          this.setLatestOutputTimestamp(timestamp);
        },
      );
    }

    this.graphRunner.attachProtoListener(
      LEFT_HAND_LANDMARKS_STREAM,
      (binaryProto, timestamp) => {
        this.addJsLandmarks(binaryProto, this.result.leftHandLandmarks);
        this.setLatestOutputTimestamp(timestamp);
      },
    );
    this.graphRunner.attachEmptyPacketListener(
      LEFT_HAND_LANDMARKS_STREAM,
      (timestamp) => {
        this.setLatestOutputTimestamp(timestamp);
      },
    );

    this.graphRunner.attachProtoListener(
      LEFT_HAND_WORLD_LANDMARKS_STREAM,
      (binaryProto, timestamp) => {
        this.addJsWorldLandmarks(
          binaryProto,
          this.result.leftHandWorldLandmarks,
        );
        this.setLatestOutputTimestamp(timestamp);
      },
    );
    this.graphRunner.attachEmptyPacketListener(
      LEFT_HAND_WORLD_LANDMARKS_STREAM,
      (timestamp) => {
        this.setLatestOutputTimestamp(timestamp);
      },
    );

    this.graphRunner.attachProtoListener(
      RIGHT_HAND_LANDMARKS_STREAM,
      (binaryProto, timestamp) => {
        this.addJsLandmarks(binaryProto, this.result.rightHandLandmarks);
        this.setLatestOutputTimestamp(timestamp);
      },
    );
    this.graphRunner.attachEmptyPacketListener(
      RIGHT_HAND_LANDMARKS_STREAM,
      (timestamp) => {
        this.setLatestOutputTimestamp(timestamp);
      },
    );

    this.graphRunner.attachProtoListener(
      RIGHT_HAND_WORLD_LANDMARKS_STREAM,
      (binaryProto, timestamp) => {
        this.addJsWorldLandmarks(
          binaryProto,
          this.result.rightHandWorldLandmarks,
        );
        this.setLatestOutputTimestamp(timestamp);
      },
    );
    this.graphRunner.attachEmptyPacketListener(
      RIGHT_HAND_WORLD_LANDMARKS_STREAM,
      (timestamp) => {
        this.setLatestOutputTimestamp(timestamp);
      },
    );

    const binaryGraph = graphConfig.serializeBinary();
    this.setGraph(new Uint8Array(binaryGraph), /* isBinary= */ true);
  }
}


