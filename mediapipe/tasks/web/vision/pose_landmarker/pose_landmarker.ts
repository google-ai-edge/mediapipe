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
import {
  LandmarkList,
  NormalizedLandmarkList,
} from '../../../../framework/formats/landmark_pb';
import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {PoseDetectorGraphOptions} from '../../../../tasks/cc/vision/pose_detector/proto/pose_detector_graph_options_pb';
import {PoseLandmarkerGraphOptions} from '../../../../tasks/cc/vision/pose_landmarker/proto/pose_landmarker_graph_options_pb';
import {PoseLandmarksDetectorGraphOptions} from '../../../../tasks/cc/vision/pose_landmarker/proto/pose_landmarks_detector_graph_options_pb';
import {
  Landmark,
  NormalizedLandmark,
} from '../../../../tasks/web/components/containers/landmark';
import {
  convertToLandmarks,
  convertToWorldLandmarks,
} from '../../../../tasks/web/components/processors/landmark_result';
import {WasmFileset} from '../../../../tasks/web/core/wasm_fileset';
import {ImageProcessingOptions} from '../../../../tasks/web/vision/core/image_processing_options';
import {MPMask} from '../../../../tasks/web/vision/core/mask';
import {
  VisionGraphRunner,
  VisionTaskRunner,
} from '../../../../tasks/web/vision/core/vision_task_runner';
import {
  ImageSource,
  WasmModule,
} from '../../../../web/graph_runner/graph_runner';
// Placeholder for internal dependency on trusted resource url

import {PoseLandmarkerOptions} from './pose_landmarker_options';
import {PoseLandmarkerResult} from './pose_landmarker_result';
import {POSE_CONNECTIONS} from './pose_landmarks_connections';

export * from './pose_landmarker_options';
export * from './pose_landmarker_result';
export {type ImageSource};

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

const IMAGE_STREAM = 'image_in';
const NORM_RECT_STREAM = 'norm_rect';
const NORM_LANDMARKS_STREAM = 'normalized_landmarks';
const WORLD_LANDMARKS_STREAM = 'world_landmarks';
const SEGMENTATION_MASK_STREAM = 'segmentation_masks';
const POSE_LANDMARKER_GRAPH =
  'mediapipe.tasks.vision.pose_landmarker.PoseLandmarkerGraph';

const DEFAULT_NUM_POSES = 1;
const DEFAULT_SCORE_THRESHOLD = 0.5;
const DEFAULT_OUTPUT_SEGMENTATION_MASKS = false;

/**
 * A callback that receives the result from the pose detector. The returned
 * masks are only valid for the duration of the callback. If asynchronous
 * processing is needed, the masks need to be copied before the callback
 * returns.
 */
export type PoseLandmarkerCallback = (result: PoseLandmarkerResult) => void;

/** Performs pose landmarks detection on images. */
export class PoseLandmarker extends VisionTaskRunner {
  private landmarks: NormalizedLandmark[][] = [];
  private worldLandmarks: Landmark[][] = [];
  private segmentationMasks?: MPMask[];
  private outputSegmentationMasks = false;
  private userCallback?: PoseLandmarkerCallback;
  private readonly options: PoseLandmarkerGraphOptions;
  private readonly poseLandmarksDetectorGraphOptions: PoseLandmarksDetectorGraphOptions;
  private readonly poseDetectorGraphOptions: PoseDetectorGraphOptions;

  /**
   * An array containing the pairs of pose landmark indices to be rendered with
   * connections.
   * @export
   * @nocollapse
   */
  static POSE_CONNECTIONS = POSE_CONNECTIONS;

  /**
   * Initializes the Wasm runtime and creates a new `PoseLandmarker` from the
   * provided options.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param poseLandmarkerOptions The options for the PoseLandmarker.
   *     Note that either a path to the model asset or a model buffer needs to
   *     be provided (via `baseOptions`).
   */
  static createFromOptions(
    wasmFileset: WasmFileset,
    poseLandmarkerOptions: PoseLandmarkerOptions,
  ): Promise<PoseLandmarker> {
    return VisionTaskRunner.createVisionInstance(
      PoseLandmarker,
      wasmFileset,
      poseLandmarkerOptions,
    );
  }

  /**
   * Initializes the Wasm runtime and creates a new `PoseLandmarker` based on
   * the provided model asset buffer.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param modelAssetBuffer An array or a stream containing a binary
   *    representation of the model.
   */
  static createFromModelBuffer(
    wasmFileset: WasmFileset,
    modelAssetBuffer: Uint8Array|ReadableStreamDefaultReader,
  ): Promise<PoseLandmarker> {
    return VisionTaskRunner.createVisionInstance(PoseLandmarker, wasmFileset, {
      baseOptions: {modelAssetBuffer},
    });
  }

  /**
   * Initializes the Wasm runtime and creates a new `PoseLandmarker` based on
   * the path to the model asset.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   */
  static createFromModelPath(
    wasmFileset: WasmFileset,
    modelAssetPath: string,
  ): Promise<PoseLandmarker> {
    return VisionTaskRunner.createVisionInstance(PoseLandmarker, wasmFileset, {
      baseOptions: {modelAssetPath},
    });
  }

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

    this.options = new PoseLandmarkerGraphOptions();
    this.options.setBaseOptions(new BaseOptionsProto());
    this.poseLandmarksDetectorGraphOptions =
      new PoseLandmarksDetectorGraphOptions();
    this.options.setPoseLandmarksDetectorGraphOptions(
      this.poseLandmarksDetectorGraphOptions,
    );
    this.poseDetectorGraphOptions = new PoseDetectorGraphOptions();
    this.options.setPoseDetectorGraphOptions(this.poseDetectorGraphOptions);

    this.initDefaults();
  }

  protected override get baseOptions(): BaseOptionsProto {
    return this.options.getBaseOptions()!;
  }

  protected override set baseOptions(proto: BaseOptionsProto) {
    this.options.setBaseOptions(proto);
  }

  /**
   * Sets new options for this `PoseLandmarker`.
   *
   * Calling `setOptions()` with a subset of options only affects those options.
   * You can reset an option back to its default value by explicitly setting it
   * to `undefined`.
   *
   * @export
   * @param options The options for the pose landmarker.
   */
  override setOptions(options: PoseLandmarkerOptions): Promise<void> {
    // Configure pose detector options.
    if ('numPoses' in options) {
      this.poseDetectorGraphOptions.setNumPoses(
        options.numPoses ?? DEFAULT_NUM_POSES,
      );
    }
    if ('minPoseDetectionConfidence' in options) {
      this.poseDetectorGraphOptions.setMinDetectionConfidence(
        options.minPoseDetectionConfidence ?? DEFAULT_SCORE_THRESHOLD,
      );
    }

    // Configure pose landmark detector options.
    if ('minTrackingConfidence' in options) {
      this.options.setMinTrackingConfidence(
        options.minTrackingConfidence ?? DEFAULT_SCORE_THRESHOLD,
      );
    }
    if ('minPosePresenceConfidence' in options) {
      this.poseLandmarksDetectorGraphOptions.setMinDetectionConfidence(
        options.minPosePresenceConfidence ?? DEFAULT_SCORE_THRESHOLD,
      );
    }

    if ('outputSegmentationMasks' in options) {
      this.outputSegmentationMasks =
        options.outputSegmentationMasks ?? DEFAULT_OUTPUT_SEGMENTATION_MASKS;
    }

    return this.applyOptions(options);
  }

  /**
   * Performs pose detection on the provided single image and invokes the
   * callback with the response. The method returns synchronously once the
   * callback returns. Only use this method when the PoseLandmarker is created
   * with running mode `image`.
   *
   * @export
   * @param image An image to process.
   * @param callback The callback that is invoked with the result. The
   *    lifetime of the returned masks is only guaranteed for the duration of
   *    the callback.
   */
  detect(image: ImageSource, callback: PoseLandmarkerCallback): void;
  /**
   * Performs pose detection on the provided single image and invokes the
   * callback with the response. The method returns synchronously once the
   * callback returns. Only use this method when the PoseLandmarker is created
   * with running mode `image`.
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
    callback: PoseLandmarkerCallback,
  ): void;
  /**
   * Performs pose detection on the provided single image and waits
   * synchronously for the response. This method creates a copy of the resulting
   * masks and should not be used in high-throughput applications. Only
   * use this method when the PoseLandmarker is created with running mode
   * `image`.
   *
   * @export
   * @param image An image to process.
   * @return The landmarker result. Any masks are copied to avoid lifetime
   *     limits.
   * @return The detected pose landmarks.
   */
  detect(image: ImageSource): PoseLandmarkerResult;
  /**
   * Performs pose detection on the provided single image and waits
   * synchronously for the response. This method creates a copy of the resulting
   * masks and should not be used in high-throughput applications. Only
   * use this method when the PoseLandmarker is created with running mode
   * `image`.
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
  ): PoseLandmarkerResult;
  /** @export */
  detect(
    image: ImageSource,
    imageProcessingOptionsOrCallback?:
      | ImageProcessingOptions
      | PoseLandmarkerCallback,
    callback?: PoseLandmarkerCallback,
  ): PoseLandmarkerResult | void {
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
   * Performs pose detection on the provided video frame and invokes the
   * callback with the response. The method returns synchronously once the
   * callback returns. Only use this method when the PoseLandmarker is created
   * with running mode `video`.
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
    callback: PoseLandmarkerCallback,
  ): void;
  /**
   * Performs pose detection on the provided video frame and invokes the
   * callback with the response. The method returns synchronously once the
   * callback returns. Only use this method when the PoseLandmarker is created
   * with running mode `video`.
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
    callback: PoseLandmarkerCallback,
  ): void;
  /**
   * Performs pose detection on the provided video frame and returns the result.
   * This method creates a copy of the resulting masks and should not be used
   * in high-throughput applications. Only use this method when the
   * PoseLandmarker is created with running mode `video`.
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
  ): PoseLandmarkerResult;
  /**
   * Performs pose detection on the provided video frame and returns the result.
   * This method creates a copy of the resulting masks and should not be used
   * in high-throughput applications. The method returns synchronously once the
   * callback returns. Only use this method when the PoseLandmarker is created
   * with running mode `video`.
   *
   * @export
   * @param videoFrame A video frame to process.
   * @param timestamp The timestamp of the current frame, in ms.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @return The landmarker result. Any masks are copied to extend the lifetime
   *     of the returned data.
   */
  detectForVideo(
    videoFrame: ImageSource,
    timestamp: number,
    imageProcessingOptions: ImageProcessingOptions,
  ): PoseLandmarkerResult;
  /** @export */
  detectForVideo(
    videoFrame: ImageSource,
    timestamp: number,
    imageProcessingOptionsOrCallback?:
      | ImageProcessingOptions
      | PoseLandmarkerCallback,
    callback?: PoseLandmarkerCallback,
  ): PoseLandmarkerResult | void {
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
    this.landmarks = [];
    this.worldLandmarks = [];
    this.segmentationMasks = undefined;
  }

  private processResults(): PoseLandmarkerResult | void {
    try {
      const result = new PoseLandmarkerResult(
        this.landmarks,
        this.worldLandmarks,
        this.segmentationMasks,
      );
      if (this.userCallback) {
        this.userCallback(result);
      } else {
        return result;
      }
    } finally {
      // Free the image memory, now that we've finished our callback.
      this.freeKeepaliveStreams();
    }
  }

  /** Sets the default values for the graph. */
  private initDefaults(): void {
    this.poseDetectorGraphOptions.setNumPoses(DEFAULT_NUM_POSES);
    this.poseDetectorGraphOptions.setMinDetectionConfidence(
      DEFAULT_SCORE_THRESHOLD,
    );
    this.poseLandmarksDetectorGraphOptions.setMinDetectionConfidence(
      DEFAULT_SCORE_THRESHOLD,
    );
    this.options.setMinTrackingConfidence(DEFAULT_SCORE_THRESHOLD);
  }

  /**
   * Converts raw data into a landmark, and adds it to our landmarks list.
   */
  private addJsLandmarks(data: Uint8Array[]): void {
    this.landmarks = [];
    for (const binaryProto of data) {
      const poseLandmarksProto =
        NormalizedLandmarkList.deserializeBinary(binaryProto);
      this.landmarks.push(convertToLandmarks(poseLandmarksProto));
    }
  }

  /**
   * Converts raw data into a world landmark, and adds it to our
   * worldLandmarks list.
   */
  private addJsWorldLandmarks(data: Uint8Array[]): void {
    this.worldLandmarks = [];
    for (const binaryProto of data) {
      const poseWorldLandmarksProto =
        LandmarkList.deserializeBinary(binaryProto);
      this.worldLandmarks.push(
        convertToWorldLandmarks(poseWorldLandmarksProto),
      );
    }
  }

  /** Updates the MediaPipe graph configuration. */
  protected override refreshGraph(): void {
    const graphConfig = new CalculatorGraphConfig();
    graphConfig.addInputStream(IMAGE_STREAM);
    graphConfig.addInputStream(NORM_RECT_STREAM);
    graphConfig.addOutputStream(NORM_LANDMARKS_STREAM);
    graphConfig.addOutputStream(WORLD_LANDMARKS_STREAM);
    graphConfig.addOutputStream(SEGMENTATION_MASK_STREAM);

    const calculatorOptions = new CalculatorOptions();
    calculatorOptions.setExtension(
      PoseLandmarkerGraphOptions.ext,
      this.options,
    );

    const landmarkerNode = new CalculatorGraphConfig.Node();
    landmarkerNode.setCalculator(POSE_LANDMARKER_GRAPH);
    landmarkerNode.addInputStream('IMAGE:' + IMAGE_STREAM);
    landmarkerNode.addInputStream('NORM_RECT:' + NORM_RECT_STREAM);
    landmarkerNode.addOutputStream('NORM_LANDMARKS:' + NORM_LANDMARKS_STREAM);
    landmarkerNode.addOutputStream('WORLD_LANDMARKS:' + WORLD_LANDMARKS_STREAM);
    landmarkerNode.setOptions(calculatorOptions);

    graphConfig.addNode(landmarkerNode);
    // We only need to keep alive the image stream, since the protos are being
    // deep-copied anyways via serialization+deserialization.
    this.addKeepaliveNode(graphConfig);

    this.graphRunner.attachProtoVectorListener(
      NORM_LANDMARKS_STREAM,
      (binaryProto, timestamp) => {
        this.addJsLandmarks(binaryProto);
        this.setLatestOutputTimestamp(timestamp);
      },
    );
    this.graphRunner.attachEmptyPacketListener(
      NORM_LANDMARKS_STREAM,
      (timestamp) => {
        this.landmarks = [];
        this.setLatestOutputTimestamp(timestamp);
      },
    );

    this.graphRunner.attachProtoVectorListener(
      WORLD_LANDMARKS_STREAM,
      (binaryProto, timestamp) => {
        this.addJsWorldLandmarks(binaryProto);
        this.setLatestOutputTimestamp(timestamp);
      },
    );
    this.graphRunner.attachEmptyPacketListener(
      WORLD_LANDMARKS_STREAM,
      (timestamp) => {
        this.worldLandmarks = [];
        this.setLatestOutputTimestamp(timestamp);
      },
    );

    if (this.outputSegmentationMasks) {
      landmarkerNode.addOutputStream(
        'SEGMENTATION_MASK:' + SEGMENTATION_MASK_STREAM,
      );
      this.keepStreamAlive(SEGMENTATION_MASK_STREAM);

      this.graphRunner.attachImageVectorListener(
        SEGMENTATION_MASK_STREAM,
        (masks, timestamp) => {
          this.segmentationMasks = masks.map((wasmImage) =>
            this.convertToMPMask(
              wasmImage,
              /* interpolateValues= */ true,
              /* shouldCopyData= */ !this.userCallback,
            ),
          );
          this.setLatestOutputTimestamp(timestamp);
        },
      );
      this.graphRunner.attachEmptyPacketListener(
        SEGMENTATION_MASK_STREAM,
        (timestamp) => {
          this.segmentationMasks = [];
          this.setLatestOutputTimestamp(timestamp);
        },
      );
    }

    const binaryGraph = graphConfig.serializeBinary();
    this.setGraph(new Uint8Array(binaryGraph), /* isBinary= */ true);
  }
}


