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

import {CalculatorGraphConfig} from '../../../../framework/calculator_pb';
import {CalculatorOptions} from '../../../../framework/calculator_options_pb';
import {ClassificationList} from '../../../../framework/formats/classification_pb';
import {
  LandmarkList,
  NormalizedLandmarkList,
} from '../../../../framework/formats/landmark_pb';
import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {HandDetectorGraphOptions} from '../../../../tasks/cc/vision/hand_detector/proto/hand_detector_graph_options_pb';
import {HandLandmarkerGraphOptions} from '../../../../tasks/cc/vision/hand_landmarker/proto/hand_landmarker_graph_options_pb';
import {HandLandmarksDetectorGraphOptions} from '../../../../tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options_pb';
import {Category} from '../../../../tasks/web/components/containers/category';
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
import {
  VisionGraphRunner,
  VisionTaskRunner,
} from '../../../../tasks/web/vision/core/vision_task_runner';
import {HAND_CONNECTIONS} from '../../../../tasks/web/vision/hand_landmarker/hand_landmarks_connections';
import {
  ImageSource,
  WasmModule,
} from '../../../../web/graph_runner/graph_runner';
// Placeholder for internal dependency on trusted resource url

import {HandLandmarkerOptions} from './hand_landmarker_options';
import {HandLandmarkerResult} from './hand_landmarker_result';

export * from './hand_landmarker_options';
export * from './hand_landmarker_result';
export {type ImageSource};

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

const IMAGE_STREAM = 'image_in';
const NORM_RECT_STREAM = 'norm_rect';
const LANDMARKS_STREAM = 'hand_landmarks';
const WORLD_LANDMARKS_STREAM = 'world_hand_landmarks';
const HANDEDNESS_STREAM = 'handedness';
const HAND_LANDMARKER_GRAPH =
  'mediapipe.tasks.vision.hand_landmarker.HandLandmarkerGraph';

const DEFAULT_NUM_HANDS = 1;
const DEFAULT_SCORE_THRESHOLD = 0.5;
const DEFAULT_CATEGORY_INDEX = -1;

/** Performs hand landmarks detection on images. */
export class HandLandmarker extends VisionTaskRunner {
  private landmarks: NormalizedLandmark[][] = [];
  private worldLandmarks: Landmark[][] = [];
  private handedness: Category[][] = [];

  private readonly options: HandLandmarkerGraphOptions;
  private readonly handLandmarksDetectorGraphOptions: HandLandmarksDetectorGraphOptions;
  private readonly handDetectorGraphOptions: HandDetectorGraphOptions;

  /**
   * An array containing the pairs of hand landmark indices to be rendered with
   * connections.
   * @export
   * @nocollapse
   */
  static HAND_CONNECTIONS = HAND_CONNECTIONS;

  /**
   * Initializes the Wasm runtime and creates a new `HandLandmarker` from the
   * provided options.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param handLandmarkerOptions The options for the HandLandmarker.
   *     Note that either a path to the model asset or a model buffer needs to
   *     be provided (via `baseOptions`).
   */
  static createFromOptions(
    wasmFileset: WasmFileset,
    handLandmarkerOptions: HandLandmarkerOptions,
  ): Promise<HandLandmarker> {
    return VisionTaskRunner.createVisionInstance(
      HandLandmarker,
      wasmFileset,
      handLandmarkerOptions,
    );
  }

  /**
   * Initializes the Wasm runtime and creates a new `HandLandmarker` based on
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
  ): Promise<HandLandmarker> {
    return VisionTaskRunner.createVisionInstance(HandLandmarker, wasmFileset, {
      baseOptions: {modelAssetBuffer},
    });
  }

  /**
   * Initializes the Wasm runtime and creates a new `HandLandmarker` based on
   * the path to the model asset.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   */
  static createFromModelPath(
    wasmFileset: WasmFileset,
    modelAssetPath: string,
  ): Promise<HandLandmarker> {
    return VisionTaskRunner.createVisionInstance(HandLandmarker, wasmFileset, {
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

    this.options = new HandLandmarkerGraphOptions();
    this.options.setBaseOptions(new BaseOptionsProto());
    this.handLandmarksDetectorGraphOptions =
      new HandLandmarksDetectorGraphOptions();
    this.options.setHandLandmarksDetectorGraphOptions(
      this.handLandmarksDetectorGraphOptions,
    );
    this.handDetectorGraphOptions = new HandDetectorGraphOptions();
    this.options.setHandDetectorGraphOptions(this.handDetectorGraphOptions);

    this.initDefaults();
  }

  protected override get baseOptions(): BaseOptionsProto {
    return this.options.getBaseOptions()!;
  }

  protected override set baseOptions(proto: BaseOptionsProto) {
    this.options.setBaseOptions(proto);
  }

  /**
   * Sets new options for this `HandLandmarker`.
   *
   * Calling `setOptions()` with a subset of options only affects those options.
   * You can reset an option back to its default value by explicitly setting it
   * to `undefined`.
   *
   * @export
   * @param options The options for the hand landmarker.
   */
  override setOptions(options: HandLandmarkerOptions): Promise<void> {
    // Configure hand detector options.
    if ('numHands' in options) {
      this.handDetectorGraphOptions.setNumHands(
        options.numHands ?? DEFAULT_NUM_HANDS,
      );
    }
    if ('minHandDetectionConfidence' in options) {
      this.handDetectorGraphOptions.setMinDetectionConfidence(
        options.minHandDetectionConfidence ?? DEFAULT_SCORE_THRESHOLD,
      );
    }

    // Configure hand landmark detector options.
    if ('minTrackingConfidence' in options) {
      this.options.setMinTrackingConfidence(
        options.minTrackingConfidence ?? DEFAULT_SCORE_THRESHOLD,
      );
    }
    if ('minHandPresenceConfidence' in options) {
      this.handLandmarksDetectorGraphOptions.setMinDetectionConfidence(
        options.minHandPresenceConfidence ?? DEFAULT_SCORE_THRESHOLD,
      );
    }

    return this.applyOptions(options);
  }

  /**
   * Performs hand landmarks detection on the provided single image and waits
   * synchronously for the response. Only use this method when the
   * HandLandmarker is created with running mode `image`.
   *
   * @export
   * @param image An image to process.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @return The detected hand landmarks.
   */
  detect(
    image: ImageSource,
    imageProcessingOptions?: ImageProcessingOptions,
  ): HandLandmarkerResult {
    this.resetResults();
    this.processImageData(image, imageProcessingOptions);
    return this.processResults();
  }

  /**
   * Performs hand landmarks detection on the provided video frame and waits
   * synchronously for the response. Only use this method when the
   * HandLandmarker is created with running mode `video`.
   *
   * @export
   * @param videoFrame A video frame to process.
   * @param timestamp The timestamp of the current frame, in ms.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @return The detected hand landmarks.
   */
  detectForVideo(
    videoFrame: ImageSource,
    timestamp: number,
    imageProcessingOptions?: ImageProcessingOptions,
  ): HandLandmarkerResult {
    this.resetResults();
    this.processVideoData(videoFrame, imageProcessingOptions, timestamp);
    return this.processResults();
  }

  private resetResults(): void {
    this.landmarks = [];
    this.worldLandmarks = [];
    this.handedness = [];
  }

  private processResults(): HandLandmarkerResult {
    return {
      landmarks: this.landmarks,
      worldLandmarks: this.worldLandmarks,
      handednesses: this.handedness,
      handedness: this.handedness,
    };
  }

  /** Sets the default values for the graph. */
  private initDefaults(): void {
    this.handDetectorGraphOptions.setNumHands(DEFAULT_NUM_HANDS);
    this.handDetectorGraphOptions.setMinDetectionConfidence(
      DEFAULT_SCORE_THRESHOLD,
    );
    this.handLandmarksDetectorGraphOptions.setMinDetectionConfidence(
      DEFAULT_SCORE_THRESHOLD,
    );
    this.options.setMinTrackingConfidence(DEFAULT_SCORE_THRESHOLD);
  }

  /** Converts the proto data to a Category[][] structure. */
  private toJsCategories(data: Uint8Array[]): Category[][] {
    const result: Category[][] = [];
    for (const binaryProto of data) {
      const inputList = ClassificationList.deserializeBinary(binaryProto);
      const outputList: Category[] = [];
      for (const classification of inputList.getClassificationList()) {
        outputList.push({
          score: classification.getScore() ?? 0,
          index: classification.getIndex() ?? DEFAULT_CATEGORY_INDEX,
          categoryName: classification.getLabel() ?? '',
          displayName: classification.getDisplayName() ?? '',
        });
      }
      result.push(outputList);
    }
    return result;
  }

  /** Converts raw data into a landmark, and adds it to our landmarks list. */
  private addJsLandmarks(data: Uint8Array[]): void {
    for (const binaryProto of data) {
      const handLandmarksProto =
        NormalizedLandmarkList.deserializeBinary(binaryProto);
      this.landmarks.push(convertToLandmarks(handLandmarksProto));
    }
  }

  /**
   * Converts raw data into a world landmark, and adds it to our worldLandmarks
   * list.
   */
  private addJsWorldLandmarks(data: Uint8Array[]): void {
    for (const binaryProto of data) {
      const handWorldLandmarksProto =
        LandmarkList.deserializeBinary(binaryProto);
      this.worldLandmarks.push(
        convertToWorldLandmarks(handWorldLandmarksProto),
      );
    }
  }

  /** Updates the MediaPipe graph configuration. */
  protected override refreshGraph(): void {
    const graphConfig = new CalculatorGraphConfig();
    graphConfig.addInputStream(IMAGE_STREAM);
    graphConfig.addInputStream(NORM_RECT_STREAM);
    graphConfig.addOutputStream(LANDMARKS_STREAM);
    graphConfig.addOutputStream(WORLD_LANDMARKS_STREAM);
    graphConfig.addOutputStream(HANDEDNESS_STREAM);

    const calculatorOptions = new CalculatorOptions();
    calculatorOptions.setExtension(
      HandLandmarkerGraphOptions.ext,
      this.options,
    );

    const landmarkerNode = new CalculatorGraphConfig.Node();
    landmarkerNode.setCalculator(HAND_LANDMARKER_GRAPH);
    landmarkerNode.addInputStream('IMAGE:' + IMAGE_STREAM);
    landmarkerNode.addInputStream('NORM_RECT:' + NORM_RECT_STREAM);
    landmarkerNode.addOutputStream('LANDMARKS:' + LANDMARKS_STREAM);
    landmarkerNode.addOutputStream('WORLD_LANDMARKS:' + WORLD_LANDMARKS_STREAM);
    landmarkerNode.addOutputStream('HANDEDNESS:' + HANDEDNESS_STREAM);
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
        this.setLatestOutputTimestamp(timestamp);
      },
    );

    this.graphRunner.attachProtoVectorListener(
      HANDEDNESS_STREAM,
      (binaryProto, timestamp) => {
        this.handedness.push(...this.toJsCategories(binaryProto));
        this.setLatestOutputTimestamp(timestamp);
      },
    );
    this.graphRunner.attachEmptyPacketListener(
      HANDEDNESS_STREAM,
      (timestamp) => {
        this.setLatestOutputTimestamp(timestamp);
      },
    );

    const binaryGraph = graphConfig.serializeBinary();
    this.setGraph(new Uint8Array(binaryGraph), /* isBinary= */ true);
  }
}


