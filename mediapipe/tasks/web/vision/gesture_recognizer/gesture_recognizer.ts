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
import {GestureClassifierGraphOptions} from '../../../../tasks/cc/vision/gesture_recognizer/proto/gesture_classifier_graph_options_pb';
import {GestureRecognizerGraphOptions} from '../../../../tasks/cc/vision/gesture_recognizer/proto/gesture_recognizer_graph_options_pb';
import {HandGestureRecognizerGraphOptions} from '../../../../tasks/cc/vision/gesture_recognizer/proto/hand_gesture_recognizer_graph_options_pb';
import {HandDetectorGraphOptions} from '../../../../tasks/cc/vision/hand_detector/proto/hand_detector_graph_options_pb';
import {HandLandmarkerGraphOptions} from '../../../../tasks/cc/vision/hand_landmarker/proto/hand_landmarker_graph_options_pb';
import {HandLandmarksDetectorGraphOptions} from '../../../../tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options_pb';
import {Category} from '../../../../tasks/web/components/containers/category';
import {
  Landmark,
  NormalizedLandmark,
} from '../../../../tasks/web/components/containers/landmark';
import {convertClassifierOptionsToProto} from '../../../../tasks/web/components/processors/classifier_options';
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

import {GestureRecognizerOptions} from './gesture_recognizer_options';
import {GestureRecognizerResult} from './gesture_recognizer_result';

export * from './gesture_recognizer_options';
export * from './gesture_recognizer_result';
export {type ImageSource};

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

const IMAGE_STREAM = 'image_in';
const NORM_RECT_STREAM = 'norm_rect';
const HAND_GESTURES_STREAM = 'hand_gestures';
const LANDMARKS_STREAM = 'hand_landmarks';
const WORLD_LANDMARKS_STREAM = 'world_hand_landmarks';
const HANDEDNESS_STREAM = 'handedness';
const GESTURE_RECOGNIZER_GRAPH =
  'mediapipe.tasks.vision.gesture_recognizer.GestureRecognizerGraph';

const DEFAULT_NUM_HANDS = 1;
const DEFAULT_CONFIDENCE = 0.5;
const DEFAULT_CATEGORY_INDEX = -1;

/** Performs hand gesture recognition on images. */
export class GestureRecognizer extends VisionTaskRunner {
  private gestures: Category[][] = [];
  private landmarks: NormalizedLandmark[][] = [];
  private worldLandmarks: Landmark[][] = [];
  private handedness: Category[][] = [];

  private readonly options: GestureRecognizerGraphOptions;
  private readonly handLandmarkerGraphOptions: HandLandmarkerGraphOptions;
  private readonly handLandmarksDetectorGraphOptions: HandLandmarksDetectorGraphOptions;
  private readonly handDetectorGraphOptions: HandDetectorGraphOptions;
  private readonly handGestureRecognizerGraphOptions: HandGestureRecognizerGraphOptions;

  /**
   * An array containing the pairs of hand landmark indices to be rendered with
   * connections.
   * @export
   * @nocollapse
   */
  static HAND_CONNECTIONS = HAND_CONNECTIONS;

  /**
   * Initializes the Wasm runtime and creates a new gesture recognizer from the
   * provided options.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param gestureRecognizerOptions The options for the gesture recognizer.
   *     Note that either a path to the model asset or a model buffer needs to
   *     be provided (via `baseOptions`).
   */
  static createFromOptions(
    wasmFileset: WasmFileset,
    gestureRecognizerOptions: GestureRecognizerOptions,
  ): Promise<GestureRecognizer> {
    return VisionTaskRunner.createVisionInstance(
      GestureRecognizer,
      wasmFileset,
      gestureRecognizerOptions,
    );
  }

  /**
   * Initializes the Wasm runtime and creates a new gesture recognizer based on
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
  ): Promise<GestureRecognizer> {
    return VisionTaskRunner.createVisionInstance(
      GestureRecognizer,
      wasmFileset,
      {baseOptions: {modelAssetBuffer}},
    );
  }

  /**
   * Initializes the Wasm runtime and creates a new gesture recognizer based on
   * the path to the model asset.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   */
  static createFromModelPath(
    wasmFileset: WasmFileset,
    modelAssetPath: string,
  ): Promise<GestureRecognizer> {
    return VisionTaskRunner.createVisionInstance(
      GestureRecognizer,
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
      NORM_RECT_STREAM,
      /* roiAllowed= */ false,
    );

    this.options = new GestureRecognizerGraphOptions();
    this.options.setBaseOptions(new BaseOptionsProto());
    this.handLandmarkerGraphOptions = new HandLandmarkerGraphOptions();
    this.options.setHandLandmarkerGraphOptions(this.handLandmarkerGraphOptions);
    this.handLandmarksDetectorGraphOptions =
      new HandLandmarksDetectorGraphOptions();
    this.handLandmarkerGraphOptions.setHandLandmarksDetectorGraphOptions(
      this.handLandmarksDetectorGraphOptions,
    );
    this.handDetectorGraphOptions = new HandDetectorGraphOptions();
    this.handLandmarkerGraphOptions.setHandDetectorGraphOptions(
      this.handDetectorGraphOptions,
    );
    this.handGestureRecognizerGraphOptions =
      new HandGestureRecognizerGraphOptions();
    this.options.setHandGestureRecognizerGraphOptions(
      this.handGestureRecognizerGraphOptions,
    );
    this.handDetectorGraphOptions.setMinDetectionConfidence(DEFAULT_CONFIDENCE);
    this.handLandmarkerGraphOptions.setMinTrackingConfidence(
      DEFAULT_CONFIDENCE,
    );
    this.handLandmarksDetectorGraphOptions.setMinDetectionConfidence(
      DEFAULT_CONFIDENCE,
    );
  }

  protected override get baseOptions(): BaseOptionsProto {
    return this.options.getBaseOptions()!;
  }

  protected override set baseOptions(proto: BaseOptionsProto) {
    this.options.setBaseOptions(proto);
  }

  /**
   * Sets new options for the gesture recognizer.
   *
   * Calling `setOptions()` with a subset of options only affects those options.
   * You can reset an option back to its default value by explicitly setting it
   * to `undefined`.
   *
   * @export
   * @param options The options for the gesture recognizer.
   */
  override setOptions(options: GestureRecognizerOptions): Promise<void> {
    this.handDetectorGraphOptions.setNumHands(
      options.numHands ?? DEFAULT_NUM_HANDS,
    );
    if ('minHandDetectionConfidence' in options) {
      this.handDetectorGraphOptions.setMinDetectionConfidence(
        options.minHandDetectionConfidence ?? DEFAULT_CONFIDENCE,
      );
    }

    if ('minTrackingConfidence' in options) {
      this.handLandmarkerGraphOptions.setMinTrackingConfidence(
        options.minTrackingConfidence ?? DEFAULT_CONFIDENCE,
      );
    }

    if ('minHandPresenceConfidence' in options) {
      this.handLandmarksDetectorGraphOptions.setMinDetectionConfidence(
        options.minHandPresenceConfidence ?? DEFAULT_CONFIDENCE,
      );
    }

    if (options.cannedGesturesClassifierOptions) {
      // Note that we have to support both JSPB and ProtobufJS and cannot
      // use JSPB's getMutableX() APIs.
      const graphOptions = new GestureClassifierGraphOptions();
      graphOptions.setClassifierOptions(
        convertClassifierOptionsToProto(
          options.cannedGesturesClassifierOptions,
          this.handGestureRecognizerGraphOptions
            .getCannedGestureClassifierGraphOptions()
            ?.getClassifierOptions(),
        ),
      );
      this.handGestureRecognizerGraphOptions.setCannedGestureClassifierGraphOptions(
        graphOptions,
      );
    } else if (options.cannedGesturesClassifierOptions === undefined) {
      this.handGestureRecognizerGraphOptions
        .getCannedGestureClassifierGraphOptions()
        ?.clearClassifierOptions();
    }

    if (options.customGesturesClassifierOptions) {
      const graphOptions = new GestureClassifierGraphOptions();
      graphOptions.setClassifierOptions(
        convertClassifierOptionsToProto(
          options.customGesturesClassifierOptions,
          this.handGestureRecognizerGraphOptions
            .getCustomGestureClassifierGraphOptions()
            ?.getClassifierOptions(),
        ),
      );
      this.handGestureRecognizerGraphOptions.setCustomGestureClassifierGraphOptions(
        graphOptions,
      );
    } else if (options.customGesturesClassifierOptions === undefined) {
      this.handGestureRecognizerGraphOptions
        .getCustomGestureClassifierGraphOptions()
        ?.clearClassifierOptions();
    }

    return this.applyOptions(options);
  }

  /**
   * Performs gesture recognition on the provided single image and waits
   * synchronously for the response. Only use this method when the
   * GestureRecognizer is created with running mode `image`.
   *
   * @export
   * @param image A single image to process.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @return The detected gestures.
   */
  recognize(
    image: ImageSource,
    imageProcessingOptions?: ImageProcessingOptions,
  ): GestureRecognizerResult {
    this.resetResults();
    this.processImageData(image, imageProcessingOptions);
    return this.processResults();
  }

  /**
   * Performs gesture recognition on the provided video frame and waits
   * synchronously for the response. Only use this method when the
   * GestureRecognizer is created with running mode `video`.
   *
   * @export
   * @param videoFrame A video frame to process.
   * @param timestamp The timestamp of the current frame, in ms.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @return The detected gestures.
   */
  recognizeForVideo(
    videoFrame: ImageSource,
    timestamp: number,
    imageProcessingOptions?: ImageProcessingOptions,
  ): GestureRecognizerResult {
    this.resetResults();
    this.processVideoData(videoFrame, imageProcessingOptions, timestamp);
    return this.processResults();
  }

  private resetResults(): void {
    this.gestures = [];
    this.landmarks = [];
    this.worldLandmarks = [];
    this.handedness = [];
  }

  private processResults(): GestureRecognizerResult {
    if (this.gestures.length === 0) {
      // If no gestures are detected in the image, just return an empty list
      return {
        gestures: [],
        landmarks: [],
        worldLandmarks: [],
        handedness: [],
        handednesses: [],
      };
    } else {
      return {
        gestures: this.gestures,
        landmarks: this.landmarks,
        worldLandmarks: this.worldLandmarks,
        handedness: this.handedness,
        handednesses: this.handedness,
      };
    }
  }

  /** Converts the proto data to a Category[][] structure. */
  private toJsCategories(
    data: Uint8Array[],
    populateIndex = true,
  ): Category[][] {
    const result: Category[][] = [];
    for (const binaryProto of data) {
      const inputList = ClassificationList.deserializeBinary(binaryProto);
      const outputList: Category[] = [];
      for (const classification of inputList.getClassificationList()) {
        const index =
          populateIndex && classification.hasIndex()
            ? classification.getIndex()! :
              DEFAULT_CATEGORY_INDEX;
        outputList.push({
          score: classification.getScore() ?? 0,
          index,
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
      const landmarks: NormalizedLandmark[] = [];
      for (const handLandmarkProto of handLandmarksProto.getLandmarkList()) {
        landmarks.push({
          x: handLandmarkProto.getX() ?? 0,
          y: handLandmarkProto.getY() ?? 0,
          z: handLandmarkProto.getZ() ?? 0,
          visibility: handLandmarkProto.getVisibility() ?? 0,
        });
      }
      this.landmarks.push(landmarks);
    }
  }

  /**
   * Converts raw data into a landmark, and adds it to our worldLandmarks
   * list.
   */
  private addJsWorldLandmarks(data: Uint8Array[]): void {
    for (const binaryProto of data) {
      const handWorldLandmarksProto =
        LandmarkList.deserializeBinary(binaryProto);
      const worldLandmarks: Landmark[] = [];
      for (const handWorldLandmarkProto of handWorldLandmarksProto.getLandmarkList()) {
        worldLandmarks.push({
          x: handWorldLandmarkProto.getX() ?? 0,
          y: handWorldLandmarkProto.getY() ?? 0,
          z: handWorldLandmarkProto.getZ() ?? 0,
          visibility: handWorldLandmarkProto.getVisibility() ?? 0,
        });
      }
      this.worldLandmarks.push(worldLandmarks);
    }
  }

  /** Updates the MediaPipe graph configuration. */
  protected override refreshGraph(): void {
    const graphConfig = new CalculatorGraphConfig();
    graphConfig.addInputStream(IMAGE_STREAM);
    graphConfig.addInputStream(NORM_RECT_STREAM);
    graphConfig.addOutputStream(HAND_GESTURES_STREAM);
    graphConfig.addOutputStream(LANDMARKS_STREAM);
    graphConfig.addOutputStream(WORLD_LANDMARKS_STREAM);
    graphConfig.addOutputStream(HANDEDNESS_STREAM);

    const calculatorOptions = new CalculatorOptions();
    calculatorOptions.setExtension(
      GestureRecognizerGraphOptions.ext,
      this.options,
    );

    const recognizerNode = new CalculatorGraphConfig.Node();
    recognizerNode.setCalculator(GESTURE_RECOGNIZER_GRAPH);
    recognizerNode.addInputStream('IMAGE:' + IMAGE_STREAM);
    recognizerNode.addInputStream('NORM_RECT:' + NORM_RECT_STREAM);
    recognizerNode.addOutputStream('HAND_GESTURES:' + HAND_GESTURES_STREAM);
    recognizerNode.addOutputStream('LANDMARKS:' + LANDMARKS_STREAM);
    recognizerNode.addOutputStream('WORLD_LANDMARKS:' + WORLD_LANDMARKS_STREAM);
    recognizerNode.addOutputStream('HANDEDNESS:' + HANDEDNESS_STREAM);
    recognizerNode.setOptions(calculatorOptions);

    graphConfig.addNode(recognizerNode);

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
      HAND_GESTURES_STREAM,
      (binaryProto, timestamp) => {
        // Gesture index is not used, because the final gesture result comes
        // from multiple classifiers.
        this.gestures.push(
          ...this.toJsCategories(binaryProto, /* populateIndex= */ false),
        );
        this.setLatestOutputTimestamp(timestamp);
      },
    );
    this.graphRunner.attachEmptyPacketListener(
      HAND_GESTURES_STREAM,
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


