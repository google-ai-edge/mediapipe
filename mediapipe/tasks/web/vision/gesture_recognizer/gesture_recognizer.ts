/**
 * Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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
import {LandmarkList, NormalizedLandmarkList} from '../../../../framework/formats/landmark_pb';
import {NormalizedRect} from '../../../../framework/formats/rect_pb';
import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {GestureClassifierGraphOptions} from '../../../../tasks/cc/vision/gesture_recognizer/proto/gesture_classifier_graph_options_pb';
import {GestureRecognizerGraphOptions} from '../../../../tasks/cc/vision/gesture_recognizer/proto/gesture_recognizer_graph_options_pb';
import {HandGestureRecognizerGraphOptions} from '../../../../tasks/cc/vision/gesture_recognizer/proto/hand_gesture_recognizer_graph_options_pb';
import {HandDetectorGraphOptions} from '../../../../tasks/cc/vision/hand_detector/proto/hand_detector_graph_options_pb';
import {HandLandmarkerGraphOptions} from '../../../../tasks/cc/vision/hand_landmarker/proto/hand_landmarker_graph_options_pb';
import {HandLandmarksDetectorGraphOptions} from '../../../../tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options_pb';
import {Category} from '../../../../tasks/web/components/containers/category';
import {Landmark} from '../../../../tasks/web/components/containers/landmark';
import {convertClassifierOptionsToProto} from '../../../../tasks/web/components/processors/classifier_options';
import {WasmLoaderOptions} from '../../../../tasks/web/core/wasm_loader_options';
import {VisionTaskRunner} from '../../../../tasks/web/vision/core/vision_task_runner';
import {createMediaPipeLib, FileLocator, ImageSource, WasmModule} from '../../../../web/graph_runner/graph_runner';
// Placeholder for internal dependency on trusted resource url

import {GestureRecognizerOptions} from './gesture_recognizer_options';
import {GestureRecognizerResult} from './gesture_recognizer_result';

export * from './gesture_recognizer_options';
export * from './gesture_recognizer_result';
export {ImageSource};

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
const DEFAULT_SCORE_THRESHOLD = 0.5;
const DEFAULT_CATEGORY_INDEX = -1;

const FULL_IMAGE_RECT = new NormalizedRect();
FULL_IMAGE_RECT.setXCenter(0.5);
FULL_IMAGE_RECT.setYCenter(0.5);
FULL_IMAGE_RECT.setWidth(1);
FULL_IMAGE_RECT.setHeight(1);

/** Performs hand gesture recognition on images. */
export class GestureRecognizer extends
    VisionTaskRunner<GestureRecognizerResult> {
  private gestures: Category[][] = [];
  private landmarks: Landmark[][] = [];
  private worldLandmarks: Landmark[][] = [];
  private handednesses: Category[][] = [];

  private readonly options: GestureRecognizerGraphOptions;
  private readonly handLandmarkerGraphOptions: HandLandmarkerGraphOptions;
  private readonly handLandmarksDetectorGraphOptions:
      HandLandmarksDetectorGraphOptions;
  private readonly handDetectorGraphOptions: HandDetectorGraphOptions;
  private readonly handGestureRecognizerGraphOptions:
      HandGestureRecognizerGraphOptions;

  /**
   * Initializes the Wasm runtime and creates a new gesture recognizer from the
   * provided options.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param gestureRecognizerOptions The options for the gesture recognizer.
   *     Note that either a path to the model asset or a model buffer needs to
   *     be provided (via `baseOptions`).
   */
  static async createFromOptions(
      wasmLoaderOptions: WasmLoaderOptions,
      gestureRecognizerOptions: GestureRecognizerOptions):
      Promise<GestureRecognizer> {
    // Create a file locator based on the loader options
    const fileLocator: FileLocator = {
      locateFile() {
        // The only file we load via this mechanism is the Wasm binary
        return wasmLoaderOptions.wasmBinaryPath.toString();
      }
    };

    const recognizer = await createMediaPipeLib(
        GestureRecognizer, wasmLoaderOptions.wasmLoaderPath,
        /* assetLoaderScript= */ undefined,
        /* glCanvas= */ undefined, fileLocator);
    await recognizer.setOptions(gestureRecognizerOptions);
    return recognizer;
  }

  /**
   * Initializes the Wasm runtime and creates a new gesture recognizer based on
   * the provided model asset buffer.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param modelAssetBuffer A binary representation of the model.
   */
  static createFromModelBuffer(
      wasmLoaderOptions: WasmLoaderOptions,
      modelAssetBuffer: Uint8Array): Promise<GestureRecognizer> {
    return GestureRecognizer.createFromOptions(
        wasmLoaderOptions, {baseOptions: {modelAssetBuffer}});
  }

  /**
   * Initializes the Wasm runtime and creates a new gesture recognizer based on
   * the path to the model asset.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   */
  static async createFromModelPath(
      wasmLoaderOptions: WasmLoaderOptions,
      modelAssetPath: string): Promise<GestureRecognizer> {
    const response = await fetch(modelAssetPath.toString());
    const graphData = await response.arrayBuffer();
    return GestureRecognizer.createFromModelBuffer(
        wasmLoaderOptions, new Uint8Array(graphData));
  }

  constructor(wasmModule: WasmModule) {
    super(wasmModule);

    this.options = new GestureRecognizerGraphOptions();
    this.handLandmarkerGraphOptions = new HandLandmarkerGraphOptions();
    this.options.setHandLandmarkerGraphOptions(this.handLandmarkerGraphOptions);
    this.handLandmarksDetectorGraphOptions =
        new HandLandmarksDetectorGraphOptions();
    this.handLandmarkerGraphOptions.setHandLandmarksDetectorGraphOptions(
        this.handLandmarksDetectorGraphOptions);
    this.handDetectorGraphOptions = new HandDetectorGraphOptions();
    this.handLandmarkerGraphOptions.setHandDetectorGraphOptions(
        this.handDetectorGraphOptions);
    this.handGestureRecognizerGraphOptions =
        new HandGestureRecognizerGraphOptions();
    this.options.setHandGestureRecognizerGraphOptions(
        this.handGestureRecognizerGraphOptions);

    this.initDefaults();
  }

  protected override get baseOptions(): BaseOptionsProto|undefined {
    return this.options.getBaseOptions();
  }

  protected override set baseOptions(proto: BaseOptionsProto|undefined) {
    this.options.setBaseOptions(proto);
  }

  /**
   * Sets new options for the gesture recognizer.
   *
   * Calling `setOptions()` with a subset of options only affects those options.
   * You can reset an option back to its default value by explicitly setting it
   * to `undefined`.
   *
   * @param options The options for the gesture recognizer.
   */
  override async setOptions(options: GestureRecognizerOptions): Promise<void> {
    await super.setOptions(options);

    if ('numHands' in options) {
      this.handDetectorGraphOptions.setNumHands(
          options.numHands ?? DEFAULT_NUM_HANDS);
    }
    if ('minHandDetectionConfidence' in options) {
      this.handDetectorGraphOptions.setMinDetectionConfidence(
          options.minHandDetectionConfidence ?? DEFAULT_SCORE_THRESHOLD);
    }
    if ('minHandPresenceConfidence' in options) {
      this.handLandmarksDetectorGraphOptions.setMinDetectionConfidence(
          options.minHandPresenceConfidence ?? DEFAULT_SCORE_THRESHOLD);
    }
    if ('minTrackingConfidence' in options) {
      this.handLandmarkerGraphOptions.setMinTrackingConfidence(
          options.minTrackingConfidence ?? DEFAULT_SCORE_THRESHOLD);
    }

    if (options.cannedGesturesClassifierOptions) {
      // Note that we have to support both JSPB and ProtobufJS and cannot
      // use JSPB's getMutableX() APIs.
      const graphOptions = new GestureClassifierGraphOptions();
      graphOptions.setClassifierOptions(convertClassifierOptionsToProto(
          options.cannedGesturesClassifierOptions,
          this.handGestureRecognizerGraphOptions
              .getCannedGestureClassifierGraphOptions()
              ?.getClassifierOptions()));
      this.handGestureRecognizerGraphOptions
          .setCannedGestureClassifierGraphOptions(graphOptions);
    } else if (options.cannedGesturesClassifierOptions === undefined) {
      this.handGestureRecognizerGraphOptions
          .getCannedGestureClassifierGraphOptions()
          ?.clearClassifierOptions();
    }

    if (options.customGesturesClassifierOptions) {
      const graphOptions = new GestureClassifierGraphOptions();
      graphOptions.setClassifierOptions(convertClassifierOptionsToProto(
          options.customGesturesClassifierOptions,
          this.handGestureRecognizerGraphOptions
              .getCustomGestureClassifierGraphOptions()
              ?.getClassifierOptions()));
      this.handGestureRecognizerGraphOptions
          .setCustomGestureClassifierGraphOptions(graphOptions);
    } else if (options.customGesturesClassifierOptions === undefined) {
      this.handGestureRecognizerGraphOptions
          .getCustomGestureClassifierGraphOptions()
          ?.clearClassifierOptions();
    }

    this.refreshGraph();
  }

  /**
   * Performs gesture recognition on the provided single image and waits
   * synchronously for the response.
   * @param image A single image to process.
   * @return The detected gestures.
   */
  recognize(image: ImageSource): GestureRecognizerResult {
    return this.processImageData(image);
  }

  /**
   * Performs gesture recognition on the provided video frame and waits
   * synchronously for the response.
   * @param videoFrame A video frame to process.
   * @param timestamp The timestamp of the current frame, in ms.
   * @return The detected gestures.
   */
  recognizeForVideo(videoFrame: ImageSource, timestamp: number):
      GestureRecognizerResult {
    return this.processVideoData(videoFrame, timestamp);
  }

  /** Runs the gesture recognition and blocks on the response. */
  protected override process(imageSource: ImageSource, timestamp: number):
      GestureRecognizerResult {
    this.gestures = [];
    this.landmarks = [];
    this.worldLandmarks = [];
    this.handednesses = [];

    this.addGpuBufferAsImageToStream(imageSource, IMAGE_STREAM, timestamp);
    this.addProtoToStream(
        FULL_IMAGE_RECT.serializeBinary(), 'mediapipe.NormalizedRect',
        NORM_RECT_STREAM, timestamp);
    this.finishProcessing();

    return {
      gestures: this.gestures,
      landmarks: this.landmarks,
      worldLandmarks: this.worldLandmarks,
      handednesses: this.handednesses
    };
  }

  /** Sets the default values for the graph. */
  private initDefaults(): void {
    this.handDetectorGraphOptions.setNumHands(DEFAULT_NUM_HANDS);
    this.handDetectorGraphOptions.setMinDetectionConfidence(
        DEFAULT_SCORE_THRESHOLD);
    this.handLandmarksDetectorGraphOptions.setMinDetectionConfidence(
        DEFAULT_SCORE_THRESHOLD);
    this.handLandmarkerGraphOptions.setMinTrackingConfidence(
        DEFAULT_SCORE_THRESHOLD);
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
      const landmarks: Landmark[] = [];
      for (const handLandmarkProto of handLandmarksProto.getLandmarkList()) {
        landmarks.push({
          x: handLandmarkProto.getX() ?? 0,
          y: handLandmarkProto.getY() ?? 0,
          z: handLandmarkProto.getZ() ?? 0,
          normalized: true
        });
      }
      this.landmarks.push(landmarks);
    }
  }

  /**
   * Converts raw data into a landmark, and adds it to our worldLandmarks
   * list.
   */
  private adddJsWorldLandmarks(data: Uint8Array[]): void {
    for (const binaryProto of data) {
      const handWorldLandmarksProto =
          LandmarkList.deserializeBinary(binaryProto);
      const worldLandmarks: Landmark[] = [];
      for (const handWorldLandmarkProto of
               handWorldLandmarksProto.getLandmarkList()) {
        worldLandmarks.push({
          x: handWorldLandmarkProto.getX() ?? 0,
          y: handWorldLandmarkProto.getY() ?? 0,
          z: handWorldLandmarkProto.getZ() ?? 0,
          normalized: false
        });
      }
      this.worldLandmarks.push(worldLandmarks);
    }
  }

  /** Updates the MediaPipe graph configuration. */
  private refreshGraph(): void {
    const graphConfig = new CalculatorGraphConfig();
    graphConfig.addInputStream(IMAGE_STREAM);
    graphConfig.addInputStream(NORM_RECT_STREAM);
    graphConfig.addOutputStream(HAND_GESTURES_STREAM);
    graphConfig.addOutputStream(LANDMARKS_STREAM);
    graphConfig.addOutputStream(WORLD_LANDMARKS_STREAM);
    graphConfig.addOutputStream(HANDEDNESS_STREAM);

    const calculatorOptions = new CalculatorOptions();
    calculatorOptions.setExtension(
        GestureRecognizerGraphOptions.ext, this.options);

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

    this.attachProtoVectorListener(LANDMARKS_STREAM, binaryProto => {
      this.addJsLandmarks(binaryProto);
    });
    this.attachProtoVectorListener(WORLD_LANDMARKS_STREAM, binaryProto => {
      this.adddJsWorldLandmarks(binaryProto);
    });
    this.attachProtoVectorListener(HAND_GESTURES_STREAM, binaryProto => {
      this.gestures.push(...this.toJsCategories(binaryProto));
    });
    this.attachProtoVectorListener(HANDEDNESS_STREAM, binaryProto => {
      this.handednesses.push(...this.toJsCategories(binaryProto));
    });

    const binaryGraph = graphConfig.serializeBinary();
    this.setGraph(new Uint8Array(binaryGraph), /* isBinary= */ true);
  }
}


