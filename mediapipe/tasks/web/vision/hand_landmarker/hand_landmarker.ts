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
import {HandDetectorGraphOptions} from '../../../../tasks/cc/vision/hand_detector/proto/hand_detector_graph_options_pb';
import {HandLandmarkerGraphOptions} from '../../../../tasks/cc/vision/hand_landmarker/proto/hand_landmarker_graph_options_pb';
import {HandLandmarksDetectorGraphOptions} from '../../../../tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options_pb';
import {Category} from '../../../../tasks/web/components/containers/category';
import {Landmark} from '../../../../tasks/web/components/containers/landmark';
import {WasmLoaderOptions} from '../../../../tasks/web/core/wasm_loader_options';
import {VisionTaskRunner} from '../../../../tasks/web/vision/core/vision_task_runner';
import {createMediaPipeLib, FileLocator, ImageSource, WasmModule} from '../../../../web/graph_runner/graph_runner';
// Placeholder for internal dependency on trusted resource url

import {HandLandmarkerOptions} from './hand_landmarker_options';
import {HandLandmarkerResult} from './hand_landmarker_result';

export * from './hand_landmarker_options';
export * from './hand_landmarker_result';
export {ImageSource};

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
const FULL_IMAGE_RECT = new NormalizedRect();
FULL_IMAGE_RECT.setXCenter(0.5);
FULL_IMAGE_RECT.setYCenter(0.5);
FULL_IMAGE_RECT.setWidth(1);
FULL_IMAGE_RECT.setHeight(1);

/** Performs hand landmarks detection on images. */
export class HandLandmarker extends VisionTaskRunner<HandLandmarkerResult> {
  private landmarks: Landmark[][] = [];
  private worldLandmarks: Landmark[][] = [];
  private handednesses: Category[][] = [];

  private readonly options: HandLandmarkerGraphOptions;
  private readonly handLandmarksDetectorGraphOptions:
      HandLandmarksDetectorGraphOptions;
  private readonly handDetectorGraphOptions: HandDetectorGraphOptions;

  /**
   * Initializes the Wasm runtime and creates a new `HandLandmarker` from the
   * provided options.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param handLandmarkerOptions The options for the HandLandmarker.
   *     Note that either a path to the model asset or a model buffer needs to
   *     be provided (via `baseOptions`).
   */
  static async createFromOptions(
      wasmLoaderOptions: WasmLoaderOptions,
      handLandmarkerOptions: HandLandmarkerOptions): Promise<HandLandmarker> {
    // Create a file locator based on the loader options
    const fileLocator: FileLocator = {
      locateFile() {
        // The only file we load via this mechanism is the Wasm binary
        return wasmLoaderOptions.wasmBinaryPath.toString();
      }
    };

    const landmarker = await createMediaPipeLib(
        HandLandmarker, wasmLoaderOptions.wasmLoaderPath,
        /* assetLoaderScript= */ undefined,
        /* glCanvas= */ undefined, fileLocator);
    await landmarker.setOptions(handLandmarkerOptions);
    return landmarker;
  }

  /**
   * Initializes the Wasm runtime and creates a new `HandLandmarker` based on
   * the provided model asset buffer.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param modelAssetBuffer A binary representation of the model.
   */
  static createFromModelBuffer(
      wasmLoaderOptions: WasmLoaderOptions,
      modelAssetBuffer: Uint8Array): Promise<HandLandmarker> {
    return HandLandmarker.createFromOptions(
        wasmLoaderOptions, {baseOptions: {modelAssetBuffer}});
  }

  /**
   * Initializes the Wasm runtime and creates a new `HandLandmarker` based on
   * the path to the model asset.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   */
  static async createFromModelPath(
      wasmLoaderOptions: WasmLoaderOptions,
      modelAssetPath: string): Promise<HandLandmarker> {
    const response = await fetch(modelAssetPath.toString());
    const graphData = await response.arrayBuffer();
    return HandLandmarker.createFromModelBuffer(
        wasmLoaderOptions, new Uint8Array(graphData));
  }

  constructor(wasmModule: WasmModule) {
    super(wasmModule);

    this.options = new HandLandmarkerGraphOptions();
    this.handLandmarksDetectorGraphOptions =
        new HandLandmarksDetectorGraphOptions();
    this.options.setHandLandmarksDetectorGraphOptions(
        this.handLandmarksDetectorGraphOptions);
    this.handDetectorGraphOptions = new HandDetectorGraphOptions();
    this.options.setHandDetectorGraphOptions(this.handDetectorGraphOptions);

    this.initDefaults();
  }

  protected override get baseOptions(): BaseOptionsProto|undefined {
    return this.options.getBaseOptions();
  }

  protected override set baseOptions(proto: BaseOptionsProto|undefined) {
    this.options.setBaseOptions(proto);
  }

  /**
   * Sets new options for this `HandLandmarker`.
   *
   * Calling `setOptions()` with a subset of options only affects those options.
   * You can reset an option back to its default value by explicitly setting it
   * to `undefined`.
   *
   * @param options The options for the hand landmarker.
   */
  override async setOptions(options: HandLandmarkerOptions): Promise<void> {
    await super.setOptions(options);

    // Configure hand detector options.
    if ('numHands' in options) {
      this.handDetectorGraphOptions.setNumHands(
          options.numHands ?? DEFAULT_NUM_HANDS);
    }
    if ('minHandDetectionConfidence' in options) {
      this.handDetectorGraphOptions.setMinDetectionConfidence(
          options.minHandDetectionConfidence ?? DEFAULT_SCORE_THRESHOLD);
    }

    // Configure hand landmark detector options.
    if ('minTrackingConfidence' in options) {
      this.options.setMinTrackingConfidence(
          options.minTrackingConfidence ?? DEFAULT_SCORE_THRESHOLD);
    }
    if ('minHandPresenceConfidence' in options) {
      this.handLandmarksDetectorGraphOptions.setMinDetectionConfidence(
          options.minHandPresenceConfidence ?? DEFAULT_SCORE_THRESHOLD);
    }

    this.refreshGraph();
  }

  /**
   * Performs hand landmarks detection on the provided single image and waits
   * synchronously for the response.
   * @param image An image to process.
   * @return The detected hand landmarks.
   */
  detect(image: ImageSource): HandLandmarkerResult {
    return this.processImageData(image);
  }

  /**
   * Performs hand landmarks detection on the provided video frame and waits
   * synchronously for the response.
   * @param videoFrame A video frame to process.
   * @param timestamp The timestamp of the current frame, in ms.
   * @return The detected hand landmarks.
   */
  detectForVideo(videoFrame: ImageSource, timestamp: number):
      HandLandmarkerResult {
    return this.processVideoData(videoFrame, timestamp);
  }

  /** Runs the hand landmarker graph and blocks on the response. */
  protected override process(imageSource: ImageSource, timestamp: number):
      HandLandmarkerResult {
    this.landmarks = [];
    this.worldLandmarks = [];
    this.handednesses = [];

    this.addGpuBufferAsImageToStream(imageSource, IMAGE_STREAM, timestamp);
    this.addProtoToStream(
        FULL_IMAGE_RECT.serializeBinary(), 'mediapipe.NormalizedRect',
        NORM_RECT_STREAM, timestamp);
    this.finishProcessing();

    return {
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
    graphConfig.addOutputStream(LANDMARKS_STREAM);
    graphConfig.addOutputStream(WORLD_LANDMARKS_STREAM);
    graphConfig.addOutputStream(HANDEDNESS_STREAM);

    const calculatorOptions = new CalculatorOptions();
    calculatorOptions.setExtension(
        HandLandmarkerGraphOptions.ext, this.options);

    const landmarkerNode = new CalculatorGraphConfig.Node();
    landmarkerNode.setCalculator(HAND_LANDMARKER_GRAPH);
    landmarkerNode.addInputStream('IMAGE:' + IMAGE_STREAM);
    landmarkerNode.addInputStream('NORM_RECT:' + NORM_RECT_STREAM);
    landmarkerNode.addOutputStream('LANDMARKS:' + LANDMARKS_STREAM);
    landmarkerNode.addOutputStream('WORLD_LANDMARKS:' + WORLD_LANDMARKS_STREAM);
    landmarkerNode.addOutputStream('HANDEDNESS:' + HANDEDNESS_STREAM);
    landmarkerNode.setOptions(calculatorOptions);

    graphConfig.addNode(landmarkerNode);

    this.attachProtoVectorListener(LANDMARKS_STREAM, binaryProto => {
      this.addJsLandmarks(binaryProto);
    });
    this.attachProtoVectorListener(WORLD_LANDMARKS_STREAM, binaryProto => {
      this.adddJsWorldLandmarks(binaryProto);
    });
    this.attachProtoVectorListener(HANDEDNESS_STREAM, binaryProto => {
      this.handednesses.push(...this.toJsCategories(binaryProto));
    });

    const binaryGraph = graphConfig.serializeBinary();
    this.setGraph(new Uint8Array(binaryGraph), /* isBinary= */ true);
  }
}


