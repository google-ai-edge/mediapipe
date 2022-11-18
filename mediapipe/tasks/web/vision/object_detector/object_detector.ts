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
import {Detection as DetectionProto} from '../../../../framework/formats/detection_pb';
import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {ObjectDetectorOptions as ObjectDetectorOptionsProto} from '../../../../tasks/cc/vision/object_detector/proto/object_detector_options_pb';
import {WasmLoaderOptions} from '../../../../tasks/web/core/wasm_loader_options';
import {VisionTaskRunner} from '../../../../tasks/web/vision/core/vision_task_runner';
import {createMediaPipeLib, FileLocator, ImageSource} from '../../../../web/graph_runner/graph_runner';
// Placeholder for internal dependency on trusted resource url

import {ObjectDetectorOptions} from './object_detector_options';
import {Detection} from './object_detector_result';

const INPUT_STREAM = 'input_frame_gpu';
const DETECTIONS_STREAM = 'detections';
const OBJECT_DETECTOR_GRAPH = 'mediapipe.tasks.vision.ObjectDetectorGraph';

const DEFAULT_CATEGORY_INDEX = -1;

export * from './object_detector_options';
export * from './object_detector_result';
export {ImageSource};  // Used in the public API

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

/** Performs object detection on images. */
export class ObjectDetector extends VisionTaskRunner<Detection[]> {
  private detections: Detection[] = [];
  private readonly options = new ObjectDetectorOptionsProto();

  /**
   * Initializes the Wasm runtime and creates a new object detector from the
   * provided options.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param objectDetectorOptions The options for the Object Detector. Note that
   *     either a path to the model asset or a model buffer needs to be
   *     provided (via `baseOptions`).
   */
  static async createFromOptions(
      wasmLoaderOptions: WasmLoaderOptions,
      objectDetectorOptions: ObjectDetectorOptions): Promise<ObjectDetector> {
    // Create a file locator based on the loader options
    const fileLocator: FileLocator = {
      locateFile() {
        // The only file we load is the Wasm binary
        return wasmLoaderOptions.wasmBinaryPath.toString();
      }
    };

    const detector = await createMediaPipeLib(
        ObjectDetector, wasmLoaderOptions.wasmLoaderPath,
        /* assetLoaderScript= */ undefined,
        /* glCanvas= */ undefined, fileLocator);
    await detector.setOptions(objectDetectorOptions);
    return detector;
  }

  /**
   * Initializes the Wasm runtime and creates a new object detector based on the
   * provided model asset buffer.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param modelAssetBuffer A binary representation of the model.
   */
  static createFromModelBuffer(
      wasmLoaderOptions: WasmLoaderOptions,
      modelAssetBuffer: Uint8Array): Promise<ObjectDetector> {
    return ObjectDetector.createFromOptions(
        wasmLoaderOptions, {baseOptions: {modelAssetBuffer}});
  }

  /**
   * Initializes the Wasm runtime and creates a new object detector based on the
   * path to the model asset.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   */
  static async createFromModelPath(
      wasmLoaderOptions: WasmLoaderOptions,
      modelAssetPath: string): Promise<ObjectDetector> {
    const response = await fetch(modelAssetPath.toString());
    const graphData = await response.arrayBuffer();
    return ObjectDetector.createFromModelBuffer(
        wasmLoaderOptions, new Uint8Array(graphData));
  }

  protected override get baseOptions(): BaseOptionsProto|undefined {
    return this.options.getBaseOptions();
  }

  protected override set baseOptions(proto: BaseOptionsProto|undefined) {
    this.options.setBaseOptions(proto);
  }

  /**
   * Sets new options for the object detector.
   *
   * Calling `setOptions()` with a subset of options only affects those options.
   * You can reset an option back to its default value by explicitly setting it
   * to `undefined`.
   *
   * @param options The options for the object detector.
   */
  override async setOptions(options: ObjectDetectorOptions): Promise<void> {
    await super.setOptions(options);

    // Note that we have to support both JSPB and ProtobufJS, hence we
    // have to expliclity clear the values instead of setting them to
    // `undefined`.
    if (options.displayNamesLocale !== undefined) {
      this.options.setDisplayNamesLocale(options.displayNamesLocale);
    } else if ('displayNamesLocale' in options) {  // Check for undefined
      this.options.clearDisplayNamesLocale();
    }

    if (options.maxResults !== undefined) {
      this.options.setMaxResults(options.maxResults);
    } else if ('maxResults' in options) {  // Check for undefined
      this.options.clearMaxResults();
    }

    if (options.scoreThreshold !== undefined) {
      this.options.setScoreThreshold(options.scoreThreshold);
    } else if ('scoreThreshold' in options) {  // Check for undefined
      this.options.clearScoreThreshold();
    }

    if (options.categoryAllowlist !== undefined) {
      this.options.setCategoryAllowlistList(options.categoryAllowlist);
    } else if ('categoryAllowlist' in options) {  // Check for undefined
      this.options.clearCategoryAllowlistList();
    }

    if (options.categoryDenylist !== undefined) {
      this.options.setCategoryDenylistList(options.categoryDenylist);
    } else if ('categoryDenylist' in options) {  // Check for undefined
      this.options.clearCategoryDenylistList();
    }

    this.refreshGraph();
  }

  /**
   * Performs object detection on the provided single image and waits
   * synchronously for the response.
   * @param image An image to process.
   * @return The list of detected objects
   */
  detect(image: ImageSource): Detection[] {
    return this.processImageData(image);
  }

  /**
   * Performs object detection on the provided vidoe frame and waits
   * synchronously for the response.
   * @param videoFrame A video frame to process.
   * @param timestamp The timestamp of the current frame, in ms.
   * @return The list of detected objects
   */
  detectForVideo(videoFrame: ImageSource, timestamp: number): Detection[] {
    return this.processVideoData(videoFrame, timestamp);
  }

  /** Runs the object detector graph and blocks on the response. */
  protected override process(imageSource: ImageSource, timestamp: number):
      Detection[] {
    // Get detections by running our MediaPipe graph.
    this.detections = [];
    this.addGpuBufferAsImageToStream(
        imageSource, INPUT_STREAM, timestamp ?? performance.now());
    this.finishProcessing();
    return [...this.detections];
  }

  /** Converts raw data into a Detection, and adds it to our detection list. */
  private addJsObjectDetections(data: Uint8Array[]): void {
    for (const binaryProto of data) {
      const detectionProto = DetectionProto.deserializeBinary(binaryProto);
      const scores = detectionProto.getScoreList();
      const indexes = detectionProto.getLabelIdList();
      const labels = detectionProto.getLabelList();
      const displayNames = detectionProto.getDisplayNameList();

      const detection: Detection = {categories: []};
      for (let i = 0; i < scores.length; i++) {
        detection.categories.push({
          score: scores[i],
          index: indexes[i] ?? DEFAULT_CATEGORY_INDEX,
          categoryName: labels[i] ?? '',
          displayName: displayNames[i] ?? '',
        });
      }

      const boundingBox = detectionProto.getLocationData()?.getBoundingBox();
      if (boundingBox) {
        detection.boundingBox = {
          originX: boundingBox.getXmin() ?? 0,
          originY: boundingBox.getYmin() ?? 0,
          width: boundingBox.getWidth() ?? 0,
          height: boundingBox.getHeight() ?? 0
        };
      }

      this.detections.push(detection);
    }
  }

  /** Updates the MediaPipe graph configuration. */
  private refreshGraph(): void {
    const graphConfig = new CalculatorGraphConfig();
    graphConfig.addInputStream(INPUT_STREAM);
    graphConfig.addOutputStream(DETECTIONS_STREAM);

    const calculatorOptions = new CalculatorOptions();
    calculatorOptions.setExtension(
        ObjectDetectorOptionsProto.ext, this.options);

    const detectorNode = new CalculatorGraphConfig.Node();
    detectorNode.setCalculator(OBJECT_DETECTOR_GRAPH);
    detectorNode.addInputStream('IMAGE:' + INPUT_STREAM);
    detectorNode.addOutputStream('DETECTIONS:' + DETECTIONS_STREAM);
    detectorNode.setOptions(calculatorOptions);

    graphConfig.addNode(detectorNode);

    this.attachProtoVectorListener(DETECTIONS_STREAM, binaryProto => {
      this.addJsObjectDetections(binaryProto);
    });

    const binaryGraph = graphConfig.serializeBinary();
    this.setGraph(new Uint8Array(binaryGraph), /* isBinary= */ true);
  }
}


