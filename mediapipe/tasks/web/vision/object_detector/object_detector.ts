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
import {Detection as DetectionProto} from '../../../../framework/formats/detection_pb';
import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {ObjectDetectorOptions as ObjectDetectorOptionsProto} from '../../../../tasks/cc/vision/object_detector/proto/object_detector_options_pb';
import {convertFromDetectionProto} from '../../../../tasks/web/components/processors/detection_result';
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

import {ObjectDetectorOptions} from './object_detector_options';
import {ObjectDetectorResult} from './object_detector_result';

const IMAGE_STREAM = 'input_frame_gpu';
const NORM_RECT_STREAM = 'norm_rect';
const DETECTIONS_STREAM = 'detections';
const OBJECT_DETECTOR_GRAPH = 'mediapipe.tasks.vision.ObjectDetectorGraph';

export * from './object_detector_options';
export * from './object_detector_result';
export {type ImageSource}; // Used in the public API

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

/**
 * Performs object detection on images.
 */
export class ObjectDetector extends VisionTaskRunner {
  private result: ObjectDetectorResult = {detections: []};
  private readonly options = new ObjectDetectorOptionsProto();

  /**
   * Initializes the Wasm runtime and creates a new object detector from the
   * provided options.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param objectDetectorOptions The options for the Object Detector. Note that
   *     either a path to the model asset or a model buffer needs to be
   *     provided (via `baseOptions`).
   */
  static createFromOptions(
    wasmFileset: WasmFileset,
    objectDetectorOptions: ObjectDetectorOptions,
  ): Promise<ObjectDetector> {
    return VisionTaskRunner.createVisionInstance(
      ObjectDetector,
      wasmFileset,
      objectDetectorOptions,
    );
  }

  /**
   * Initializes the Wasm runtime and creates a new object detector based on the
   * provided model asset buffer.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param modelAssetBuffer An array or a stream containing a binary
   *    representation of the model.
   */
  static createFromModelBuffer(
    wasmFileset: WasmFileset,
    modelAssetBuffer: Uint8Array | ReadableStreamDefaultReader,
  ): Promise<ObjectDetector> {
    return VisionTaskRunner.createVisionInstance(ObjectDetector, wasmFileset, {
      baseOptions: {modelAssetBuffer},
    });
  }

  /**
   * Initializes the Wasm runtime and creates a new object detector based on the
   * path to the model asset.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   */
  static async createFromModelPath(
    wasmFileset: WasmFileset,
    modelAssetPath: string,
  ): Promise<ObjectDetector> {
    return VisionTaskRunner.createVisionInstance(ObjectDetector, wasmFileset, {
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
    this.options.setBaseOptions(new BaseOptionsProto());
  }

  protected override get baseOptions(): BaseOptionsProto {
    return this.options.getBaseOptions()!;
  }

  protected override set baseOptions(proto: BaseOptionsProto) {
    this.options.setBaseOptions(proto);
  }

  /**
   * Sets new options for the object detector.
   *
   * Calling `setOptions()` with a subset of options only affects those options.
   * You can reset an option back to its default value by explicitly setting it
   * to `undefined`.
   *
   * @export
   * @param options The options for the object detector.
   */
  override setOptions(options: ObjectDetectorOptions): Promise<void> {
    // Note that we have to support both JSPB and ProtobufJS, hence we
    // have to expliclity clear the values instead of setting them to
    // `undefined`.
    if (options.displayNamesLocale !== undefined) {
      this.options.setDisplayNamesLocale(options.displayNamesLocale);
    } else if ('displayNamesLocale' in options) {
      // Check for undefined
      this.options.clearDisplayNamesLocale();
    }

    if (options.maxResults !== undefined) {
      this.options.setMaxResults(options.maxResults);
    } else if ('maxResults' in options) {
      // Check for undefined
      this.options.clearMaxResults();
    }

    if (options.scoreThreshold !== undefined) {
      this.options.setScoreThreshold(options.scoreThreshold);
    } else if ('scoreThreshold' in options) {
      // Check for undefined
      this.options.clearScoreThreshold();
    }

    if (options.categoryAllowlist !== undefined) {
      this.options.setCategoryAllowlistList(options.categoryAllowlist);
    } else if ('categoryAllowlist' in options) {
      // Check for undefined
      this.options.clearCategoryAllowlistList();
    }

    if (options.categoryDenylist !== undefined) {
      this.options.setCategoryDenylistList(options.categoryDenylist);
    } else if ('categoryDenylist' in options) {
      // Check for undefined
      this.options.clearCategoryDenylistList();
    }

    return this.applyOptions(options);
  }

  /**
   * Performs object detection on the provided single image and waits
   * synchronously for the response. Only use this method when the
   * ObjectDetector is created with running mode `image`.
   *
   * @export
   * @param image An image to process.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @return A result containing a list of detected objects.
   */
  detect(
    image: ImageSource,
    imageProcessingOptions?: ImageProcessingOptions,
  ): ObjectDetectorResult {
    this.result = {detections: []};
    this.processImageData(image, imageProcessingOptions);
    return this.result;
  }

  /**
   * Performs object detection on the provided video frame and waits
   * synchronously for the response. Only use this method when the
   * ObjectDetector is created with running mode `video`.
   *
   * @export
   * @param videoFrame A video frame to process.
   * @param timestamp The timestamp of the current frame, in ms.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @return A result containing a list of detected objects.
   */
  detectForVideo(
    videoFrame: ImageSource,
    timestamp: number,
    imageProcessingOptions?: ImageProcessingOptions,
  ): ObjectDetectorResult {
    this.result = {detections: []};
    this.processVideoData(videoFrame, imageProcessingOptions, timestamp);
    return this.result;
  }

  /** Converts raw data into a Detection, and adds it to our detection list. */
  private addJsObjectDetections(data: Uint8Array[]): void {
    for (const binaryProto of data) {
      const detectionProto = DetectionProto.deserializeBinary(binaryProto);
      this.result.detections.push(convertFromDetectionProto(detectionProto));
    }
  }

  /** Updates the MediaPipe graph configuration. */
  protected override refreshGraph(): void {
    const graphConfig = new CalculatorGraphConfig();
    graphConfig.addInputStream(IMAGE_STREAM);
    graphConfig.addInputStream(NORM_RECT_STREAM);
    graphConfig.addOutputStream(DETECTIONS_STREAM);

    const calculatorOptions = new CalculatorOptions();
    calculatorOptions.setExtension(
      ObjectDetectorOptionsProto.ext,
      this.options,
    );

    const detectorNode = new CalculatorGraphConfig.Node();
    detectorNode.setCalculator(OBJECT_DETECTOR_GRAPH);
    detectorNode.addInputStream('IMAGE:' + IMAGE_STREAM);
    detectorNode.addInputStream('NORM_RECT:' + NORM_RECT_STREAM);
    detectorNode.addOutputStream('DETECTIONS:' + DETECTIONS_STREAM);
    detectorNode.setOptions(calculatorOptions);

    graphConfig.addNode(detectorNode);

    this.graphRunner.attachProtoVectorListener(
      DETECTIONS_STREAM,
      (binaryProto, timestamp) => {
        this.addJsObjectDetections(binaryProto);
        this.setLatestOutputTimestamp(timestamp);
      },
    );
    this.graphRunner.attachEmptyPacketListener(
      DETECTIONS_STREAM,
      (timestamp) => {
        this.setLatestOutputTimestamp(timestamp);
      },
    );

    const binaryGraph = graphConfig.serializeBinary();
    this.setGraph(new Uint8Array(binaryGraph), /* isBinary= */ true);
  }
}


