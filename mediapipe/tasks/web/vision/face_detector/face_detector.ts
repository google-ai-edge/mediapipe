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
import {FaceDetectorGraphOptions as FaceDetectorGraphOptionsProto} from '../../../../tasks/cc/vision/face_detector/proto/face_detector_graph_options_pb';
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

import {FaceDetectorOptions} from './face_detector_options';
import {FaceDetectorResult} from './face_detector_result';

const IMAGE_STREAM = 'image_in';
const NORM_RECT_STREAM = 'norm_rect_in';
const DETECTIONS_STREAM = 'detections';
const FACE_DETECTOR_GRAPH =
  'mediapipe.tasks.vision.face_detector.FaceDetectorGraph';

export * from './face_detector_options';
export * from './face_detector_result';
export {type ImageSource}; // Used in the public API

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

/** Performs face detection on images. */
export class FaceDetector extends VisionTaskRunner {
  private result: FaceDetectorResult = {detections: []};
  private readonly options = new FaceDetectorGraphOptionsProto();

  /**
   * Initializes the Wasm runtime and creates a new face detector from the
   * provided options.
   *
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param faceDetectorOptions The options for the FaceDetector. Note that
   *     either a path to the model asset or a model buffer needs to be
   *     provided (via `baseOptions`).
   */
  static createFromOptions(
    wasmFileset: WasmFileset,
    faceDetectorOptions: FaceDetectorOptions,
  ): Promise<FaceDetector> {
    return VisionTaskRunner.createVisionInstance(
      FaceDetector,
      wasmFileset,
      faceDetectorOptions,
    );
  }

  /**
   * Initializes the Wasm runtime and creates a new face detector based on the
   * provided model asset buffer.
   *
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param modelAssetBuffer An array or a stream containing a binary
   *    representation of the model.
   */
  static createFromModelBuffer(
    wasmFileset: WasmFileset,
    modelAssetBuffer: Uint8Array | ReadableStreamDefaultReader,
  ): Promise<FaceDetector> {
    return VisionTaskRunner.createVisionInstance(FaceDetector, wasmFileset, {
      baseOptions: {modelAssetBuffer},
    });
  }

  /**
   * Initializes the Wasm runtime and creates a new face detector based on the
   * path to the model asset.
   *
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   */
  static async createFromModelPath(
    wasmFileset: WasmFileset,
    modelAssetPath: string,
  ): Promise<FaceDetector> {
    return VisionTaskRunner.createVisionInstance(FaceDetector, wasmFileset, {
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
    this.options.setMinDetectionConfidence(0.5);
    this.options.setMinSuppressionThreshold(0.3);
  }

  protected override get baseOptions(): BaseOptionsProto {
    return this.options.getBaseOptions()!;
  }

  protected override set baseOptions(proto: BaseOptionsProto) {
    this.options.setBaseOptions(proto);
  }

  /**
   * Sets new options for the FaceDetector.
   *
   * Calling `setOptions()` with a subset of options only affects those options.
   * You can reset an option back to its default value by explicitly setting it
   * to `undefined`.
   *
   * @export
   * @param options The options for the FaceDetector.
   */
  override setOptions(options: FaceDetectorOptions): Promise<void> {
    if ('minDetectionConfidence' in options) {
      this.options.setMinDetectionConfidence(
        options.minDetectionConfidence ?? 0.5,
      );
    }
    if ('minSuppressionThreshold' in options) {
      this.options.setMinSuppressionThreshold(
        options.minSuppressionThreshold ?? 0.3,
      );
    }
    return this.applyOptions(options);
  }

  /**
   * Performs face detection on the provided single image and waits
   * synchronously for the response. Only use this method when the
   * FaceDetector is created with running mode `image`.
   *
   * @export
   * @param image An image to process.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @return A result containing the list of detected faces.
   */
  detect(
    image: ImageSource,
    imageProcessingOptions?: ImageProcessingOptions,
  ): FaceDetectorResult {
    this.result = {detections: []};
    this.processImageData(image, imageProcessingOptions);
    return this.result;
  }

  /**
   * Performs face detection on the provided video frame and waits
   * synchronously for the response. Only use this method when the
   * FaceDetector is created with running mode `video`.
   *
   * @export
   * @param videoFrame A video frame to process.
   * @param timestamp The timestamp of the current frame, in ms.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @return A result containing the list of detected faces.
   */
  detectForVideo(
    videoFrame: ImageSource,
    timestamp: number,
    imageProcessingOptions?: ImageProcessingOptions,
  ): FaceDetectorResult {
    this.result = {detections: []};
    this.processVideoData(videoFrame, imageProcessingOptions, timestamp);
    return this.result;
  }

  /** Converts raw data into a Detection, and adds it to our detection list. */
  private addJsFaceDetections(data: Uint8Array[]): void {
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
      FaceDetectorGraphOptionsProto.ext,
      this.options,
    );

    const detectorNode = new CalculatorGraphConfig.Node();
    detectorNode.setCalculator(FACE_DETECTOR_GRAPH);
    detectorNode.addInputStream('IMAGE:' + IMAGE_STREAM);
    detectorNode.addInputStream('NORM_RECT:' + NORM_RECT_STREAM);
    detectorNode.addOutputStream('DETECTIONS:' + DETECTIONS_STREAM);
    detectorNode.setOptions(calculatorOptions);

    graphConfig.addNode(detectorNode);

    this.graphRunner.attachProtoVectorListener(
      DETECTIONS_STREAM,
      (binaryProto, timestamp) => {
        this.addJsFaceDetections(binaryProto);
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


