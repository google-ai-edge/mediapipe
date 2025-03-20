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
import {ClassificationResult} from '../../../../tasks/cc/components/containers/proto/classifications_pb';
import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {ImageClassifierGraphOptions} from '../../../../tasks/cc/vision/image_classifier/proto/image_classifier_graph_options_pb';
import {convertClassifierOptionsToProto} from '../../../../tasks/web/components/processors/classifier_options';
import {convertFromClassificationResultProto} from '../../../../tasks/web/components/processors/classifier_result';
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

import {ImageClassifierOptions} from './image_classifier_options';
import {ImageClassifierResult} from './image_classifier_result';

const IMAGE_CLASSIFIER_GRAPH =
  'mediapipe.tasks.vision.image_classifier.ImageClassifierGraph';
const IMAGE_STREAM = 'input_image';
const NORM_RECT_STREAM = 'norm_rect';
const CLASSIFICATIONS_STREAM = 'classifications';

export * from './image_classifier_options';
export * from './image_classifier_result';
export {type ImageSource}; // Used in the public API

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

/** Performs classification on images. */
export class ImageClassifier extends VisionTaskRunner {
  private classificationResult: ImageClassifierResult = {classifications: []};
  private readonly options = new ImageClassifierGraphOptions();

  /**
   * Initializes the Wasm runtime and creates a new image classifier from the
   * provided options.
   * @export
   * @param wasmFileset A configuration object that provides the location
   *     Wasm binary and its loader.
   * @param imageClassifierOptions The options for the image classifier. Note
   *     that either a path to the model asset or a model buffer needs to be
   *     provided (via `baseOptions`).
   */
  static createFromOptions(
    wasmFileset: WasmFileset,
    imageClassifierOptions: ImageClassifierOptions,
  ): Promise<ImageClassifier> {
    return VisionTaskRunner.createVisionInstance(
      ImageClassifier,
      wasmFileset,
      imageClassifierOptions,
    );
  }

  /**
   * Initializes the Wasm runtime and creates a new image classifier based on
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
  ): Promise<ImageClassifier> {
    return VisionTaskRunner.createVisionInstance(ImageClassifier, wasmFileset, {
      baseOptions: {modelAssetBuffer},
    });
  }

  /**
   * Initializes the Wasm runtime and creates a new image classifier based on
   * the path to the model asset.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   */
  static createFromModelPath(
    wasmFileset: WasmFileset,
    modelAssetPath: string,
  ): Promise<ImageClassifier> {
    return VisionTaskRunner.createVisionInstance(ImageClassifier, wasmFileset, {
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
      /* roiAllowed= */ true,
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
   * Sets new options for the image classifier.
   *
   * Calling `setOptions()` with a subset of options only affects those options.
   * You can reset an option back to its default value by explicitly setting it
   * to `undefined`.
   *
   * @export
   * @param options The options for the image classifier.
   */
  override setOptions(options: ImageClassifierOptions): Promise<void> {
    this.options.setClassifierOptions(
      convertClassifierOptionsToProto(
        options,
        this.options.getClassifierOptions(),
      ),
    );
    return this.applyOptions(options);
  }

  /**
   * Performs image classification on the provided single image and waits
   * synchronously for the response. Only use this method when the
   * ImageClassifier is created with running mode `image`.
   *
   * @export
   * @param image An image to process.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @return The classification result of the image
   */
  classify(
    image: ImageSource,
    imageProcessingOptions?: ImageProcessingOptions,
  ): ImageClassifierResult {
    this.classificationResult = {classifications: []};
    this.processImageData(image, imageProcessingOptions);
    return this.classificationResult;
  }

  /**
   * Performs image classification on the provided video frame and waits
   * synchronously for the response. Only use this method when the
   * ImageClassifier is created with running mode `video`.
   *
   * @export
   * @param videoFrame A video frame to process.
   * @param timestamp The timestamp of the current frame, in ms.
   * @param imageProcessingOptions the `ImageProcessingOptions` specifying how
   *    to process the input image before running inference.
   * @return The classification result of the image
   */
  classifyForVideo(
    videoFrame: ImageSource,
    timestamp: number,
    imageProcessingOptions?: ImageProcessingOptions,
  ): ImageClassifierResult {
    this.classificationResult = {classifications: []};
    this.processVideoData(videoFrame, imageProcessingOptions, timestamp);
    return this.classificationResult;
  }

  /** Updates the MediaPipe graph configuration. */
  protected override refreshGraph(): void {
    const graphConfig = new CalculatorGraphConfig();
    graphConfig.addInputStream(IMAGE_STREAM);
    graphConfig.addInputStream(NORM_RECT_STREAM);
    graphConfig.addOutputStream(CLASSIFICATIONS_STREAM);

    const calculatorOptions = new CalculatorOptions();
    calculatorOptions.setExtension(
      ImageClassifierGraphOptions.ext,
      this.options,
    );

    // Perform image classification. Pre-processing and results post-processing
    // are built-in.
    const classifierNode = new CalculatorGraphConfig.Node();
    classifierNode.setCalculator(IMAGE_CLASSIFIER_GRAPH);
    classifierNode.addInputStream('IMAGE:' + IMAGE_STREAM);
    classifierNode.addInputStream('NORM_RECT:' + NORM_RECT_STREAM);
    classifierNode.addOutputStream('CLASSIFICATIONS:' + CLASSIFICATIONS_STREAM);
    classifierNode.setOptions(calculatorOptions);

    graphConfig.addNode(classifierNode);

    this.graphRunner.attachProtoListener(
      CLASSIFICATIONS_STREAM,
      (binaryProto, timestamp) => {
        this.classificationResult = convertFromClassificationResultProto(
          ClassificationResult.deserializeBinary(binaryProto),
        );
        this.setLatestOutputTimestamp(timestamp);
      },
    );
    this.graphRunner.attachEmptyPacketListener(
      CLASSIFICATIONS_STREAM,
      (timestamp) => {
        this.setLatestOutputTimestamp(timestamp);
      },
    );

    const binaryGraph = graphConfig.serializeBinary();
    this.setGraph(new Uint8Array(binaryGraph), /* isBinary= */ true);
  }
}


