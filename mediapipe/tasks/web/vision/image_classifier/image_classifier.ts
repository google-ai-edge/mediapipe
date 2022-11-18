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
import {ClassificationResult} from '../../../../tasks/cc/components/containers/proto/classifications_pb';
import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {ImageClassifierGraphOptions} from '../../../../tasks/cc/vision/image_classifier/proto/image_classifier_graph_options_pb';
import {convertClassifierOptionsToProto} from '../../../../tasks/web/components/processors/classifier_options';
import {convertFromClassificationResultProto} from '../../../../tasks/web/components/processors/classifier_result';
import {WasmLoaderOptions} from '../../../../tasks/web/core/wasm_loader_options';
import {VisionTaskRunner} from '../../../../tasks/web/vision/core/vision_task_runner';
import {createMediaPipeLib, FileLocator, ImageSource} from '../../../../web/graph_runner/graph_runner';
// Placeholder for internal dependency on trusted resource url

import {ImageClassifierOptions} from './image_classifier_options';
import {ImageClassifierResult} from './image_classifier_result';

const IMAGE_CLASSIFIER_GRAPH =
    'mediapipe.tasks.vision.image_classifier.ImageClassifierGraph';
const INPUT_STREAM = 'input_image';
const CLASSIFICATIONS_STREAM = 'classifications';

export * from './image_classifier_options';
export * from './image_classifier_result';
export {ImageSource};  // Used in the public API

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

/** Performs classification on images. */
export class ImageClassifier extends VisionTaskRunner<ImageClassifierResult> {
  private classificationResult: ImageClassifierResult = {classifications: []};
  private readonly options = new ImageClassifierGraphOptions();

  /**
   * Initializes the Wasm runtime and creates a new image classifier from the
   * provided options.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param imageClassifierOptions The options for the image classifier. Note
   *     that either a path to the model asset or a model buffer needs to be
   *     provided (via `baseOptions`).
   */
  static async createFromOptions(
      wasmLoaderOptions: WasmLoaderOptions,
      imageClassifierOptions: ImageClassifierOptions):
      Promise<ImageClassifier> {
    // Create a file locator based on the loader options
    const fileLocator: FileLocator = {
      locateFile() {
        // The only file we load is the Wasm binary
        return wasmLoaderOptions.wasmBinaryPath.toString();
      }
    };

    const classifier = await createMediaPipeLib(
        ImageClassifier, wasmLoaderOptions.wasmLoaderPath,
        /* assetLoaderScript= */ undefined,
        /* glCanvas= */ undefined, fileLocator);
    await classifier.setOptions(imageClassifierOptions);
    return classifier;
  }

  /**
   * Initializes the Wasm runtime and creates a new image classifier based on
   * the provided model asset buffer.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param modelAssetBuffer A binary representation of the model.
   */
  static createFromModelBuffer(
      wasmLoaderOptions: WasmLoaderOptions,
      modelAssetBuffer: Uint8Array): Promise<ImageClassifier> {
    return ImageClassifier.createFromOptions(
        wasmLoaderOptions, {baseOptions: {modelAssetBuffer}});
  }

  /**
   * Initializes the Wasm runtime and creates a new image classifier based on
   * the path to the model asset.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   */
  static async createFromModelPath(
      wasmLoaderOptions: WasmLoaderOptions,
      modelAssetPath: string): Promise<ImageClassifier> {
    const response = await fetch(modelAssetPath.toString());
    const graphData = await response.arrayBuffer();
    return ImageClassifier.createFromModelBuffer(
        wasmLoaderOptions, new Uint8Array(graphData));
  }

  protected override get baseOptions(): BaseOptionsProto|undefined {
    return this.options.getBaseOptions();
  }

  protected override set baseOptions(proto: BaseOptionsProto|undefined) {
    this.options.setBaseOptions(proto);
  }

  /**
   * Sets new options for the image classifier.
   *
   * Calling `setOptions()` with a subset of options only affects those options.
   * You can reset an option back to its default value by explicitly setting it
   * to `undefined`.
   *
   * @param options The options for the image classifier.
   */
  override async setOptions(options: ImageClassifierOptions): Promise<void> {
    await super.setOptions(options);
    this.options.setClassifierOptions(convertClassifierOptionsToProto(
        options, this.options.getClassifierOptions()));
    this.refreshGraph();
  }

  /**
   * Performs image classification on the provided single image and waits
   * synchronously for the response.
   *
   * @param image An image to process.
   * @return The classification result of the image
   */
  classify(image: ImageSource): ImageClassifierResult {
    return this.processImageData(image);
  }

  /**
   * Performs image classification on the provided video frame and waits
   * synchronously for the response.
   *
   * @param videoFrame A video frame to process.
   * @param timestamp The timestamp of the current frame, in ms.
   * @return The classification result of the image
   */
  classifyForVideo(videoFrame: ImageSource, timestamp: number):
      ImageClassifierResult {
    return this.processVideoData(videoFrame, timestamp);
  }

  /** Runs the image classification graph and blocks on the response. */
  protected override process(imageSource: ImageSource, timestamp: number):
      ImageClassifierResult {
    // Get classification result by running our MediaPipe graph.
    this.classificationResult = {classifications: []};
    this.addGpuBufferAsImageToStream(
        imageSource, INPUT_STREAM, timestamp ?? performance.now());
    this.finishProcessing();
    return this.classificationResult;
  }

  /** Updates the MediaPipe graph configuration. */
  private refreshGraph(): void {
    const graphConfig = new CalculatorGraphConfig();
    graphConfig.addInputStream(INPUT_STREAM);
    graphConfig.addOutputStream(CLASSIFICATIONS_STREAM);

    const calculatorOptions = new CalculatorOptions();
    calculatorOptions.setExtension(
        ImageClassifierGraphOptions.ext, this.options);

    // Perform image classification. Pre-processing and results post-processing
    // are built-in.
    const classifierNode = new CalculatorGraphConfig.Node();
    classifierNode.setCalculator(IMAGE_CLASSIFIER_GRAPH);
    classifierNode.addInputStream('IMAGE:' + INPUT_STREAM);
    classifierNode.addOutputStream('CLASSIFICATIONS:' + CLASSIFICATIONS_STREAM);
    classifierNode.setOptions(calculatorOptions);

    graphConfig.addNode(classifierNode);

    this.attachProtoListener(CLASSIFICATIONS_STREAM, binaryProto => {
      this.classificationResult = convertFromClassificationResultProto(
          ClassificationResult.deserializeBinary(binaryProto));
    });

    const binaryGraph = graphConfig.serializeBinary();
    this.setGraph(new Uint8Array(binaryGraph), /* isBinary= */ true);
  }
}


