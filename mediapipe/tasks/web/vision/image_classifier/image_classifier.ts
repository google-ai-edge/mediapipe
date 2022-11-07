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
import {ImageClassifierGraphOptions} from '../../../../tasks/cc/vision/image_classifier/proto/image_classifier_graph_options_pb';
import {convertBaseOptionsToProto} from '../../../../tasks/web/components/processors/base_options';
import {convertClassifierOptionsToProto} from '../../../../tasks/web/components/processors/classifier_options';
import {convertFromClassificationResultProto} from '../../../../tasks/web/components/processors/classifier_result';
import {TaskRunner} from '../../../../tasks/web/core/task_runner';
import {WasmLoaderOptions} from '../../../../tasks/web/core/wasm_loader_options';
import {createMediaPipeLib, FileLocator, ImageSource} from '../../../../web/graph_runner/wasm_mediapipe_lib';
// Placeholder for internal dependency on trusted resource url

import {ImageClassifierOptions} from './image_classifier_options';
import {Classifications} from './image_classifier_result';

const IMAGE_CLASSIFIER_GRAPH =
    'mediapipe.tasks.vision.image_classifier.ImageClassifierGraph';
const INPUT_STREAM = 'input_image';
const CLASSIFICATION_RESULT_STREAM = 'classification_result';

export {ImageSource};  // Used in the public API

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

/** Performs classification on images. */
export class ImageClassifier extends TaskRunner {
  private classifications: Classifications[] = [];
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

  /**
   * Sets new options for the image classifier.
   *
   * Calling `setOptions()` with a subset of options only affects those options.
   * You can reset an option back to its default value by explicitly setting it
   * to `undefined`.
   *
   * @param options The options for the image classifier.
   */
  async setOptions(options: ImageClassifierOptions): Promise<void> {
    if (options.baseOptions) {
      const baseOptionsProto =
          await convertBaseOptionsToProto(options.baseOptions);
      this.options.setBaseOptions(baseOptionsProto);
    }

    this.options.setClassifierOptions(convertClassifierOptionsToProto(
        options, this.options.getClassifierOptions()));
    this.refreshGraph();
  }

  /**
   * Performs image classification on the provided image and waits synchronously
   * for the response.
   *
   * @param imageSource An image source to process.
   * @param timestamp The timestamp of the current frame, in ms. If not
   *     provided, defaults to `performance.now()`.
   * @return The classification result of the image
   */
  classify(imageSource: ImageSource, timestamp?: number): Classifications[] {
    // Get classification classes by running our MediaPipe graph.
    this.classifications = [];
    this.addGpuBufferAsImageToStream(
        imageSource, INPUT_STREAM, timestamp ?? performance.now());
    this.finishProcessing();
    return [...this.classifications];
  }

  /**
   * Internal function for converting raw data into a classification, and
   * adding it to our classfications list.
   **/
  private addJsImageClassification(binaryProto: Uint8Array): void {
    const classificationResult =
        ClassificationResult.deserializeBinary(binaryProto);
    this.classifications.push(
        ...convertFromClassificationResultProto(classificationResult));
  }

  /** Updates the MediaPipe graph configuration. */
  private refreshGraph(): void {
    const graphConfig = new CalculatorGraphConfig();
    graphConfig.addInputStream(INPUT_STREAM);
    graphConfig.addOutputStream(CLASSIFICATION_RESULT_STREAM);

    const calculatorOptions = new CalculatorOptions();
    calculatorOptions.setExtension(
        ImageClassifierGraphOptions.ext, this.options);

    // Perform image classification. Pre-processing and results post-processing
    // are built-in.
    const classifierNode = new CalculatorGraphConfig.Node();
    classifierNode.setCalculator(IMAGE_CLASSIFIER_GRAPH);
    classifierNode.addInputStream('IMAGE:' + INPUT_STREAM);
    classifierNode.addOutputStream(
        'CLASSIFICATION_RESULT:' + CLASSIFICATION_RESULT_STREAM);
    classifierNode.setOptions(calculatorOptions);

    graphConfig.addNode(classifierNode);

    this.attachProtoListener(CLASSIFICATION_RESULT_STREAM, binaryProto => {
      this.addJsImageClassification(binaryProto);
    });

    const binaryGraph = graphConfig.serializeBinary();
    this.setGraph(new Uint8Array(binaryGraph), /* isBinary= */ true);
  }
}


