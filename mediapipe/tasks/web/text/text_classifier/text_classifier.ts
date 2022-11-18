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
import {TextClassifierGraphOptions} from '../../../../tasks/cc/text/text_classifier/proto/text_classifier_graph_options_pb';
import {convertBaseOptionsToProto} from '../../../../tasks/web/components/processors/base_options';
import {convertClassifierOptionsToProto} from '../../../../tasks/web/components/processors/classifier_options';
import {convertFromClassificationResultProto} from '../../../../tasks/web/components/processors/classifier_result';
import {TaskRunner} from '../../../../tasks/web/core/task_runner';
import {WasmLoaderOptions} from '../../../../tasks/web/core/wasm_loader_options';
import {createMediaPipeLib, FileLocator} from '../../../../web/graph_runner/graph_runner';
// Placeholder for internal dependency on trusted resource url

import {TextClassifierOptions} from './text_classifier_options';
import {TextClassifierResult} from './text_classifier_result';

export * from './text_classifier_options';
export * from './text_classifier_result';

const INPUT_STREAM = 'text_in';
const CLASSIFICATIONS_STREAM = 'classifications_out';
const TEXT_CLASSIFIER_GRAPH =
    'mediapipe.tasks.text.text_classifier.TextClassifierGraph';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

/** Performs Natural Language classification. */
export class TextClassifier extends TaskRunner {
  private classificationResult: TextClassifierResult = {classifications: []};
  private readonly options = new TextClassifierGraphOptions();

  /**
   * Initializes the Wasm runtime and creates a new text classifier from the
   * provided options.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param textClassifierOptions The options for the text classifier. Note that
   *     either a path to the TFLite model or the model itself needs to be
   *     provided (via `baseOptions`).
   */
  static async createFromOptions(
      wasmLoaderOptions: WasmLoaderOptions,
      textClassifierOptions: TextClassifierOptions): Promise<TextClassifier> {
    // Create a file locator based on the loader options
    const fileLocator: FileLocator = {
      locateFile() {
        // The only file we load is the Wasm binary
        return wasmLoaderOptions.wasmBinaryPath.toString();
      }
    };

    const classifier = await createMediaPipeLib(
        TextClassifier, wasmLoaderOptions.wasmLoaderPath,
        /* assetLoaderScript= */ undefined,
        /* glCanvas= */ undefined, fileLocator);
    await classifier.setOptions(textClassifierOptions);
    return classifier;
  }

  /**
   * Initializes the Wasm runtime and creates a new text classifier based on the
   * provided model asset buffer.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param modelAssetBuffer A binary representation of the model.
   */
  static createFromModelBuffer(
      wasmLoaderOptions: WasmLoaderOptions,
      modelAssetBuffer: Uint8Array): Promise<TextClassifier> {
    return TextClassifier.createFromOptions(
        wasmLoaderOptions, {baseOptions: {modelAssetBuffer}});
  }

  /**
   * Initializes the Wasm runtime and creates a new text classifier based on the
   * path to the model asset.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   */
  static async createFromModelPath(
      wasmLoaderOptions: WasmLoaderOptions,
      modelAssetPath: string): Promise<TextClassifier> {
    const response = await fetch(modelAssetPath.toString());
    const graphData = await response.arrayBuffer();
    return TextClassifier.createFromModelBuffer(
        wasmLoaderOptions, new Uint8Array(graphData));
  }

  /**
   * Sets new options for the text classifier.
   *
   * Calling `setOptions()` with a subset of options only affects those options.
   * You can reset an option back to its default value by explicitly setting it
   * to `undefined`.
   *
   * @param options The options for the text classifier.
   */
  async setOptions(options: TextClassifierOptions): Promise<void> {
    if (options.baseOptions) {
      const baseOptionsProto = await convertBaseOptionsToProto(
          options.baseOptions, this.options.getBaseOptions());
      this.options.setBaseOptions(baseOptionsProto);
    }

    this.options.setClassifierOptions(convertClassifierOptionsToProto(
        options, this.options.getClassifierOptions()));
    this.refreshGraph();
  }


  /**
   * Performs Natural Language classification on the provided text and waits
   * synchronously for the response.
   *
   * @param text The text to process.
   * @return The classification result of the text
   */
  classify(text: string): TextClassifierResult {
    // Get classification result by running our MediaPipe graph.
    this.classificationResult = {classifications: []};
    this.addStringToStream(
        text, INPUT_STREAM, /* timestamp= */ performance.now());
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
        TextClassifierGraphOptions.ext, this.options);

    const classifierNode = new CalculatorGraphConfig.Node();
    classifierNode.setCalculator(TEXT_CLASSIFIER_GRAPH);
    classifierNode.addInputStream('TEXT:' + INPUT_STREAM);
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



