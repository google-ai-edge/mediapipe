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
import {TextClassifierGraphOptions} from '../../../../tasks/cc/text/text_classifier/proto/text_classifier_graph_options_pb';
import {convertClassifierOptionsToProto} from '../../../../tasks/web/components/processors/classifier_options';
import {convertFromClassificationResultProto} from '../../../../tasks/web/components/processors/classifier_result';
import {
  CachedGraphRunner,
  TaskRunner,
} from '../../../../tasks/web/core/task_runner';
import {WasmFileset} from '../../../../tasks/web/core/wasm_fileset';
import {WasmModule} from '../../../../web/graph_runner/graph_runner';
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
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param textClassifierOptions The options for the text classifier. Note that
   *     either a path to the TFLite model or the model itself needs to be
   *     provided (via `baseOptions`).
   */
  static createFromOptions(
    wasmFileset: WasmFileset,
    textClassifierOptions: TextClassifierOptions,
  ): Promise<TextClassifier> {
    return TaskRunner.createInstance(
      TextClassifier,
      /* canvas= */ null,
      wasmFileset,
      textClassifierOptions,
    );
  }

  /**
   * Initializes the Wasm runtime and creates a new text classifier based on the
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
  ): Promise<TextClassifier> {
    return TaskRunner.createInstance(
      TextClassifier,
      /* canvas= */ null,
      wasmFileset,
      {baseOptions: {modelAssetBuffer}},
    );
  }

  /**
   * Initializes the Wasm runtime and creates a new text classifier based on the
   * path to the model asset.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   */
  static createFromModelPath(
    wasmFileset: WasmFileset,
    modelAssetPath: string,
  ): Promise<TextClassifier> {
    return TaskRunner.createInstance(
      TextClassifier,
      /* canvas= */ null,
      wasmFileset,
      {baseOptions: {modelAssetPath}},
    );
  }

  /** @hideconstructor */
  constructor(
    wasmModule: WasmModule,
    glCanvas?: HTMLCanvasElement | OffscreenCanvas | null,
  ) {
    super(new CachedGraphRunner(wasmModule, glCanvas));
    this.options.setBaseOptions(new BaseOptionsProto());
  }

  /**
   * Sets new options for the text classifier.
   *
   * Calling `setOptions()` with a subset of options only affects those options.
   * You can reset an option back to its default value by explicitly setting it
   * to `undefined`.
   *
   * @export
   * @param options The options for the text classifier.
   */
  override setOptions(options: TextClassifierOptions): Promise<void> {
    this.options.setClassifierOptions(
      convertClassifierOptionsToProto(
        options,
        this.options.getClassifierOptions(),
      ),
    );
    return this.applyOptions(options);
  }

  protected override get baseOptions(): BaseOptionsProto {
    return this.options.getBaseOptions()!;
  }

  protected override set baseOptions(proto: BaseOptionsProto) {
    this.options.setBaseOptions(proto);
  }

  /**
   * Performs Natural Language classification on the provided text and waits
   * synchronously for the response.
   *
   * @export
   * @param text The text to process.
   * @return The classification result of the text
   */
  classify(text: string): TextClassifierResult {
    this.classificationResult = {classifications: []};
    this.graphRunner.addStringToStream(
      text,
      INPUT_STREAM,
      this.getSynctheticTimestamp(),
    );
    this.finishProcessing();
    return this.classificationResult;
  }

  /** Updates the MediaPipe graph configuration. */
  protected override refreshGraph(): void {
    const graphConfig = new CalculatorGraphConfig();
    graphConfig.addInputStream(INPUT_STREAM);
    graphConfig.addOutputStream(CLASSIFICATIONS_STREAM);

    const calculatorOptions = new CalculatorOptions();
    calculatorOptions.setExtension(
      TextClassifierGraphOptions.ext,
      this.options,
    );

    const classifierNode = new CalculatorGraphConfig.Node();
    classifierNode.setCalculator(TEXT_CLASSIFIER_GRAPH);
    classifierNode.addInputStream('TEXT:' + INPUT_STREAM);
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


