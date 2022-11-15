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
import {EmbeddingResult} from '../../../../tasks/cc/components/containers/proto/embeddings_pb';
import {ImageEmbedderGraphOptions} from '../../../../tasks/cc/vision/image_embedder/proto/image_embedder_graph_options_pb';
import {convertBaseOptionsToProto} from '../../../../tasks/web/components/processors/base_options';
import {convertEmbedderOptionsToProto} from '../../../../tasks/web/components/processors/embedder_options';
import {convertFromEmbeddingResultProto} from '../../../../tasks/web/components/processors/embedder_result';
import {TaskRunner} from '../../../../tasks/web/core/task_runner';
import {WasmLoaderOptions} from '../../../../tasks/web/core/wasm_loader_options';
import {configureRunningMode} from '../../../../tasks/web/vision/core/running_mode';
import {createMediaPipeLib, FileLocator, ImageSource} from '../../../../web/graph_runner/wasm_mediapipe_lib';
// Placeholder for internal dependency on trusted resource url

import {ImageEmbedderOptions} from './image_embedder_options';
import {ImageEmbedderResult} from './image_embedder_result';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

const INPUT_STREAM = 'image_in';
const EMBEDDINGS_STREAM = 'embeddings_out';
const TEXT_EMBEDDER_CALCULATOR =
    'mediapipe.tasks.vision.image_embedder.ImageEmbedderGraph';

export * from './image_embedder_options';
export * from './image_embedder_result';
export {ImageSource};  // Used in the public API

/** Performs embedding extraction on images. */
export class ImageEmbedder extends TaskRunner {
  private readonly options = new ImageEmbedderGraphOptions();
  private embeddings: ImageEmbedderResult = {embeddings: []};

  /**
   * Initializes the Wasm runtime and creates a new image embedder from the
   * provided options.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param imageEmbedderOptions The options for the image embedder. Note that
   *     either a path to the TFLite model or the model itself needs to be
   *     provided (via `baseOptions`).
   */
  static async createFromOptions(
      wasmLoaderOptions: WasmLoaderOptions,
      imageEmbedderOptions: ImageEmbedderOptions): Promise<ImageEmbedder> {
    // Create a file locator based on the loader options
    const fileLocator: FileLocator = {
      locateFile() {
        // The only file we load is the Wasm binary
        return wasmLoaderOptions.wasmBinaryPath.toString();
      }
    };

    const embedder = await createMediaPipeLib(
        ImageEmbedder, wasmLoaderOptions.wasmLoaderPath,
        /* assetLoaderScript= */ undefined,
        /* glCanvas= */ undefined, fileLocator);
    await embedder.setOptions(imageEmbedderOptions);
    return embedder;
  }

  /**
   * Initializes the Wasm runtime and creates a new image embedder based on the
   * provided model asset buffer.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param modelAssetBuffer A binary representation of the TFLite model.
   */
  static createFromModelBuffer(
      wasmLoaderOptions: WasmLoaderOptions,
      modelAssetBuffer: Uint8Array): Promise<ImageEmbedder> {
    return ImageEmbedder.createFromOptions(
        wasmLoaderOptions, {baseOptions: {modelAssetBuffer}});
  }

  /**
   * Initializes the Wasm runtime and creates a new image embedder based on the
   * path to the model asset.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param modelAssetPath The path to the TFLite model.
   */
  static async createFromModelPath(
      wasmLoaderOptions: WasmLoaderOptions,
      modelAssetPath: string): Promise<ImageEmbedder> {
    const response = await fetch(modelAssetPath.toString());
    const graphData = await response.arrayBuffer();
    return ImageEmbedder.createFromModelBuffer(
        wasmLoaderOptions, new Uint8Array(graphData));
  }

  /**
   * Sets new options for the image embedder.
   *
   * Calling `setOptions()` with a subset of options only affects those options.
   * You can reset an option back to its default value by explicitly setting it
   * to `undefined`.
   *
   * @param options The options for the image embedder.
   */
  async setOptions(options: ImageEmbedderOptions): Promise<void> {
    let baseOptionsProto = this.options.getBaseOptions();
    if (options.baseOptions) {
      baseOptionsProto = await convertBaseOptionsToProto(
          options.baseOptions, baseOptionsProto);
    }
    baseOptionsProto = configureRunningMode(options, baseOptionsProto);
    this.options.setBaseOptions(baseOptionsProto);

    this.options.setEmbedderOptions(convertEmbedderOptionsToProto(
        options, this.options.getEmbedderOptions()));

    this.refreshGraph();
  }

  /**
   * Performs embedding extraction on the provided image and waits synchronously
   * for the response.
   *
   * Only use this method when the `useStreamMode` option is not set or
   * expliclity set to `false`.
   *
   * @param image The image to process.
   * @return The classification result of the image
   */
  embed(image: ImageSource): ImageEmbedderResult {
    if (!!this.options.getBaseOptions()?.getUseStreamMode()) {
      throw new Error(
          'Task is not initialized with image mode. ' +
          '\'runningMode\' must be set to \'image\'.');
    }
    return this.performEmbeddingExtraction(image, performance.now());
  }

  /**
   * Performs embedding extraction on the provided video frame and waits
   * synchronously for the response.
   *
   * Only use this method when the `useStreamMode` option is set to `true`.
   *
   * @param imageFrame The image frame to process.
   * @param timestamp The timestamp of the current frame, in ms.
   * @return The classification result of the image
   */
  embedForVideo(imageFrame: ImageSource, timestamp: number):
      ImageEmbedderResult {
    if (!this.options.getBaseOptions()?.getUseStreamMode()) {
      throw new Error(
          'Task is not initialized with video mode. ' +
          '\'runningMode\' must be set to \'video\' or \'live_stream\'.');
    }
    return this.performEmbeddingExtraction(imageFrame, timestamp);
  }

  /** Runs the embedding extractio and blocks on the response. */
  private performEmbeddingExtraction(image: ImageSource, timestamp: number):
      ImageEmbedderResult {
    // Get embeddings by running our MediaPipe graph.
    this.addGpuBufferAsImageToStream(
        image, INPUT_STREAM, timestamp ?? performance.now());
    this.finishProcessing();
    return this.embeddings;
  }

  /**
   * Internal function for converting raw data into an embedding, and setting it
   * as our embeddings result.
   */
  private addJsImageEmdedding(binaryProto: Uint8Array): void {
    const embeddingResult = EmbeddingResult.deserializeBinary(binaryProto);
    this.embeddings = convertFromEmbeddingResultProto(embeddingResult);
  }

  /** Updates the MediaPipe graph configuration. */
  private refreshGraph(): void {
    const graphConfig = new CalculatorGraphConfig();
    graphConfig.addInputStream(INPUT_STREAM);
    graphConfig.addOutputStream(EMBEDDINGS_STREAM);

    const calculatorOptions = new CalculatorOptions();
    calculatorOptions.setExtension(ImageEmbedderGraphOptions.ext, this.options);

    const embedderNode = new CalculatorGraphConfig.Node();
    embedderNode.setCalculator(TEXT_EMBEDDER_CALCULATOR);
    embedderNode.addInputStream('IMAGE:' + INPUT_STREAM);
    embedderNode.addOutputStream('EMBEDDINGS:' + EMBEDDINGS_STREAM);
    embedderNode.setOptions(calculatorOptions);

    graphConfig.addNode(embedderNode);

    this.attachProtoListener(EMBEDDINGS_STREAM, binaryProto => {
      this.addJsImageEmdedding(binaryProto);
    });

    const binaryGraph = graphConfig.serializeBinary();
    this.setGraph(new Uint8Array(binaryGraph), /* isBinary= */ true);
  }
}


