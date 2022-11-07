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
import {AudioClassifierGraphOptions} from '../../../../tasks/cc/audio/audio_classifier/proto/audio_classifier_graph_options_pb';
import {ClassificationResult} from '../../../../tasks/cc/components/containers/proto/classifications_pb';
import {convertBaseOptionsToProto} from '../../../../tasks/web/components/processors/base_options';
import {convertClassifierOptionsToProto} from '../../../../tasks/web/components/processors/classifier_options';
import {convertFromClassificationResultProto} from '../../../../tasks/web/components/processors/classifier_result';
import {TaskRunner} from '../../../../tasks/web/core/task_runner';
import {WasmLoaderOptions} from '../../../../tasks/web/core/wasm_loader_options';
import {createMediaPipeLib, FileLocator} from '../../../../web/graph_runner/wasm_mediapipe_lib';
// Placeholder for internal dependency on trusted resource url

import {AudioClassifierOptions} from './audio_classifier_options';
import {Classifications} from './audio_classifier_result';

const MEDIAPIPE_GRAPH =
    'mediapipe.tasks.audio.audio_classifier.AudioClassifierGraph';

// Note: `input_audio` is hardcoded in 'gl_graph_runner_internal_audio' and
// cannot be changed
// TODO: Change this to `audio_in` to match the name in the CC
// implementation
const AUDIO_STREAM = 'input_audio';
const SAMPLE_RATE_STREAM = 'sample_rate';
const CLASSIFICATION_RESULT_STREAM = 'classification_result';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

/** Performs audio classification. */
export class AudioClassifier extends TaskRunner {
  private classifications: Classifications[] = [];
  private defaultSampleRate = 48000;
  private readonly options = new AudioClassifierGraphOptions();

  /**
   * Initializes the Wasm runtime and creates a new audio classifier from the
   * provided options.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param audioClassifierOptions The options for the audio classifier. Note
   *     that either a path to the model asset or a model buffer needs to be
   *     provided (via `baseOptions`).
   */
  static async createFromOptions(
      wasmLoaderOptions: WasmLoaderOptions,
      audioClassifierOptions: AudioClassifierOptions):
      Promise<AudioClassifier> {
    // Create a file locator based on the loader options
    const fileLocator: FileLocator = {
      locateFile() {
        // The only file loaded with this mechanism is the Wasm binary
        return wasmLoaderOptions.wasmBinaryPath.toString();
      }
    };

    const classifier = await createMediaPipeLib(
        AudioClassifier, wasmLoaderOptions.wasmLoaderPath,
        /* assetLoaderScript= */ undefined,
        /* glCanvas= */ undefined, fileLocator);
    await classifier.setOptions(audioClassifierOptions);
    return classifier;
  }

  /**
   * Initializes the Wasm runtime and creates a new audio classifier based on
   * the provided model asset buffer.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param modelAssetBuffer A binary representation of the model.
   */
  static createFromModelBuffer(
      wasmLoaderOptions: WasmLoaderOptions,
      modelAssetBuffer: Uint8Array): Promise<AudioClassifier> {
    return AudioClassifier.createFromOptions(
        wasmLoaderOptions, {baseOptions: {modelAssetBuffer}});
  }

  /**
   * Initializes the Wasm runtime and creates a new audio classifier based on
   * the path to the model asset.
   * @param wasmLoaderOptions A configuration object that provides the location
   *     of the Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   */
  static async createFromModelPath(
      wasmLoaderOptions: WasmLoaderOptions,
      modelAssetPath: string): Promise<AudioClassifier> {
    const response = await fetch(modelAssetPath.toString());
    const graphData = await response.arrayBuffer();
    return AudioClassifier.createFromModelBuffer(
        wasmLoaderOptions, new Uint8Array(graphData));
  }

  /**
   * Sets new options for the audio classifier.
   *
   * Calling `setOptions()` with a subset of options only affects those options.
   * You can reset an option back to its default value by explicitly setting it
   * to `undefined`.
   *
   * @param options The options for the audio classifier.
   */
  async setOptions(options: AudioClassifierOptions): Promise<void> {
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
   * Sets the sample rate for all calls to `classify()` that omit an explicit
   * sample rate. `48000` is used as a default if this method is not called.
   *
   * @param sampleRate A sample rate (e.g. `44100`).
   */
  setDefaultSampleRate(sampleRate: number) {
    this.defaultSampleRate = sampleRate;
  }

  /**
   * Performs audio classification on the provided audio data and waits
   * synchronously for the response.
   *
   * @param audioData An array of raw audio capture data, like
   *     from a call to getChannelData on an AudioBuffer.
   * @param sampleRate The sample rate in Hz of the provided audio data. If not
   *     set, defaults to the sample rate set via `setDefaultSampleRate()` or
   *     `48000` if no custom default was set.
   * @return The classification result of the audio datas
   */
  classify(audioData: Float32Array, sampleRate?: number): Classifications[] {
    sampleRate = sampleRate ?? this.defaultSampleRate;

    // Configures the number of samples in the WASM layer. We re-configure the
    // number of samples and the sample rate for every frame, but ignore other
    // side effects of this function (such as sending the input side packet and
    // the input stream header).
    this.configureAudio(
        /* numChannels= */ 1, /* numSamples= */ audioData.length, sampleRate);

    const timestamp = performance.now();
    this.addDoubleToStream(sampleRate, SAMPLE_RATE_STREAM, timestamp);
    this.addAudioToStream(audioData, timestamp);

    this.classifications = [];
    this.finishProcessing();
    return [...this.classifications];
  }

  /**
   * Internal function for converting raw data into a classification, and
   * adding it to our classfications list.
   **/
  private addJsAudioClassification(binaryProto: Uint8Array): void {
    const classificationResult =
        ClassificationResult.deserializeBinary(binaryProto);
    this.classifications.push(
        ...convertFromClassificationResultProto(classificationResult));
  }

  /** Updates the MediaPipe graph configuration. */
  private refreshGraph(): void {
    const graphConfig = new CalculatorGraphConfig();
    graphConfig.addInputStream(AUDIO_STREAM);
    graphConfig.addInputStream(SAMPLE_RATE_STREAM);
    graphConfig.addOutputStream(CLASSIFICATION_RESULT_STREAM);

    const calculatorOptions = new CalculatorOptions();
    calculatorOptions.setExtension(
        AudioClassifierGraphOptions.ext, this.options);

    // Perform audio classification. Pre-processing and results post-processing
    // are built-in.
    const classifierNode = new CalculatorGraphConfig.Node();
    classifierNode.setCalculator(MEDIAPIPE_GRAPH);
    classifierNode.addInputStream('AUDIO:' + AUDIO_STREAM);
    classifierNode.addInputStream('SAMPLE_RATE:' + SAMPLE_RATE_STREAM);
    classifierNode.addOutputStream(
        'CLASSIFICATION_RESULT:' + CLASSIFICATION_RESULT_STREAM);
    classifierNode.setOptions(calculatorOptions);

    graphConfig.addNode(classifierNode);

    this.attachProtoListener(CLASSIFICATION_RESULT_STREAM, binaryProto => {
      this.addJsAudioClassification(binaryProto);
    });

    const binaryGraph = graphConfig.serializeBinary();
    this.setGraph(new Uint8Array(binaryGraph), /* isBinary= */ true);
  }
}


