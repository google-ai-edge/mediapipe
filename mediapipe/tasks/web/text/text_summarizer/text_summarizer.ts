/**
 * Copyright 2026 The MediaPipe Authors.
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

import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {
  CachedGraphRunner,
  TaskRunner,
} from '../../../../tasks/web/core/task_runner';
import {WasmFileset} from '../../../../tasks/web/core/wasm_fileset';
import {WasmModule} from '../../../../web/graph_runner/graph_runner';
// Placeholder for internal dependency on trusted resource url
import type {
  SummarizationMode,
  TextSummarizerOptions,
} from './text_summarizer_options';
import type {TextSummarizerResult} from './text_summarizer_result';

export type {SummarizationMode, TextSummarizerOptions, TextSummarizerResult};

/**
 * The Wasm module for TextSummarizer with custom C++ bindings.
 * This interface defines the Emscripten filesystem API and the C++ functions
 * exposed via Embind, which are required for the TypeScript task implementation
 * to manage the model files and execute text summarization.
 */
declare interface TextSummarizerWasmModule {
  /**
   * Creates a new TextSummarizer instance.
   * @param modelPath The path to the model asset.
   * @param maxNumTokens The maximum number of tokens to generate.
   * @param mode The summarization mode.
   * @return A pointer to the created TextSummarizer instance.
   */
  createTextSummarizer(
    modelPath: string,
    maxNumTokens: number,
    mode: string,
  ): number;

  /**
   * Deletes a TextSummarizer instance.
   * @param summarizerPtr A pointer to the TextSummarizer instance to delete.
   */
  deleteTextSummarizer(summarizerPtr: number): void;

  /**
   * Summarizes the given text.
   * @param summarizerPtr A pointer to the TextSummarizer instance.
   * @param text The input text to be summarized.
   * @return The generated summary.
   */
  summarize(summarizerPtr: number, text: string): string;
}

/**
 * MediaPipe Task that performs text summarization.
 */
export class TextSummarizer extends TaskRunner {
  protected override baseOptions: BaseOptionsProto = new BaseOptionsProto();
  private maxNumTokens = 0;
  private mode: SummarizationMode = 'KEYPOINTS';
  private summarizerPtr = 0;

  /**
   * Initializes the Wasm runtime and creates a new text summarizer based on the
   * provided options.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param textSummarizerOptions The options for TextSummarizer.
   */
  static createFromOptions(
    wasmFileset: WasmFileset,
    textSummarizerOptions: TextSummarizerOptions,
  ): Promise<TextSummarizer> {
    return TaskRunner.createInstance(
      TextSummarizer,
      /* canvas= */ null,
      wasmFileset,
      textSummarizerOptions,
    );
  }

  /**
   * Initializes the Wasm runtime and creates a new text summarizer based on the
   * path to the model asset.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   * @param maxNumTokens The maximum number of tokens to generate.
   * @param mode The summarization mode.
   */
  static createFromModelPath(
    wasmFileset: WasmFileset,
    modelAssetPath: string,
    maxNumTokens = 0,
    mode: SummarizationMode = 'KEYPOINTS',
  ): Promise<TextSummarizer> {
    return TextSummarizer.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetPath,
      },
      maxNumTokens,
      mode,
    });
  }

  /**
   * Initializes the Wasm runtime and creates a new text summarizer based on the
   * provided model asset buffer.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param modelAssetBuffer An array or a stream containing a binary
   *    representation of the model.
   * @param maxNumTokens The maximum number of tokens to generate.
   * @param mode The summarization mode.
   */
  static createFromModelBuffer(
    wasmFileset: WasmFileset,
    modelAssetBuffer: Uint8Array | ReadableStreamDefaultReader,
    maxNumTokens = 0,
    mode: SummarizationMode = 'KEYPOINTS',
  ): Promise<TextSummarizer> {
    return TextSummarizer.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetBuffer,
      },
      maxNumTokens,
      mode,
    });
  }

  /** @hideconstructor */
  constructor(
    wasmModule: WasmModule,
    glCanvas?: HTMLCanvasElement | OffscreenCanvas | null,
  ) {
    super(new CachedGraphRunner(wasmModule, glCanvas));
  }

  protected override getTaskName(): string {
    return 'TextSummarizer';
  }

  /**
   * Sets the initial options for the text summarizer.
   * LiteRT LM Tasks do not support updating options after initialization.
   * @param options The options for the text summarizer.
   */
  override async setOptions(options: TextSummarizerOptions): Promise<void> {
    if (this.summarizerPtr !== 0) {
      throw new Error(
        'LiteRT LM Tasks do not support updating options after initialization.',
      );
    }
    if (options.maxNumTokens !== undefined) {
      this.maxNumTokens = options.maxNumTokens;
    }
    if (options.mode !== undefined) {
      this.mode = options.mode;
    }
    const loadTfliteModel =
      !!options.baseOptions &&
      (!!options.baseOptions.modelAssetPath ||
        !!options.baseOptions.modelAssetBuffer);
    return this.applyOptions(
      options,
      /* loadTfliteModel= */ loadTfliteModel,
      /* isLiteRtLmModel= */ true,
    );
  }

  protected override refreshGraph(): void {
    const wasmModule = this.graphRunner
      .wasmModule as unknown as TextSummarizerWasmModule;
    const modelPath = '/model.litertlm';

    this.summarizerPtr = wasmModule.createTextSummarizer(
      modelPath,
      this.maxNumTokens,
      this.mode,
    );
  }

  /**
   * Summarizes the given text.
   * @param text The input text to be summarized.
   * @return The generated summary.
   */
  summarize(text: string): TextSummarizerResult {
    const wasmModule = this.graphRunner
      .wasmModule as unknown as TextSummarizerWasmModule;
    if (this.summarizerPtr === 0) {
      throw new Error('TextSummarizer is already closed.');
    }
    return wasmModule.summarize(this.summarizerPtr, text);
  }

  /**
   * Closes the TextSummarizer and frees resources.
   */
  override close(): void {
    const wasmModule = this.graphRunner
      .wasmModule as unknown as TextSummarizerWasmModule;
    if (this.summarizerPtr !== 0) {
      try {
        wasmModule.deleteTextSummarizer(this.summarizerPtr);
      } catch {}
      this.summarizerPtr = 0;
    }
    super.close();
  }
}


