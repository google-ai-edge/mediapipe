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
import type {TextProofreaderOptions} from './text_proofreader_options';
import type {
  Correction,
  CorrectionType,
  TextProofreaderResult,
} from './text_proofreader_result';

export type {
  Correction,
  CorrectionType,
  TextProofreaderOptions,
  TextProofreaderResult,
};

/**
 * A listener that receives each newly generated part of the proofread text
 * during a streaming proofread.
 * @param partialResult The newly generated part of the proofread text.
 * @param done Whether this is the last part of the proofread text.
 */
export type ProofreaderProgressListener = (
  partialResult: string,
  done: boolean,
) => void;

/**
 * The listener invoked by the Wasm module for each streaming result.
 * `errorMessage` is non-empty if proofreading failed, in which case `done` is
 * true. `corrections` is only set on the final successful result.
 */
type WasmProofreaderListener = (
  partialResult: string,
  done: boolean,
  errorMessage: string,
  corrections?: Correction[],
) => void;

/**
 * The Wasm module for TextProofreader with custom C++ bindings.
 * This interface defines the C++ functions exposed via Embind, which are
 * required for the TypeScript task implementation to execute text
 * proofreading.
 */
declare interface TextProofreaderWasmModule {
  /**
   * The listener that receives streaming results. Installed for the duration
   * of a `proofreadStreaming()` call.
   */
  _userProofreaderListener?: WasmProofreaderListener;

  /**
   * Creates a new TextProofreader instance.
   * @param modelPath The path to the model asset.
   * @param maxNumTokens The maximum number of tokens.
   * @return A pointer to the created TextProofreader instance.
   */
  createTextProofreader(modelPath: string, maxNumTokens: number): number;

  /**
   * Deletes a TextProofreader instance.
   * @param proofreaderPtr A pointer to the TextProofreader instance to delete.
   */
  deleteTextProofreader(proofreaderPtr: number): void;

  /**
   * Proofreads the given text.
   * @param proofreaderPtr A pointer to the TextProofreader instance.
   * @param text The input text to be proofread.
   * @return The proofread text and the corrections.
   */
  proofread(proofreaderPtr: number, text: string): TextProofreaderResult;

  /**
   * Proofreads the given text and reports the results incrementally to
   * `_userProofreaderListener`.
   * @param proofreaderPtr A pointer to the TextProofreader instance.
   * @param text The input text to be proofread.
   */
  proofreadStreaming(proofreaderPtr: number, text: string): void;
}

/**
 * MediaPipe Task that performs text proofreading.
 */
export class TextProofreader extends TaskRunner {
  protected override baseOptions: BaseOptionsProto = new BaseOptionsProto();
  private maxNumTokens = 0;
  private proofreaderPtr = 0;
  private loggerTimestamp = 0;

  /**
   * Initializes the Wasm runtime and creates a new text proofreader based on
   * the provided options.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param textProofreaderOptions The options for TextProofreader.
   */
  static createFromOptions(
    wasmFileset: WasmFileset,
    textProofreaderOptions: TextProofreaderOptions,
  ): Promise<TextProofreader> {
    return TaskRunner.createInstance(
      TextProofreader,
      /* canvas= */ null,
      wasmFileset,
      textProofreaderOptions,
    );
  }

  /**
   * Initializes the Wasm runtime and creates a new text proofreader based on
   * the path to the model asset.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   * @param maxNumTokens The maximum number of tokens.
   */
  static createFromModelPath(
    wasmFileset: WasmFileset,
    modelAssetPath: string,
    maxNumTokens = 0,
  ): Promise<TextProofreader> {
    return TextProofreader.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetPath,
      },
      maxNumTokens,
    });
  }

  /**
   * Initializes the Wasm runtime and creates a new text proofreader based on
   * the provided model asset buffer.
   * @export
   * @param wasmFileset A configuration object that provides the location of the
   *     Wasm binary and its loader.
   * @param modelAssetBuffer An array or a stream containing a binary
   *    representation of the model.
   * @param maxNumTokens The maximum number of tokens.
   */
  static createFromModelBuffer(
    wasmFileset: WasmFileset,
    modelAssetBuffer: Uint8Array | ReadableStreamDefaultReader,
    maxNumTokens = 0,
  ): Promise<TextProofreader> {
    return TextProofreader.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetBuffer,
      },
      maxNumTokens,
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
    return 'TextProofreader';
  }

  /**
   * Sets the initial options for the text proofreader.
   * LiteRT LM Tasks do not support updating options after initialization.
   * @param options The options for the text proofreader.
   */
  override async setOptions(options: TextProofreaderOptions): Promise<void> {
    if (this.proofreaderPtr !== 0) {
      throw new Error(
        'LiteRT LM Tasks do not support updating options after initialization.',
      );
    }
    if (options.maxNumTokens !== undefined) {
      this.maxNumTokens = options.maxNumTokens;
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
      .wasmModule as unknown as TextProofreaderWasmModule;
    const modelPath = '/model.litertlm';

    this.proofreaderPtr = wasmModule.createTextProofreader(
      modelPath,
      this.maxNumTokens,
    );
    this.logger!.logSessionStart();
  }

  /**
   * Proofreads the given text.
   * @param text The input text to be proofread.
   * @return The proofread text and the corrections.
   */
  proofread(text: string): TextProofreaderResult {
    const wasmModule = this.graphRunner
      .wasmModule as unknown as TextProofreaderWasmModule;
    if (this.proofreaderPtr === 0) {
      throw new Error('TextProofreader is already closed.');
    }
    const timestamp = this.loggerTimestamp++;
    this.logger!.recordCpuInputArrival(timestamp);
    try {
      return wasmModule.proofread(this.proofreaderPtr, text);
    } finally {
      this.logger!.recordInvocationEnd(timestamp);
    }
  }

  /**
   * Proofreads the given text and invokes `progressListener` with each newly
   * generated part of the proofread text.
   *
   * Only one proofread can be in flight at a time. As the underlying Wasm
   * engine runs single-threaded, the calling browser thread remains blocked
   * for the duration of this inference call, and the listener is invoked
   * synchronously. Run the task in a Web Worker to keep the UI thread
   * responsive.
   *
   * @export
   * @param text The input text to be proofread.
   * @param progressListener An optional listener that receives each newly
   *     generated part of the proofread text.
   * @return A promise that resolves to the complete proofread text and the
   *     corrections.
   */
  proofreadStreaming(
    text: string,
    progressListener?: ProofreaderProgressListener,
  ): Promise<TextProofreaderResult> {
    const wasmModule = this.graphRunner
      .wasmModule as unknown as TextProofreaderWasmModule;
    if (this.proofreaderPtr === 0) {
      throw new Error('TextProofreader is already closed.');
    }
    if (wasmModule._userProofreaderListener) {
      throw new Error('Another proofread is already in progress.');
    }

    const timestamp = this.loggerTimestamp++;
    this.logger!.recordCpuInputArrival(timestamp);

    let resolvePromise!: (result: TextProofreaderResult) => void;
    let rejectPromise!: (reason: unknown) => void;
    const promise = new Promise<TextProofreaderResult>((resolve, reject) => {
      resolvePromise = resolve;
      rejectPromise = reject;
    });

    const finishInvocation = () => {
      wasmModule._userProofreaderListener = undefined;
      this.logger!.recordInvocationEnd(timestamp);
    };

    // Install the listener in the same synchronous block as the check above so
    // that no other caller can start a proofread in between.
    let proofreadText = '';
    wasmModule._userProofreaderListener = (
      partialResult: string,
      done: boolean,
      errorMessage: string,
      corrections?: Correction[],
    ) => {
      if (errorMessage) {
        finishInvocation();
        rejectPromise(new Error(errorMessage));
        return;
      }
      proofreadText += partialResult;
      if (progressListener) {
        try {
          progressListener(partialResult, done);
        } catch (e) {
          // Never let an exception from user code unwind into the Wasm
          // stack, as that would leave the engine in an undefined state.
          finishInvocation();
          rejectPromise(e);
          return;
        }
      }
      if (done) {
        finishInvocation();
        resolvePromise({proofreadText, corrections: corrections ?? []});
      }
    };

    try {
      wasmModule.proofreadStreaming(this.proofreaderPtr, text);
    } catch (e) {
      finishInvocation();
      rejectPromise(e);
    }
    return promise;
  }

  /**
   * Closes the TextProofreader and frees resources.
   */
  override close(): void {
    const wasmModule = this.graphRunner
      .wasmModule as unknown as TextProofreaderWasmModule;
    if (this.proofreaderPtr !== 0) {
      try {
        wasmModule.deleteTextProofreader(this.proofreaderPtr);
      } catch {}
      this.proofreaderPtr = 0;
    }
    super.close();
  }
}


