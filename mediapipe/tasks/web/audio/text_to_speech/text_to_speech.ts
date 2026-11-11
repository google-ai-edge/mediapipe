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
  SynthesisOptions,
  TextToSpeechOptions,
} from './text_to_speech_options';
import type {
  TextToSpeechAudioChunk,
  TextToSpeechResult,
  VoiceInfo,
  WordTimestamp,
} from './text_to_speech_result';

export type {
  SynthesisOptions,
  TextToSpeechAudioChunk,
  TextToSpeechOptions,
  TextToSpeechResult,
  VoiceInfo,
  WordTimestamp,
};

/**
 * `delegate` value passed to `createTextToSpeech` to select `MP_DELEGATE_CPU`.
 */
const CPU_DELEGATE_TYPE = -1;

/**
 * `delegate` value passed to `createTextToSpeech` to select `MP_DELEGATE_GPU`.
 */
const GPU_DELEGATE_TYPE = 0;

/**
 * The Wasm module for TextToSpeech with custom C++ Embind bindings.
 */
declare interface TextToSpeechWasmModule {
  createTextToSpeech(
    modelPath: string,
    delegate: number,
    defaultVoiceId: string,
    defaultLocale: string,
    speakingRate: number,
    targetBucket: number,
  ): Promise<number>;

  deleteTextToSpeech(ttsPtr: number): void;

  listTextToSpeechVoices(ttsPtr: number): Promise<VoiceInfo[]>;

  synthesizeTextToSpeech(
    ttsPtr: number,
    text: string,
    voiceId: string,
    locale: string,
  ): Promise<TextToSpeechResult>;

  startTextToSpeechStream(
    ttsPtr: number,
    initialText: string,
    voiceId: string,
    locale: string,
    finishInput: boolean,
  ): Promise<number>;

  /**
   * Resolves to the next chunk, to `null` at end-of-stream, or to `undefined`
   * when the stream needs more input before it can produce audio.
   */
  readNextTextToSpeechChunk(
    streamPtr: number,
  ): Promise<TextToSpeechAudioChunk | null | undefined>;

  pushTextToSpeechStream(
    streamPtr: number,
    textDelta: string,
  ): Promise<boolean>;

  finishTextToSpeechStreamInput(streamPtr: number): Promise<boolean>;

  cancelTextToSpeechStream(streamPtr: number): void;

  deleteTextToSpeechStream(streamPtr: number): void;

  cancelTextToSpeech(ttsPtr: number): void;
}

/**
 * Active streaming synthesis session returned by
 * `TextToSpeech.synthesizeStream()`. Supports both pull-based `readNext()`
 * and `for await (const chunk of stream)` iteration, as well as incremental
 * text input via `pushText()` and `finishInput()`.
 */
export class TextToSpeechStream
  implements AsyncIterable<TextToSpeechAudioChunk>
{
  private isClosed = false;
  /** Incremented whenever new input arrives or the stream state changes. */
  private inputVersion = 0;
  private inputWaiters: Array<() => void> = [];

  /** @hideconstructor */
  constructor(
    private streamPtr: number,
    private readonly wasmModule: TextToSpeechWasmModule,
    // Both callbacks run `fn` after all queued Wasm calls of the owning
    // `TextToSpeech`: a JSPI call can suspend inside Wasm, so other calls must
    // not re-enter the module. Only the first logs a telemetry invocation.
    private readonly queueLoggedWasmCall: <T>(
      fn: () => Promise<T>,
    ) => Promise<T>,
    private readonly queueWasmCall: <T>(fn: () => Promise<T>) => Promise<T>,
    private inputFinished = true,
    private readonly onClose?: () => void,
  ) {}

  /**
   * Pulls the next synthesized PCM audio chunk, or returns `null` when the
   * stream reaches end-of-stream. On a stream created with
   * `finishInput = false`, waits for more text from `pushText()` or for
   * `finishInput()` when the pushed text doesn't yet form a complete clause.
   *
   * @export
   */
  async readNext(): Promise<TextToSpeechAudioChunk | null> {
    while (true) {
      if (this.isClosed || this.streamPtr === 0) {
        return null;
      }
      const version = this.inputVersion;
      const chunk = await this.queueLoggedWasmCall(() => {
        if (this.isClosed || this.streamPtr === 0) {
          return Promise.resolve(null);
        }
        return this.wasmModule.readNextTextToSpeechChunk(this.streamPtr);
      });
      if (chunk !== undefined) {
        return chunk;
      }
      // The stream needs more input. The single-threaded Wasm runtime can't
      // block for it, so wait here until the caller pushes text, finishes the
      // input, or cancels or closes the stream.
      if (this.inputFinished && version === this.inputVersion) {
        return null; // Shouldn't happen; avoids waiting forever.
      }
      if (version === this.inputVersion) {
        await new Promise<void>((resolve) => {
          this.inputWaiters.push(resolve);
        });
      }
    }
  }

  /** Wakes `readNext()` calls that wait for input so that they read again. */
  private notifyInput(): void {
    ++this.inputVersion;
    const waiters = this.inputWaiters;
    this.inputWaiters = [];
    for (const resolve of waiters) {
      resolve();
    }
  }

  /**
   * Appends an incremental text fragment (e.g. from an LLM token stream) to an
   * open stream created with `finishInput = false`.
   *
   * @export
   * @param textDelta The incremental text delta to append.
   */
  async pushText(textDelta: string): Promise<void> {
    if (this.isClosed || this.streamPtr === 0) {
      throw new Error('TextToSpeechStream is already closed.');
    }
    const streamPtr = this.streamPtr;
    await this.queueWasmCall(() =>
      this.wasmModule.pushTextToSpeechStream(streamPtr, textDelta),
    );
    this.notifyInput();
  }

  /**
   * Marks the end of incremental text input on this stream, flushing any
   * remaining partial sentence into the synthesis pipeline.
   *
   * @export
   */
  async finishInput(): Promise<void> {
    if (this.isClosed || this.streamPtr === 0) {
      throw new Error('TextToSpeechStream is already closed.');
    }
    const streamPtr = this.streamPtr;
    await this.queueWasmCall(() =>
      this.wasmModule.finishTextToSpeechStreamInput(streamPtr),
    );
    this.inputFinished = true;
    this.notifyInput();
  }

  /**
   * Cancels this streaming synthesis session immediately.
   *
   * @export
   */
  cancel(): void {
    if (!this.isClosed && this.streamPtr !== 0) {
      this.wasmModule.cancelTextToSpeechStream(this.streamPtr);
    }
    this.notifyInput();
  }

  /**
   * Releases native stream resources.
   *
   * @export
   */
  close(): void {
    if (this.isClosed) return;
    this.isClosed = true;
    if (this.streamPtr !== 0) {
      try {
        this.wasmModule.deleteTextToSpeechStream(this.streamPtr);
      } catch {}
      this.streamPtr = 0;
    }
    this.notifyInput();
    this.onClose?.();
  }

  async *[Symbol.asyncIterator](): AsyncIterator<TextToSpeechAudioChunk> {
    try {
      while (true) {
        const chunk = await this.readNext();
        if (!chunk) {
          break;
        }
        yield chunk;
      }
    } finally {
      this.close();
    }
  }
}

const recycledWasmModules: WasmModule[] = [];

/**
 * MediaPipe Audio Task that performs neural text-to-speech synthesis in unary
 * and streaming modes via WASM SIMD or WebGPU.
 */
export class TextToSpeech extends TaskRunner {
  protected override baseOptions: BaseOptionsProto = new BaseOptionsProto();
  private delegateType = CPU_DELEGATE_TYPE;
  private defaultVoiceId = '';
  private defaultLocale = '';
  private speakingRate = 1.0;
  private targetBucket = 0;
  private readonly openStreams = new Set<TextToSpeechStream>();
  private ttsPtr = 0;
  private isClosed = false;
  private loggerTimestamp = 0;
  private invocationQueue: Promise<unknown> = Promise.resolve();

  /**
   * Initializes the Wasm runtime and creates a new `TextToSpeech` instance
   * based on the provided options.
   *
   * @export
   * @param wasmFileset A configuration object that provides the location of
   *     the Wasm binary and its loader.
   * @param options The options for TextToSpeech. Note that either a path to the
   *     model or the model buffer itself must be provided via `baseOptions`.
   */
  static async createFromOptions(
    wasmFileset: WasmFileset,
    options: TextToSpeechOptions,
  ): Promise<TextToSpeech> {
    const recycledWasm = recycledWasmModules.pop();
    if (recycledWasm) {
      const instance = new TextToSpeech(recycledWasm, null);
      instance.enableLogging(options);
      try {
        await instance.setOptions(options);
        return instance;
      } catch (e) {
        instance.close();
        throw e;
      }
    }
    return TaskRunner.createInstance(
      TextToSpeech,
      /* canvas= */ null,
      wasmFileset,
      options,
    );
  }

  /**
   * Initializes the Wasm runtime and creates a new `TextToSpeech` instance
   * from a model asset path.
   *
   * @export
   * @param wasmFileset A configuration object that provides the location of
   *     the Wasm binary and its loader.
   * @param modelAssetPath The path or URL to the `.litertlm` model asset.
   */
  static createFromModelPath(
    wasmFileset: WasmFileset,
    modelAssetPath: string | string,
  ): Promise<TextToSpeech> {
    return TextToSpeech.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetPath: modelAssetPath as string,
      },
    });
  }

  /**
   * Initializes the Wasm runtime and creates a new `TextToSpeech` instance
   * from a model asset buffer or stream.
   *
   * @export
   * @param wasmFileset A configuration object that provides the location of
   *     the Wasm binary and its loader.
   * @param modelAssetBuffer A `Uint8Array` or `ReadableStreamDefaultReader`
   *     containing the `.litertlm` model bytes.
   */
  static createFromModelBuffer(
    wasmFileset: WasmFileset,
    modelAssetBuffer: Uint8Array | ReadableStreamDefaultReader,
  ): Promise<TextToSpeech> {
    return TextToSpeech.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetBuffer,
      },
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
    return 'TextToSpeech';
  }

  /**
   * Runs `run` after all queued Wasm calls, so that calls never overlap: a JSPI
   * call can suspend inside Wasm, and other calls must not re-enter the module.
   */
  private queueWasmCall<T>(run: () => Promise<T>): Promise<T> {
    const next = this.invocationQueue.then(run);
    this.invocationQueue = next.then(
      () => {},
      () => {},
    );
    return next;
  }

  /** Like `queueWasmCall()`, and also logs a telemetry invocation. */
  private logInvocation<T>(run: () => Promise<T>): Promise<T> {
    return this.queueWasmCall(() => {
      const timestamp = this.loggerTimestamp++;
      if (this.logger) {
        if (this.delegateType === CPU_DELEGATE_TYPE) {
          this.logger.recordCpuInputArrival(timestamp);
        } else {
          this.logger.recordGpuInputArrival(timestamp);
        }
      }
      return Promise.resolve()
        .then(run)
        .then(
          (result) => {
            this.logger?.recordInvocationEnd(timestamp);
            return result;
          },
          (error: unknown) => {
            this.logger?.recordInvocationEnd(timestamp);
            throw error;
          },
        );
    });
  }

  /**
   * Sets the options for `TextToSpeech` and initializes the underlying engine.
   * Options cannot be updated once the engine has been created.
   *
   * @export
   * @param options The options for TextToSpeech.
   */
  override async setOptions(options: TextToSpeechOptions): Promise<void> {
    if (this.ttsPtr !== 0) {
      throw new Error(
        'TextToSpeech does not support updating options after initialization.',
      );
    }
    if (options.baseOptions?.delegate === 'GPU') {
      this.delegateType = GPU_DELEGATE_TYPE;
    } else {
      this.delegateType = CPU_DELEGATE_TYPE;
    }
    if (options.defaultVoiceId !== undefined) {
      this.defaultVoiceId = options.defaultVoiceId;
    }
    if (options.defaultLocale !== undefined) {
      this.defaultLocale = options.defaultLocale;
    }
    if (options.speakingRate !== undefined) {
      this.speakingRate = options.speakingRate;
    }
    if (options.targetBucket !== undefined) {
      this.targetBucket = options.targetBucket;
    }

    if (this.delegateType === CPU_DELEGATE_TYPE) {
      const wasmModule = this.graphRunner.wasmModule as unknown as Record<
        string,
        unknown
      >;
      wasmModule['preinitializedWebGPUDevice'] = undefined;
      const globalScope = (typeof self !== 'undefined'
        ? self
        : globalThis) as unknown as Record<string, unknown>;
      if (typeof globalScope['Module'] === 'object' && globalScope['Module']) {
        (globalScope['Module'] as Record<string, unknown>)[
          'preinitializedWebGPUDevice'
        ] = undefined;
      }
    } else if (typeof navigator !== 'undefined' && navigator.gpu) {
      try {
        const wasmModule = this.graphRunner.wasmModule as unknown as Record<
          string,
          unknown
        >;
        if (!wasmModule['preinitializedWebGPUDevice']) {
          const adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance',
          });
          const adapterInfo = (
            adapter as unknown as {
              info?: {
                vendor?: string;
                architecture?: string;
                description?: string;
              };
            }
          )?.info;
          const isFallback = Boolean(
            (adapter as unknown as {isFallbackAdapter?: boolean})
              ?.isFallbackAdapter,
          );
          const isSoftwareAdapter =
            isFallback ||
            /swiftshader|llvmpipe|software|lavapipe/i.test(
              `${adapterInfo?.vendor ?? ''} ${adapterInfo?.architecture ?? ''} ${adapterInfo?.description ?? ''}`,
            );
          if (adapter && !isSoftwareAdapter) {
            const requiredFeatures: GPUFeatureName[] = [];
            for (const feat of [
              'shader-f16',
              'subgroups',
            ] as GPUFeatureName[]) {
              if (adapter.features.has(feat)) {
                requiredFeatures.push(feat);
              }
            }
            const requiredLimits: Record<string, number> = {};
            for (const limitKey of [
              'maxBufferSize',
              'maxStorageBufferBindingSize',
              'maxStorageBuffersPerShaderStage',
              'maxComputeWorkgroupStorageSize',
              'maxTextureDimension2D',
            ]) {
              const val = (
                adapter.limits as unknown as Record<string, number | undefined>
              )[limitKey];
              if (typeof val === 'number') {
                requiredLimits[limitKey] = val;
              }
            }
            const device = await adapter.requestDevice({
              requiredFeatures,
              requiredLimits,
            });
            const deviceWithInfo = device as unknown as {adapterInfo?: unknown};
            if (!deviceWithInfo.adapterInfo && adapterInfo) {
              try {
                Object.defineProperty(device, 'adapterInfo', {
                  value: adapterInfo,
                  writable: true,
                  configurable: true,
                });
              } catch {
                try {
                  deviceWithInfo.adapterInfo = adapterInfo;
                } catch {}
              }
            }
            wasmModule['preinitializedWebGPUDevice'] = device;
          }
        }
        if (wasmModule['preinitializedWebGPUDevice']) {
          const globalScope = (typeof self !== 'undefined'
            ? self
            : globalThis) as unknown as Record<string, unknown>;
          if (
            typeof globalScope['Module'] === 'object' &&
            globalScope['Module']
          ) {
            (globalScope['Module'] as Record<string, unknown>)[
              'preinitializedWebGPUDevice'
            ] = wasmModule['preinitializedWebGPUDevice'];
          }
        }
      } catch (e) {
        console.warn('WebGPU device initialization failed:', e);
      }
    }

    const wasmModule = this.graphRunner
      .wasmModule as unknown as TextToSpeechWasmModule;

    const loadModel =
      !!options.baseOptions &&
      (!!options.baseOptions.modelAssetPath ||
        !!options.baseOptions.modelAssetBuffer);

    // The native TextToSpeech only loads models from a file path, so the
    // model is written to the Wasm virtual filesystem (as for TextSummarizer
    // and UniversalEmbedder) rather than passed as an in-memory buffer.
    await this.applyOptions(
      options,
      /* loadTfliteModel= */ loadModel,
      /* isLiteRtLmModel= */ true,
    );

    const modelPath = '/model.litertlm';
    this.ttsPtr = await wasmModule.createTextToSpeech(
      modelPath,
      this.delegateType,
      this.defaultVoiceId,
      this.defaultLocale,
      this.speakingRate,
      this.targetBucket,
    );
    this.logger?.logSessionStart();
  }

  protected override refreshGraph(): void {
    // Initialization is performed asynchronously in setOptions().
  }

  /**
   * Queries available voices bundled in the initialized model.
   *
   * @export
   */
  listVoices(): Promise<VoiceInfo[]> {
    if (this.ttsPtr === 0) {
      throw new Error('TextToSpeech is already closed.');
    }
    const wasmModule = this.graphRunner
      .wasmModule as unknown as TextToSpeechWasmModule;
    return this.logInvocation(() =>
      wasmModule.listTextToSpeechVoices(this.ttsPtr),
    );
  }

  /**
   * Synthesizes a complete utterance and returns the full PCM audio buffer.
   *
   * @export
   * @param text The text to synthesize.
   * @param options Optional per-utterance voice or locale overrides.
   */
  synthesize(
    text: string,
    options?: SynthesisOptions,
  ): Promise<TextToSpeechResult> {
    if (this.ttsPtr === 0) {
      throw new Error('TextToSpeech is already closed.');
    }
    const wasmModule = this.graphRunner
      .wasmModule as unknown as TextToSpeechWasmModule;
    return this.logInvocation(() =>
      wasmModule.synthesizeTextToSpeech(
        this.ttsPtr,
        text,
        options?.voiceId ?? '',
        options?.locale ?? '',
      ),
    );
  }

  /**
   * Starts a streaming synthesis session returning a `TextToSpeechStream`
   * (supporting both `readNext()` and `for await (const chunk of stream)`).
   *
   * @export
   * @param initialText Initial text to synthesize (pass `""` when streaming
   *     incremental LLM text deltas via `stream.pushText()`).
   * @param options Optional per-utterance voice or locale overrides.
   * @param finishInput Pass `true` (default) for single-utterance streaming, or
   *     `false` to keep the input stream open for `stream.pushText()` and
   *     `stream.finishInput()`.
   */
  async synthesizeStream(
    initialText = '',
    options?: SynthesisOptions,
    finishInput = true,
  ): Promise<TextToSpeechStream> {
    if (this.ttsPtr === 0) {
      throw new Error('TextToSpeech is already closed.');
    }
    const wasmModule = this.graphRunner
      .wasmModule as unknown as TextToSpeechWasmModule;
    const streamPtr = await this.logInvocation(() =>
      wasmModule.startTextToSpeechStream(
        this.ttsPtr,
        initialText,
        options?.voiceId ?? '',
        options?.locale ?? '',
        finishInput,
      ),
    );
    const stream: TextToSpeechStream = new TextToSpeechStream(
      streamPtr,
      wasmModule,
      (fn) => this.logInvocation(fn),
      (fn) => this.queueWasmCall(fn),
      finishInput,
      () => {
        this.openStreams.delete(stream);
      },
    );
    this.openStreams.add(stream);
    return stream;
  }

  /**
   * Opens an incremental streaming synthesis session returning a
   * `TextToSpeechStream` with `finishInput` defaulting to `false` so callers
   * can push LLM text deltas via `stream.pushText()` and finish with
   * `stream.finishInput()`.
   *
   * @export
   * @param initialText Optional initial text fragment to enqueue immediately.
   * @param options Optional per-utterance voice or locale overrides.
   * @param finishInput Pass `false` (default) to keep the input stream open for
   *     `stream.pushText()`, or `true` for a single-utterance stream.
   */
  async openStream(
    initialText = '',
    options?: SynthesisOptions,
    finishInput = false,
  ): Promise<TextToSpeechStream> {
    return this.synthesizeStream(initialText, options, finishInput);
  }

  /**
   * Cancels any in-flight unary or streaming synthesis on this `TextToSpeech`
   * instance.
   *
   * @export
   */
  cancel(): void {
    if (!this.isClosed && this.ttsPtr !== 0) {
      const wasmModule = this.graphRunner
        .wasmModule as unknown as TextToSpeechWasmModule;
      wasmModule.cancelTextToSpeech(this.ttsPtr);
    }
    // Also cancel each open stream, which wakes reads that wait for input so
    // that they observe the cancellation. (The native cancel above already
    // cancelled the streams' synthesis.)
    for (const stream of this.openStreams) {
      stream.cancel();
    }
  }

  /**
   * Closes the `TextToSpeech` instance and releases native engine and Wasm
   * memory.
   *
   * @export
   */
  override close(): void {
    if (this.isClosed) return;
    this.isClosed = true;
    // Close remaining streams first so that pending `readNext()` calls wake up
    // and no stream touches the engine after it is deleted. `stream.close()`
    // removes the stream from `openStreams`, so iterate over a copy.
    for (const stream of Array.from(this.openStreams)) {
      stream.close();
    }
    const rawWasm = this.graphRunner.wasmModule;
    const wasmModule = rawWasm as unknown as TextToSpeechWasmModule;
    if (this.ttsPtr !== 0) {
      try {
        wasmModule.deleteTextToSpeech(this.ttsPtr);
      } catch {}
      this.ttsPtr = 0;
    }
    super.close();
    if (rawWasm && !recycledWasmModules.includes(rawWasm)) {
      recycledWasmModules.push(rawWasm);
    }
  }
}



