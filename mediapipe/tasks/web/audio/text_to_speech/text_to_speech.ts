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
    cacheDirectory: string,
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

  readNextTextToSpeechChunk(
    streamPtr: number,
  ): Promise<TextToSpeechAudioChunk | null>;

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

  /** @hideconstructor */
  constructor(
    private streamPtr: number,
    private readonly wasmModule: TextToSpeechWasmModule,
    private readonly runSerialized: <T>(fn: () => Promise<T>) => Promise<T>,
  ) {}

  /**
   * Pulls the next synthesized PCM audio chunk, or returns `null` when the
   * stream reaches end-of-stream.
   *
   * @export
   */
  async readNext(): Promise<TextToSpeechAudioChunk | null> {
    if (this.isClosed || this.streamPtr === 0) {
      return null;
    }
    return this.runSerialized(() => {
      if (this.isClosed || this.streamPtr === 0) {
        return Promise.resolve(null);
      }
      return this.wasmModule.readNextTextToSpeechChunk(this.streamPtr);
    });
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
    await this.wasmModule.pushTextToSpeechStream(this.streamPtr, textDelta);
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
    await this.wasmModule.finishTextToSpeechStreamInput(this.streamPtr);
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
  private cacheDirectory = '';
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

  private logInvocation<T>(run: () => Promise<T>): Promise<T> {
    const next = this.invocationQueue.then(() => {
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
    this.invocationQueue = next.then(
      () => {},
      () => {},
    );
    return next;
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
    if (options.cacheDirectory !== undefined) {
      this.cacheDirectory = options.cacheDirectory;
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
      this.cacheDirectory,
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
    return new TextToSpeechStream(streamPtr, wasmModule, (fn) =>
      this.logInvocation(fn),
    );
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



