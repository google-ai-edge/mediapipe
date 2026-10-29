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
  AudioTranscriberModelArchitecture,
  AudioTranscriberOptions,
} from './audio_transcriber_options';
import type {
  AudioTranscriberResult,
  TranscriptionSegment,
} from './audio_transcriber_result';

export type {
  AudioTranscriberModelArchitecture,
  AudioTranscriberOptions,
  AudioTranscriberResult,
  TranscriptionSegment,
};

/**
 * `delegate` value passed to `createAudioTranscriber` to select
 * `MP_DELEGATE_CPU`.
 */
const CPU_DELEGATE_TYPE = -1;

/**
 * `delegate` value passed to `createAudioTranscriber` to select
 * `MP_DELEGATE_GPU`.
 */
const GPU_DELEGATE_TYPE = 0;

/**
 * The Wasm module for AudioTranscriber with custom C++ Embind bindings.
 */
declare interface AudioTranscriberWasmModule {
  createAudioTranscriber(
    modelPath: string,
    delegate: number,
    sampleRateHz: number,
    enableInterimResults: boolean,
    modelArchitecture: number,
    prompt: string,
    cacheDir: string,
  ): Promise<number>;

  deleteAudioTranscriber(transcriberPtr: number): void;

  transcribeAudio(
    transcriberPtr: number,
    audioSamples: Float32Array | number[],
    endOfStream: boolean,
  ): Promise<AudioTranscriberResult>;

  finishAudioTranscriberStream(
    transcriberPtr: number,
  ): Promise<AudioTranscriberResult>;

  prewarmAudioTranscriber(transcriberPtr: number): Promise<boolean>;
}

function parseModelArchitecture(
  arch?: AudioTranscriberModelArchitecture | number,
): number {
  if (typeof arch === 'number') {
    return arch;
  }
  switch (arch) {
    case 'OMNI':
      return 1;
    case 'TINY_GEMMA':
      return 2;
    case 'AUTO':
    case undefined:
      return 0;
    default:
      throw new Error(`Unsupported model architecture: ${arch}`);
  }
}

const recycledWasmModules: WasmModule[] = [];

/**
 * MediaPipe Audio Task that performs automatic speech recognition (ASR) in
 * unary and streaming modes via WASM SIMD or WebGPU.
 */
export class AudioTranscriber extends TaskRunner {
  protected override baseOptions: BaseOptionsProto = new BaseOptionsProto();
  private delegateType = GPU_DELEGATE_TYPE;
  private sampleRateHz = 16000;
  private enableInterimResults = false;
  private modelArchitecture = 0;
  private prompt = '';
  private cacheDir = '';
  private transcriberPtr = 0;
  private isClosed = false;
  private loggerTimestamp = 0;
  private invocationQueue: Promise<unknown> = Promise.resolve();

  /**
   * Initializes the Wasm runtime and creates a new `AudioTranscriber` based on
   * the provided options.
   *
   * @export
   * @param wasmFileset A configuration object that provides the location of
   *     the Wasm binary and its loader.
   * @param options The options for the AudioTranscriber. Note that either a
   *     path to the model or the model buffer itself must be provided via
   *     `baseOptions`.
   */
  static async createFromOptions(
    wasmFileset: WasmFileset,
    options: AudioTranscriberOptions,
  ): Promise<AudioTranscriber> {
    const recycledWasm = recycledWasmModules.pop();
    if (recycledWasm) {
      const instance = new AudioTranscriber(recycledWasm, null);
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
      AudioTranscriber,
      /* canvas= */ null,
      wasmFileset,
      options,
    );
  }

  /**
   * Initializes the Wasm runtime and creates a new `AudioTranscriber` from a
   * model asset path.
   *
   * @export
   * @param wasmFileset A configuration object that provides the location of
   *     the Wasm binary and its loader.
   * @param modelAssetPath The path or URL to the `.litertlm` model asset.
   */
  static createFromModelPath(
    wasmFileset: WasmFileset,
    modelAssetPath: string | string,
  ): Promise<AudioTranscriber> {
    return AudioTranscriber.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetPath: modelAssetPath as string,
      },
    });
  }

  /**
   * Initializes the Wasm runtime and creates a new `AudioTranscriber` from a
   * model asset buffer or stream.
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
  ): Promise<AudioTranscriber> {
    return AudioTranscriber.createFromOptions(wasmFileset, {
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
    return 'AudioTranscriber';
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
   * Sets the options for the `AudioTranscriber` and initializes the underlying
   * engine. Options cannot be updated once the engine has been created.
   *
   * @export
   * @param options The options for the AudioTranscriber.
   */
  override async setOptions(options: AudioTranscriberOptions): Promise<void> {
    if (this.transcriberPtr !== 0) {
      throw new Error(
        'AudioTranscriber does not support updating options after initialization.',
      );
    }
    if (options.baseOptions?.delegate === 'CPU') {
      this.delegateType = CPU_DELEGATE_TYPE;
    } else {
      this.delegateType = GPU_DELEGATE_TYPE;
    }
    if (options.sampleRateHz !== undefined) {
      this.sampleRateHz = options.sampleRateHz;
    }
    if (options.enableInterimResults !== undefined) {
      this.enableInterimResults = options.enableInterimResults;
    }
    if (options.modelArchitecture !== undefined) {
      this.modelArchitecture = parseModelArchitecture(
        options.modelArchitecture,
      );
    }
    if (options.prompt !== undefined) {
      this.prompt = options.prompt;
    }
    if (options.cacheDir !== undefined) {
      this.cacheDir = options.cacheDir;
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
      .wasmModule as unknown as AudioTranscriberWasmModule;

    const loadModel =
      !!options.baseOptions &&
      (!!options.baseOptions.modelAssetPath ||
        !!options.baseOptions.modelAssetBuffer);

    // The native AudioTranscriber only loads models from a file path, so the
    // model is written to the Wasm virtual filesystem (as for TextSummarizer
    // and UniversalEmbedder) rather than passed as an in-memory buffer.
    await this.applyOptions(
      options,
      /* loadTfliteModel= */ loadModel,
      /* isLiteRtLmModel= */ true,
    );

    const modelPath = '/model.litertlm';
    this.transcriberPtr = await wasmModule.createAudioTranscriber(
      modelPath,
      this.delegateType,
      this.sampleRateHz,
      this.enableInterimResults,
      this.modelArchitecture,
      this.prompt,
      this.cacheDir,
    );
    this.logger?.logSessionStart();
  }

  protected override refreshGraph(): void {
    // Initialization is performed asynchronously in setOptions().
  }

  /**
   * Transcribes mono PCM float audio samples in `[-1.0, 1.0]`.
   *
   * @export
   * @param audioSamples Mono PCM float audio samples at `sampleRateHz`.
   * @param endOfStream Whether this chunk marks the end of the audio stream.
   *     Defaults to `true` (unary single-utterance mode).
   * @return The transcription result containing segments and concatenated text.
   */
  transcribe(
    audioSamples: Float32Array | number[],
    endOfStream = true,
  ): Promise<AudioTranscriberResult> {
    if (this.transcriberPtr === 0) {
      throw new Error('AudioTranscriber is already closed.');
    }
    const wasmModule = this.graphRunner
      .wasmModule as unknown as AudioTranscriberWasmModule;
    return this.logInvocation(() =>
      wasmModule.transcribeAudio(
        this.transcriberPtr,
        audioSamples,
        endOfStream,
      ),
    );
  }

  /**
   * Feeds a streaming chunk of mono PCM float audio samples (`endOfStream = false`)
   * and returns any newly emitted interim or final segments.
   *
   * @export
   * @param audioSamples Mono PCM float audio samples at `sampleRateHz`.
   * @return Newly emitted segments from the streaming session.
   */
  transcribeStream(
    audioSamples: Float32Array | number[],
  ): Promise<AudioTranscriberResult> {
    return this.transcribe(audioSamples, /* endOfStream= */ false);
  }

  /**
   * Flushes any remaining buffered audio and finalizes the current stream.
   *
   * @export
   * @return Finalized transcription segments.
   */
  finishStream(): Promise<AudioTranscriberResult> {
    if (this.transcriberPtr === 0) {
      throw new Error('AudioTranscriber is already closed.');
    }
    const wasmModule = this.graphRunner
      .wasmModule as unknown as AudioTranscriberWasmModule;
    return this.logInvocation(() =>
      wasmModule.finishAudioTranscriberStream(this.transcriberPtr),
    );
  }

  /**
   * Prewarms the underlying inference engine to reduce first-chunk latency.
   *
   * @export
   */
  async prewarm(): Promise<void> {
    if (this.transcriberPtr === 0) {
      throw new Error('AudioTranscriber is already closed.');
    }
    const wasmModule = this.graphRunner
      .wasmModule as unknown as AudioTranscriberWasmModule;
    await this.logInvocation(() =>
      wasmModule.prewarmAudioTranscriber(this.transcriberPtr),
    );
  }

  /**
   * Closes the `AudioTranscriber` and releases native engine and Wasm memory.
   *
   * @export
   */
  override close(): void {
    if (this.isClosed) return;
    this.isClosed = true;
    const rawWasm = this.graphRunner.wasmModule;
    const wasmModule = rawWasm as unknown as AudioTranscriberWasmModule;
    if (this.transcriberPtr !== 0) {
      try {
        wasmModule.deleteAudioTranscriber(this.transcriberPtr);
      } catch {}
      this.transcriberPtr = 0;
    }
    super.close();
    if (rawWasm && !recycledWasmModules.includes(rawWasm)) {
      recycledWasmModules.push(rawWasm);
    }
  }
}


