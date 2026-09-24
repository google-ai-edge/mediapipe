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

import {Embedding} from '../../../../tasks/web/components/containers/embedding_result';
import {computeCosineSimilarity} from '../../../../tasks/web/components/utils/cosine_similarity';
import {WasmFileset} from '../../../../tasks/web/core/wasm_fileset';
import {streamToUint8Array} from '../../../../tasks/web/genai/llm_inference/model_loading_utils';
import {
  createMediaPipeLib,
  FileLocator,
  WasmModule,
} from '../../../../web/graph_runner/graph_runner';
// Placeholder for internal dependency on trusted resource url

import type {
  ActivationDataType,
  ContentPart,
  EmbeddingProvider,
  UniversalEmbedderOptions,
} from './universal_embedder_options';
import type {UniversalEmbedderResult} from './universal_embedder_result';

export type {
  ActivationDataType,
  AudioPart,
  ContentPart,
  EmbeddingProvider,
  ImagePart,
  TextPart,
  UniversalEmbedderOptions,
} from './universal_embedder_options';
export type {
  Embedding,
  UniversalEmbedderResult,
} from './universal_embedder_result';

/**
 * Maps the user-provided ActivationDataType to the corresponding numeric code
 * expected by the C++ UniversalEmbedder API. Returns 0 if undefined.
 */
function parseActivationDataType(type?: ActivationDataType): number {
  if (type === undefined) {
    return 0;
  }
  switch (type) {
    case 'FLOAT32':
      return 1;
    case 'FLOAT16':
      return 2;
    case 'INT16':
      return 3;
    case 'INT8':
      return 4;
    default:
      throw new Error(`Unsupported activation data type: ${type}`);
  }
}

/**
 * Declarations for the native UniversalEmbedder WebAssembly module.
 *
 * Using `declare interface` instructs Tsickle to generate Closure Compiler
 * externs, preventing properties from being renamed by ADVANCED_OPTIMIZATIONS.
 */
// Exposes raw WebAssembly symbol exports that match C++ EMSCRIPTEN_BINDINGS.
// tslint:disable:name-casing
/* eslint-disable @typescript-eslint/naming-convention */
export declare interface UniversalEmbedderWasmModule extends WasmModule {
  preinitializedWebGPUDevice?: GPUDevice;
  createUniversalEmbedder(
    modelPath: string,
    l2Normalize: boolean,
    useGpuBackend: boolean,
    maxInputLength: number,
    visionTokensPerImage: number,
    activationDataType: number,
  ): number;
  universalEmbedder_embedText(handle: number, text: string): Float32Array;
  universalEmbedder_embedImage(
    handle: number,
    imageBytes: Uint8Array | string,
  ): Float32Array;
  universalEmbedder_embedAudio(
    handle: number,
    audioSamples: number[] | Float32Array,
  ): Float32Array;
  universalEmbedder_createContentBuilder(): number;
  universalEmbedder_builderAddText(builderHandle: number, text: string): void;
  universalEmbedder_builderAddImage(
    builderHandle: number,
    imageBytes: Uint8Array | string,
  ): void;
  universalEmbedder_builderAddAudio(
    builderHandle: number,
    audioSamples: number[] | Float32Array,
  ): void;
  universalEmbedder_executeEmbedContent(
    handle: number,
    builderHandle: number,
  ): Float32Array;
  universalEmbedder_freeContentBuilder(builderHandle: number): void;
  universalEmbedder_close(handle: number): void;
}

/**
 * UniversalEmbedder performs multimodal embedding extraction directly
 * using LiteRT LM.
 */
export class UniversalEmbedder {
  // Tracks number of instances and used to maintain dedicated per instance
  // model files in the virtual file system.
  private static instanceCounter = 0;
  private isClosed = false;
  private nativeHandle = 0;

  /**
   * Initializes the Wasm runtime and creates a new UniversalEmbedder instance.
   * @export
   */
  static async createFromOptions(
    wasmFileset: WasmFileset,
    options: UniversalEmbedderOptions,
  ): Promise<UniversalEmbedder> {
    const isGpu = options.baseOptions.delegate === 'GPU';
    let device = options.baseOptions.device;

    if (isGpu && !device) {
      device = await UniversalEmbedder.createWebGpuDevice();
    }

    const fileLocator: FileLocator = {
      locateFile(file: string): string {
        if (file.endsWith('.wasm')) {
          return wasmFileset.wasmBinaryPath.toString();
        }
        return file;
      },
    };

    const instance = await createMediaPipeLib(
      UniversalEmbedder,
      wasmFileset.wasmLoaderPath,
      wasmFileset.assetLoaderPath,
      /* glCanvas= */ null,
      fileLocator,
    );

    if (device) {
      instance.wasmModule.preinitializedWebGPUDevice = device;
    }

    await instance.initialize(options, device);
    return instance;
  }

  /**
   * Creates a WebGPU device for GPU-accelerated universal multimodal embedding.
   * @export
   */
  static async createWebGpuDevice(): Promise<GPUDevice> {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      throw new Error('WebGPU is not supported on this platform.');
    }
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
    if (!adapter) {
      throw new Error('No appropriate WebGPU adapter found.');
    }
    const requiredFeatures: GPUFeatureName[] = [];
    if (adapter.features.has('subgroups' as GPUFeatureName)) {
      requiredFeatures.push('subgroups' as GPUFeatureName);
    }
    if (adapter.features.has('shader-f16' as GPUFeatureName)) {
      requiredFeatures.push('shader-f16' as GPUFeatureName);
    }
    const device = await adapter.requestDevice({
      requiredFeatures,
    });
    // Emscripten / LiteRT WebGPU backends read device.adapterInfo for shader
    // tuning. Populate it from adapter.info if missing on this platform.
    if (!device.adapterInfo && adapter.info) {
      try {
        Object.defineProperty(device, 'adapterInfo', {
          value: adapter.info,
          writable: true,
          configurable: true,
        });
      } catch {
        // Silently ignore if the property is not configurable or already
        // defined.
      }
    }
    return device;
  }

  /**
   * Initializes the Wasm runtime and creates a UniversalEmbedder from a model buffer.
   * @export
   */
  static createFromModelBuffer(
    wasmFileset: WasmFileset,
    modelAssetBuffer: Uint8Array | ReadableStreamDefaultReader,
  ): Promise<UniversalEmbedder> {
    return UniversalEmbedder.createFromOptions(wasmFileset, {
      baseOptions: {modelAssetBuffer},
    });
  }

  /**
   * Initializes the Wasm runtime and creates a UniversalEmbedder from a model path.
   * @export
   */
  static createFromModelPath(
    wasmFileset: WasmFileset,
    modelAssetPath: string | string,
  ): Promise<UniversalEmbedder> {
    return UniversalEmbedder.createFromOptions(wasmFileset, {
      baseOptions: {modelAssetPath: modelAssetPath.toString()},
    });
  }

  private readonly wasmModule: UniversalEmbedderWasmModule;
  private vfsModelPath?: string;

  /** @hideconstructor */
  constructor(wasmModule: WasmModule) {
    this.wasmModule = wasmModule as UniversalEmbedderWasmModule;
  }

  private async initialize(
    options: UniversalEmbedderOptions,
    injectedDevice?: GPUDevice,
  ): Promise<void> {
    const isGpu = options.baseOptions.delegate === 'GPU';
    const l2Normalize = options.l2Normalize !== false;
    let modelPath = options.baseOptions.modelAssetPath
      ? options.baseOptions.modelAssetPath.toString()
      : '';

    let device =
      injectedDevice ||
      options.baseOptions.device ||
      this.wasmModule.preinitializedWebGPUDevice;
    if (isGpu && !device) {
      device = await UniversalEmbedder.createWebGpuDevice();
    }
    if (device) {
      this.wasmModule.preinitializedWebGPUDevice = device;
    }

    if (options.baseOptions.modelAssetBuffer) {
      const buffer =
        options.baseOptions.modelAssetBuffer instanceof Uint8Array
          ? options.baseOptions.modelAssetBuffer
          : await streamToUint8Array(
              options.baseOptions
                .modelAssetBuffer as ReadableStreamDefaultReader<Uint8Array>,
            );

      const fileName = `universal_embedder_model_${++UniversalEmbedder.instanceCounter}.litertlm`;
      this.vfsModelPath = this.writeVfsFile(fileName, buffer);
      modelPath = this.vfsModelPath;
    }

    if (!modelPath) {
      throw new Error('No model asset path or buffer provided.');
    }

    const maxInputLength = options.maxInputLength ?? 0;
    const visionTokensPerImage = options.visionTokensPerImage ?? 0;
    const activationDataType = parseActivationDataType(
      options.activationDataType,
    );

    try {
      this.nativeHandle = this.wasmModule.createUniversalEmbedder(
        modelPath,
        l2Normalize,
        isGpu,
        maxInputLength,
        visionTokensPerImage,
        activationDataType,
      );
    } catch (e) {
      this.unlinkVfsFile();
      throw e;
    }
  }

  /**
   * Performs embedding extraction on the input text.
   * @export
   */
  async embedText(text: string): Promise<UniversalEmbedderResult> {
    this.ensureNotClosed();
    const floatArray = this.wasmModule.universalEmbedder_embedText(
      this.nativeHandle,
      text,
    );
    return {
      embeddings: [
        {
          floatEmbedding: floatArray ? Array.from(floatArray) : [],
          headIndex: 0,
          headName: 'default',
        },
      ],
    };
  }

  /**
   * Performs embedding extraction on raw image bytes.
   * @export
   */
  async embedImage(imageBytes: Uint8Array): Promise<UniversalEmbedderResult> {
    this.ensureNotClosed();
    const floatArray = this.wasmModule.universalEmbedder_embedImage(
      this.nativeHandle,
      imageBytes,
    );
    return {
      embeddings: [
        {
          floatEmbedding: floatArray ? Array.from(floatArray) : [],
          headIndex: 0,
          headName: 'default',
        },
      ],
    };
  }

  /**
   * Performs embedding extraction on audio float samples.
   * @export
   */
  async embedAudio(audioData: Float32Array): Promise<UniversalEmbedderResult> {
    this.ensureNotClosed();
    const floatArray = this.wasmModule.universalEmbedder_embedAudio(
      this.nativeHandle,
      audioData,
    );
    return {
      embeddings: [
        {
          floatEmbedding: floatArray ? Array.from(floatArray) : [],
          headIndex: 0,
          headName: 'default',
        },
      ],
    };
  }

  /**
   * Performs embedding extraction on multimodal content parts.
   * @export
   */
  async embedContent(
    content: readonly ContentPart[],
  ): Promise<UniversalEmbedderResult> {
    this.ensureNotClosed();
    if (!Array.isArray(content)) {
      throw new Error('Content must be an array of ContentParts.');
    }

    const builderHandle =
      this.wasmModule.universalEmbedder_createContentBuilder();
    try {
      for (const part of content) {
        if ('text' in part && typeof part.text === 'string') {
          this.wasmModule.universalEmbedder_builderAddText(
            builderHandle,
            part.text,
          );
        } else if (
          'imageBytes' in part &&
          (part.imageBytes instanceof Uint8Array ||
            typeof part.imageBytes === 'string')
        ) {
          this.wasmModule.universalEmbedder_builderAddImage(
            builderHandle,
            part.imageBytes,
          );
        } else if (
          'audioData' in part &&
          (part.audioData instanceof Float32Array ||
            Array.isArray(part.audioData))
        ) {
          this.wasmModule.universalEmbedder_builderAddAudio(
            builderHandle,
            part.audioData,
          );
        } else {
          throw new Error(
            `Unsupported or invalid ContentPart: ${JSON.stringify(part)}`,
          );
        }
      }

      const floatArray = this.wasmModule.universalEmbedder_executeEmbedContent(
        this.nativeHandle,
        builderHandle,
      );

      return {
        embeddings: [
          {
            floatEmbedding: floatArray ? Array.from(floatArray) : [],
            headIndex: 0,
            headName: 'default',
          },
        ],
      };
    } finally {
      this.wasmModule.universalEmbedder_freeContentBuilder(builderHandle);
    }
  }

  /**
   * Computes cosine similarity between two embeddings.
   * @export
   */
  static cosineSimilarity(u: Embedding, v: Embedding): number {
    return computeCosineSimilarity(u, v);
  }

  /**
   * Returns an {@link EmbeddingProvider} for this embedder.
   * @export
   */
  getProvider(): EmbeddingProvider {
    return {
      embedContent: async (
        content: readonly ContentPart[],
      ): Promise<Float32Array | number[] | null> => {
        const result = await this.embedContent(content);
        return result?.embeddings?.[0]?.floatEmbedding ?? null;
      },
    };
  }

  /**
   * Shuts down the UniversalEmbedder and releases resources.
   * @export
   */
  close(): void {
    if (!this.isClosed) {
      this.isClosed = true;
      if (this.nativeHandle !== 0) {
        this.wasmModule.universalEmbedder_close(this.nativeHandle);
        this.nativeHandle = 0;
      }
      this.unlinkVfsFile();
    }
  }

  /**
   * Writes a binary model buffer into the Emscripten virtual filesystem (VFS).
   */
  private writeVfsFile(fileName: string, buffer: Uint8Array): string {
    const vfsPath = `/${fileName}`;
    try {
      this.wasmModule.FS_unlink?.(vfsPath);
    } catch {
      // Best-effort cleanup if file existed previously.
    }
    this.wasmModule.FS_createDataFile(
      '/',
      fileName,
      buffer,
      /* canRead= */ true,
      /* canWrite= */ false,
      /* canOwn= */ false,
    );
    return vfsPath;
  }

  /**
   * Unlinks the active model file from the Emscripten virtual filesystem (VFS).
   */
  private unlinkVfsFile(): void {
    if (this.vfsModelPath) {
      try {
        this.wasmModule.FS_unlink?.(this.vfsModelPath);
      } catch {
        // Best-effort VFS cleanup.
      }
      this.vfsModelPath = undefined;
    }
  }

  private ensureNotClosed(): void {
    if (this.isClosed || this.nativeHandle === 0) {
      throw new Error('UniversalEmbedder has already been closed.');
    }
  }
}


