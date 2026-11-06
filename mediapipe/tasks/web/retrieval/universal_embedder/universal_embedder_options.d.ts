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

// Placeholder for internal dependency on trusted resource url

/** Backend an encoder runs on. */
export type UniversalEmbedderDelegate = 'CPU' | 'GPU';

/** Options for configuring the model asset and hardware accelerator. */
export declare interface UniversalEmbedderBaseOptions {
  /** The model asset buffer containing the .litertlm file. */
  modelAssetBuffer?: Uint8Array | ReadableStreamDefaultReader;
  /** The path to the model asset file. */
  modelAssetPath?: string | string;
  /**
   * The default backend for all encoders. Defaults to 'GPU' (WebGPU). Can be
   * overridden per encoder with `textDelegate`, `visionDelegate`, and
   * `audioDelegate` on `UniversalEmbedderOptions`. When no encoder runs on
   * GPU, no WebGPU device is created and `device` is ignored.
   */
  delegate?: UniversalEmbedderDelegate;
  /**
   * An optional pre-created GPUDevice to use for GPU inference. Ignored when no
   * encoder runs on GPU.
   */
  device?: GPUDevice;
}

import {
  AudioPart,
  ContentPart,
  EmbeddingProvider,
  ImagePart,
  TextPart,
} from '../../../../tasks/web/core/embedding_provider';

export type {AudioPart, ContentPart, EmbeddingProvider, ImagePart, TextPart};

/** Activation data type for model execution. */
export type ActivationDataType = 'FLOAT32' | 'FLOAT16' | 'INT16' | 'INT8';

/** Options to configure the MediaPipe Universal Embedder Task. */
export declare interface UniversalEmbedderOptions {
  /** Base options specifying the model asset and WebGPU device. */
  baseOptions: UniversalEmbedderBaseOptions;
  /**
   * Optional backend for the text encoder. Defaults to
   * `baseOptions.delegate`.
   */
  textDelegate?: UniversalEmbedderDelegate;
  /**
   * Optional backend for the vision encoder. Defaults to
   * `baseOptions.delegate`.
   */
  visionDelegate?: UniversalEmbedderDelegate;
  /**
   * Optional backend for the audio encoder. Defaults to
   * `baseOptions.delegate`.
   */
  audioDelegate?: UniversalEmbedderDelegate;
  /** Whether to L2-normalize the output embedding vector. Defaults to true. */
  l2Normalize?: boolean;
  /** Optional maximum sequence length (in tokens) for text encoder signatures. */
  maxInputLength?: number;
  /** Optional vision tokens per image for vision encoder signatures. */
  visionTokensPerImage?: number;
  /** Optional activation data type for model execution. */
  activationDataType?: ActivationDataType;
}
