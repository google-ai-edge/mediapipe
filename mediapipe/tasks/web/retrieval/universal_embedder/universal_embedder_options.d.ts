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

/** Options for configuring the model asset and hardware accelerator. */
export declare interface UniversalEmbedderBaseOptions {
  /** The model asset buffer containing the .litertlm file. */
  modelAssetBuffer?: Uint8Array | ReadableStreamDefaultReader;
  /** The path to the model asset file. */
  modelAssetPath?: string | string;
  /** The hardware accelerator delegate to use ('CPU' or 'GPU'). */
  delegate?: 'CPU' | 'GPU';
  /** An optional pre-created GPUDevice to use for GPU inference. */
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
  /** Base options specifying the model asset and delegate. */
  baseOptions: UniversalEmbedderBaseOptions;
  /** Whether to L2-normalize the output embedding vector. Defaults to true. */
  l2Normalize?: boolean;
  /** Optional maximum sequence length (in tokens) for text encoder signatures. */
  maxInputLength?: number;
  /** Optional vision tokens per image for vision encoder signatures. */
  visionTokensPerImage?: number;
  /** Optional activation data type for model execution. */
  activationDataType?: ActivationDataType;
}
