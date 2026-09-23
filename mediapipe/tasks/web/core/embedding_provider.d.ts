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

/** A text part in a multimodal content block. */
export declare interface TextPart {
  text: string;
}

/** An image part in a multimodal content block. */
export declare interface ImagePart {
  imageBytes: Uint8Array;
}

/** An audio part in a multimodal content block. */
export declare interface AudioPart {
  audioData: Float32Array;
}

/** A single multimodal content part. */
export type ContentPart = TextPart | ImagePart | AudioPart;

/** Package-neutral interface for generating embeddings across modalities. */
export interface EmbeddingProvider {
  /**
   * Generates a high-dimensional vector embedding for the given content parts.
   * Returns null if the content parts are unsupported by this provider.
   */
  embedContent(
    content: readonly ContentPart[],
  ): Promise<Float32Array | number[] | null> | Float32Array | number[] | null;
}
