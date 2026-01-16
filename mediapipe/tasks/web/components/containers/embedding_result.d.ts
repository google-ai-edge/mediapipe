/**
 * Copyright 2022 The MediaPipe Authors.
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

/**
 * List of embeddings with an optional timestamp.
 *
 * One and only one of the two 'floatEmbedding' and 'quantizedEmbedding' will
 * contain data, based on whether or not the embedder was configured to perform
 * scalar quantization.
 */
export declare interface Embedding {
  /**
   *  Floating-point embedding. Empty if the embedder was configured to perform
   * scalar-quantization.
   */
  floatEmbedding?: number[];

  /**
   * Scalar-quantized embedding. Empty if the embedder was not configured to
   * perform scalar quantization.
   */
  quantizedEmbedding?: Uint8Array;

  /**
   * The index of the classifier head these categories refer to. This is
   * useful for multi-head models.
   */
  headIndex: number;

  /**
   * The name of the classifier head, which is the corresponding tensor
   * metadata name.
   */
  headName: string;
}

/**  Embedding results for a given embedder model. */
export declare interface EmbeddingResult {
  /**
   * The embedding results for each model head, i.e. one for each output tensor.
   */
  embeddings: Embedding[];

  /**
   * The optional timestamp (in milliseconds) of the start of the chunk of
   * data corresponding to these results.
   *
   * This is only used for embedding extraction on time series (e.g. audio
   * embedding). In these use cases, the amount of data to process might
   * exceed the maximum size that the model can process: to solve this, the
   * input data is split into multiple chunks starting at different timestamps.
   */
  timestampMs?: number;
}
