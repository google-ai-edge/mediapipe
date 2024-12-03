/**
 * Copyright 2022 The MediaPipe Authors.
 *
 * <p>Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * <p>http://www.apache.org/licenses/LICENSE-2.0
 *
 * <p>Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

import {Embedding} from '../../../../tasks/web/components/containers/embedding_result';

/**
 * Computes cosine similarity[1] between two `Embedding` objects.
 *
 * [1]: https://en.wikipedia.org/wiki/Cosine_similarity
 *
 * @throws if the embeddings are of different types (float vs. quantized),
 *     have different sizes, or have an L2-norm of 0.
 */
export function computeCosineSimilarity(u: Embedding, v: Embedding): number {
  if (u.floatEmbedding && v.floatEmbedding) {
    return compute(u.floatEmbedding, v.floatEmbedding);
  }
  if (u.quantizedEmbedding && v.quantizedEmbedding) {
    return compute(
        convertToBytes(u.quantizedEmbedding),
        convertToBytes(v.quantizedEmbedding));
  }
  throw new Error(
      'Cannot compute cosine similarity between quantized and float embeddings.');
}

function convertToBytes(data: Uint8Array): number[] {
  return Array.from(data, v => v > 127 ? v - 256 : v);
}

function compute(u: readonly number[], v: readonly number[]) {
  if (u.length !== v.length) {
    throw new Error(
        `Cannot compute cosine similarity between embeddings of different sizes (${
            u.length} vs. ${v.length}).`);
  }
  let dotProduct = 0.0;
  let normU = 0.0;
  let normV = 0.0;
  for (let i = 0; i < u.length; i++) {
    dotProduct += u[i] * v[i];
    normU += u[i] * u[i];
    normV += v[i] * v[i];
  }
  if (normU <= 0 || normV <= 0) {
    throw new Error(
        'Cannot compute cosine similarity on embedding with 0 norm.');
  }
  return dotProduct / Math.sqrt(normU * normV);
}
