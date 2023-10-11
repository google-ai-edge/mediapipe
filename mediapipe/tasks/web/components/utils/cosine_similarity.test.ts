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

import {computeCosineSimilarity} from './cosine_similarity';

describe('computeCosineSimilarity', () => {
  it('fails with quantized and float embeddings', () => {
    const u: Embedding = {floatEmbedding: [1.0], headIndex: 0, headName: ''};
    const v: Embedding = {
      quantizedEmbedding: new Uint8Array([1.0]),
      headIndex: 0,
      headName: ''
    };

    expect(() => computeCosineSimilarity(u, v))
        .toThrowError(
            /Cannot compute cosine similarity between quantized and float embeddings/);
  });

  it('fails with zero norm', () => {
    const u = {floatEmbedding: [0.0], headIndex: 0, headName: ''};
    expect(() => computeCosineSimilarity(u, u))
        .toThrowError(
            /Cannot compute cosine similarity on embedding with 0 norm/);
  });

  it('fails with different sizes', () => {
    const u:
        Embedding = {floatEmbedding: [1.0, 2.0], headIndex: 0, headName: ''};
    const v: Embedding = {
      floatEmbedding: [1.0, 2.0, 3.0],
      headIndex: 0,
      headName: ''
    };

    expect(() => computeCosineSimilarity(u, v))
        .toThrowError(
            /Cannot compute cosine similarity between embeddings of different sizes/);
  });

  it('succeeds with float embeddings', () => {
    const u: Embedding = {
      floatEmbedding: [1.0, 0.0, 0.0, 0.0],
      headIndex: 0,
      headName: ''
    };
    const v: Embedding = {
      floatEmbedding: [0.5, 0.5, 0.5, 0.5],
      headIndex: 0,
      headName: ''
    };

    expect(computeCosineSimilarity(u, v)).toEqual(0.5);
  });

  it('succeeds with quantized embeddings', () => {
    const u: Embedding = {
      quantizedEmbedding: new Uint8Array([127, 0, 0, 0]),
      headIndex: 0,
      headName: ''
    };
    const v: Embedding = {
      quantizedEmbedding: new Uint8Array([128, 0, 0, 0]),
      headIndex: 0,
      headName: ''
    };

    expect(computeCosineSimilarity(u, v)).toEqual(-1.0);
  });
});
