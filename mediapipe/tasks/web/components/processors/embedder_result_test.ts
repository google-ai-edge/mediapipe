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

import 'jasmine';

import {Embedding, EmbeddingResult, FloatEmbedding, QuantizedEmbedding} from '../../../../tasks/cc/components/containers/proto/embeddings_pb';

import {convertFromEmbeddingResultProto} from './embedder_result';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

describe('convertFromEmbeddingResultProto()', () => {
  it('transforms custom values', () => {
    const embedding = new Embedding();
    embedding.setHeadIndex(1);
    embedding.setHeadName('headName');

    const floatEmbedding = new FloatEmbedding();
    floatEmbedding.setValuesList([0.1, 0.9]);

    embedding.setFloatEmbedding(floatEmbedding);
    const resultProto = new EmbeddingResult();
    resultProto.addEmbeddings(embedding);
    resultProto.setTimestampMs(1);

    const embedderResult = convertFromEmbeddingResultProto(resultProto);
    const embeddings = embedderResult.embeddings;
    const timestampMs = embedderResult.timestampMs;
    expect(embeddings.length).toEqual(1);
    expect(embeddings[0])
        .toEqual(
            {floatEmbedding: [0.1, 0.9], headIndex: 1, headName: 'headName'});
    expect(timestampMs).toEqual(1);
  });

  it('transforms custom quantized values', () => {
    const embedding = new Embedding();
    embedding.setHeadIndex(1);
    embedding.setHeadName('headName');

    const quantizedEmbedding = new QuantizedEmbedding();
    const quantizedValues = new Uint8Array([1, 2, 3]);
    quantizedEmbedding.setValues(quantizedValues);

    embedding.setQuantizedEmbedding(quantizedEmbedding);
    const resultProto = new EmbeddingResult();
    resultProto.addEmbeddings(embedding);
    resultProto.setTimestampMs(1);

    const embedderResult = convertFromEmbeddingResultProto(resultProto);
    const embeddings = embedderResult.embeddings;
    const timestampMs = embedderResult.timestampMs;
    expect(embeddings.length).toEqual(1);
    expect(embeddings[0]).toEqual({
      quantizedEmbedding: new Uint8Array([1, 2, 3]),
      headIndex: 1,
      headName: 'headName'
    });
    expect(timestampMs).toEqual(1);
  });
});
