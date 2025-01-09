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

import {Embedding as EmbeddingProto, EmbeddingResult as EmbeddingResultProto} from '../../../../tasks/cc/components/containers/proto/embeddings_pb';
import {Embedding, EmbeddingResult} from '../../../../tasks/web/components/containers/embedding_result';

const DEFAULT_INDEX = -1;

/**
 * Converts an Embedding proto to the Embedding object.
 */
function convertFromEmbeddingsProto(source: EmbeddingProto): Embedding {
  const embedding: Embedding = {
    headIndex: source.getHeadIndex() ?? DEFAULT_INDEX,
    headName: source.getHeadName() ?? '',
  };

  if (source.hasFloatEmbedding()) {
    embedding.floatEmbedding =
        source.getFloatEmbedding()!.getValuesList().slice();
  } else {
      const encodedValue = source.getQuantizedEmbedding()?.getValues() ?? '';
      embedding.quantizedEmbedding = typeof encodedValue == 'string' ?
          Uint8Array.from(atob(encodedValue), c => c.charCodeAt(0)) : encodedValue;
  }

  return embedding;
}

/**
 * Converts an EmbedderResult proto to an EmbeddingResult object.
 */
export function convertFromEmbeddingResultProto(
    embeddingResult: EmbeddingResultProto): EmbeddingResult {
  const result: EmbeddingResult = {
    embeddings: embeddingResult.getEmbeddingsList().map(
        e => convertFromEmbeddingsProto(e)),
    timestampMs: embeddingResult.getTimestampMs(),
  };
  return result;
}
