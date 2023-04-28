/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/cc/components/containers/embedding_result.h"

#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "mediapipe/tasks/cc/components/containers/proto/embeddings.pb.h"

namespace mediapipe::tasks::components::containers {

Embedding ConvertToEmbedding(const proto::Embedding& proto) {
  Embedding embedding;
  if (proto.has_float_embedding()) {
    embedding.float_embedding = {
        std::make_move_iterator(proto.float_embedding().values().begin()),
        std::make_move_iterator(proto.float_embedding().values().end())};
  } else {
    embedding.quantized_embedding = {
        std::make_move_iterator(proto.quantized_embedding().values().begin()),
        std::make_move_iterator(proto.quantized_embedding().values().end())};
  }
  embedding.head_index = proto.head_index();
  if (proto.has_head_name()) {
    embedding.head_name = proto.head_name();
  }
  return embedding;
}

EmbeddingResult ConvertToEmbeddingResult(const proto::EmbeddingResult& proto) {
  EmbeddingResult embedding_result;
  embedding_result.embeddings.reserve(proto.embeddings_size());
  for (const auto& embedding : proto.embeddings()) {
    embedding_result.embeddings.push_back(ConvertToEmbedding(embedding));
  }
  if (proto.has_timestamp_ms()) {
    embedding_result.timestamp_ms = proto.timestamp_ms();
  }
  return embedding_result;
}

}  // namespace mediapipe::tasks::components::containers
