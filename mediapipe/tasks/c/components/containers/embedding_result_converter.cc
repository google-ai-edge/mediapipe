/* Copyright 2023 The MediaPipe Authors.

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

#include "mediapipe/tasks/c/components/containers/embedding_result_converter.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "mediapipe/tasks/c/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"

namespace mediapipe::tasks::c::components::containers {

void CppConvertToEmbeddingResult(
    const mediapipe::tasks::components::containers::EmbeddingResult& in,
    EmbeddingResult* out) {
  out->has_timestamp_ms = in.timestamp_ms.has_value();
  out->timestamp_ms = out->has_timestamp_ms ? in.timestamp_ms.value() : 0;

  out->embeddings_count = in.embeddings.size();
  out->embeddings =
      out->embeddings_count ? new Embedding[out->embeddings_count] : nullptr;

  for (uint32_t i = 0; i < out->embeddings_count; ++i) {
    auto embedding_in = in.embeddings[i];
    auto& embedding_out = out->embeddings[i];
    if (!embedding_in.float_embedding.empty()) {
      // Handle float embeddings
      embedding_out.values_count = embedding_in.float_embedding.size();
      embedding_out.float_embedding = new float[embedding_out.values_count];

      std::copy(embedding_in.float_embedding.begin(),
                embedding_in.float_embedding.end(),
                embedding_out.float_embedding);

      embedding_out.quantized_embedding = nullptr;
    } else if (!embedding_in.quantized_embedding.empty()) {
      // Handle quantized embeddings
      embedding_out.values_count = embedding_in.quantized_embedding.size();
      embedding_out.quantized_embedding = new char[embedding_out.values_count];

      std::copy(embedding_in.quantized_embedding.begin(),
                embedding_in.quantized_embedding.end(),
                embedding_out.quantized_embedding);

      embedding_out.float_embedding = nullptr;
    }

    embedding_out.head_index = embedding_in.head_index;
    embedding_out.head_name = embedding_in.head_name.has_value()
                                  ? strdup(embedding_in.head_name->c_str())
                                  : nullptr;
  }
}

void CppConvertToCppEmbedding(
    const Embedding& in,  // C struct as input
    mediapipe::tasks::components::containers::Embedding* out) {
  // Handle float embeddings
  if (in.float_embedding != nullptr) {
    out->float_embedding.assign(in.float_embedding,
                                in.float_embedding + in.values_count);
  }

  // Handle quantized embeddings
  if (in.quantized_embedding != nullptr) {
    out->quantized_embedding.assign(in.quantized_embedding,
                                    in.quantized_embedding + in.values_count);
  }

  out->head_index = in.head_index;

  // Copy head_name if it is present.
  if (in.head_name) {
    out->head_name = std::string(in.head_name);
  }
}

void CppCloseEmbeddingResult(EmbeddingResult* in) {
  for (uint32_t i = 0; i < in->embeddings_count; ++i) {
    auto embedding_in = in->embeddings[i];

    delete[] embedding_in.float_embedding;
    delete[] embedding_in.quantized_embedding;
    embedding_in.float_embedding = nullptr;
    embedding_in.quantized_embedding = nullptr;

    free(embedding_in.head_name);
  }
  delete[] in->embeddings;
  in->embeddings = nullptr;
}

}  // namespace mediapipe::tasks::c::components::containers
