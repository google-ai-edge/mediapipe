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

#include "mediapipe/tasks/cc/components/utils/cosine_similarity.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"

namespace mediapipe {
namespace tasks {
namespace components {
namespace utils {

namespace {

using ::mediapipe::tasks::components::containers::Embedding;

template <typename T>
absl::StatusOr<double> ComputeCosineSimilarity(const T& u, const T& v,
                                               int num_elements) {
  if (num_elements <= 0) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Cannot compute cosing similarity on empty embeddings",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  double dot_product = 0.0;
  double norm_u = 0.0;
  double norm_v = 0.0;
  for (int i = 0; i < num_elements; ++i) {
    dot_product += u[i] * v[i];
    norm_u += u[i] * u[i];
    norm_v += v[i] * v[i];
  }
  if (norm_u <= 0.0 || norm_v <= 0.0) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Cannot compute cosine similarity on embedding with 0 norm",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  return dot_product / std::sqrt(norm_u * norm_v);
}

}  // namespace

// Utility function to compute cosine similarity [1] between two embedding
// entries. May return an InvalidArgumentError if e.g. the feature vectors are
// of different types (quantized vs. float), have different sizes, or have a
// an L2-norm of 0.
//
// [1]: https://en.wikipedia.org/wiki/Cosine_similarity
absl::StatusOr<double> CosineSimilarity(const Embedding& u,
                                        const Embedding& v) {
  if (!u.float_embedding.empty() && !v.float_embedding.empty()) {
    if (u.float_embedding.size() != v.float_embedding.size()) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat("Cannot compute cosine similarity between embeddings "
                          "of different sizes (%d vs. %d)",
                          u.float_embedding.size(), v.float_embedding.size()),
          MediaPipeTasksStatus::kInvalidArgumentError);
    }
    return ComputeCosineSimilarity(u.float_embedding.data(),
                                   v.float_embedding.data(),
                                   u.float_embedding.size());
  }
  if (!u.quantized_embedding.empty() && !v.quantized_embedding.empty()) {
    if (u.quantized_embedding.size() != v.quantized_embedding.size()) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat("Cannot compute cosine similarity between embeddings "
                          "of different sizes (%d vs. %d)",
                          u.quantized_embedding.size(),
                          v.quantized_embedding.size()),
          MediaPipeTasksStatus::kInvalidArgumentError);
    }
    return ComputeCosineSimilarity(
        reinterpret_cast<const int8_t*>(u.quantized_embedding.data()),
        reinterpret_cast<const int8_t*>(v.quantized_embedding.data()),
        u.quantized_embedding.size());
  }
  return CreateStatusWithPayload(
      absl::StatusCode::kInvalidArgument,
      "Cannot compute cosine similarity between quantized and float embeddings",
      MediaPipeTasksStatus::kInvalidArgumentError);
}

}  // namespace utils
}  // namespace components
}  // namespace tasks
}  // namespace mediapipe
