// Copyright 2024 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MEDIAPIPE_TASKS_GENAI_INFERENCE_UTILS_XNN_UTILS_DISCRETE_WEIGTH_ACCESSOR_H_
#define MEDIAPIPE_TASKS_GENAI_INFERENCE_UTILS_XNN_UTILS_DISCRETE_WEIGTH_ACCESSOR_H_

#include <cstddef>
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/xnn_tensor.h"

namespace mediapipe::tasks::genai::xnn_utils {

// An implementation of WeightLoader that tries to read discrete files from
// `weight_path`.
class DiscreteWeightWeightAccessor : public WeightAccessor {
 public:
  DiscreteWeightWeightAccessor(absl::string_view weight_path,
                               absl::string_view cache_path)
      : weight_path_(weight_path), cache_path_(cache_path) {}
  ~DiscreteWeightWeightAccessor() override = default;

  // Load file with `prefix` and check file size. Use file size to determine
  // data type (FP32, QC8, etc.)
  absl::StatusOr<std::shared_ptr<Tensor>> LoadWeight(
      absl::string_view, Tensor::DimsType,
      size_t dim_scale_if_any) const override;
  // Try loading from cache_path first, return if found; otherwise, load from
  // weight_path, dump the transposed tensor to cache_path and return.
  absl::StatusOr<std::shared_ptr<Tensor>> LoadTransposedWeight(
      absl::string_view, Tensor::DimsType,
      size_t dim_scale_if_any) const override;

 protected:
  std::string weight_path_;
  std::string cache_path_;
};

}  // namespace mediapipe::tasks::genai::xnn_utils

#endif  // MEDIAPIPE_TASKS_GENAI_INFERENCE_UTILS_XNN_UTILS_DISCRETE_WEIGTH_ACCESSOR_H_
