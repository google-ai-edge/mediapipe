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

#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/benchmark_weight_accessor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/xnn_tensor.h"
#include "xnnpack.h"  // from @XNNPACK

namespace mediapipe::tasks::genai {
namespace xnn_utils {

uint64_t Hash(absl::string_view s) {
  return absl::Hash<absl::string_view>()(s);
}

absl::StatusOr<std::shared_ptr<Tensor>> BenchmarkWeightAccessor::LoadWeight(
    absl::string_view prefix, Tensor::DimsType dims,
    size_t dim_scale_if_any) const {
  std::optional<std::mt19937> rng =
      seed_.has_value()
          ? std::make_optional(std::mt19937(Hash(prefix) ^ seed_.value()))
          : std::nullopt;
  std::shared_ptr<Tensor> result;
  if (data_type_ == xnn_datatype_fp32 || !absl::StrContains(prefix, ".w")) {
    result = std::make_shared<Tensor>(dims, xnn_datatype_fp32);
    std::vector<float> real_data(
        // -2.8735182454e-16 == 0xA5A5A5A5
        result->num_elements, -2.8735182454e-16);
    if (rng.has_value()) {
      std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
      for (auto& c : real_data) {
        c = dist(*rng);
      }
    }
    MP_RETURN_IF_ERROR(result->LoadFromBuffer(real_data.data()));
  } else {
    std::string real_data;
    auto q_result =
        std::make_shared<QCTensor>(dims, dim_scale_if_any, data_type_);
    switch (data_type_) {
      case xnn_datatype_qcint8: {
        real_data = std::string(q_result->num_elements, 0xA5);
        break;
      }
      case xnn_datatype_qcint4: {
        real_data = std::string((q_result->num_elements + 1) / 2, 0xA5);
        break;
      }
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unknown datatype ", data_type_));
    }
    if (rng.has_value()) {
      std::uniform_int_distribution<int8_t> dist(-127, 126);
      for (auto& c : real_data) {
        c = dist(*rng);
      }
    }
    MP_RETURN_IF_ERROR(q_result->LoadFromBuffer(real_data.data()));
    auto real_scale =
        std::make_shared<std::vector<float>>(dims[dim_scale_if_any], 1.0f);
    q_result->scale_data =
        std::shared_ptr<float>(real_scale, real_scale->data());
    result = q_result;
  }
  return result;
}

absl::StatusOr<std::shared_ptr<Tensor>>
BenchmarkWeightAccessor::LoadTransposedWeight(absl::string_view prefix,
                                              Tensor::DimsType dims,
                                              size_t dim_scale_if_any) const {
  return LoadWeight(prefix, Tensor::DimsType(dims.rbegin(), dims.rend()),
                    1 - dim_scale_if_any);
}

absl::StatusOr<std::shared_ptr<Tensor>>
BenchmarkMixedInt48WeightAccessor::LoadWeight(absl::string_view filename_prefix,
                                              Tensor::DimsType dims,
                                              size_t dim_scale_if_any) const {
  if (absl::StrContains(filename_prefix, "ff_layer.ffn_layer1") ||
      absl::StrContains(filename_prefix, "ff_layer.ffn_layer2") ||
      absl::StrContains(filename_prefix, "softmax.logits_ffn") ||
      absl::StrContains(filename_prefix, "embedding") ||
      absl::StrContains(filename_prefix, "mlp")) {
    return int4_weight_loader_->LoadWeight(filename_prefix, dims,
                                           dim_scale_if_any);
  }
  return BenchmarkWeightAccessor::LoadWeight(filename_prefix, dims,
                                             dim_scale_if_any);
}

}  // namespace xnn_utils
}  // namespace mediapipe::tasks::genai
