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

// TODO: Add unit test.

#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm_builder_factory.h"

#include <memory>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/tasks/cc/genai/inference/proto/llm_params.pb.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/falcon.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/graph_builder.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/llm_weights.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/phi.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/sampling.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/stablelm.h"

namespace mediapipe::tasks::genai::xnn_utils {

absl::StatusOr<std::unique_ptr<LlmBuilder>> CreateLlmBuilder(
    const LlmParams& llm_params,
    std::unique_ptr<RuntimeConfigs> runtime_configs,
    std::unique_ptr<Sampler> sampler,
    odml::infra::proto::LlmModelType model_type) {
  std::unique_ptr<LlmBuilder> builder;
  switch (model_type) {
    case odml::infra::proto::LLM_MODEL_TYPE_FALCON_RW_1B:
      builder = std::make_unique<FalconRW1BBuilder>(
          llm_params, std::move(sampler), std::move(runtime_configs));
      break;
    case odml::infra::proto::LLM_MODEL_TYPE_STABLELM_4E1T_3B:
      builder = std::make_unique<Stablelm4E1T3BBuilder>(
          llm_params, std::move(sampler), std::move(runtime_configs));
      break;
    case odml::infra::proto::LLM_MODEL_TYPE_PHI_2:
      builder = std::make_unique<Phi2Builder>(llm_params, std::move(sampler),
                                              std::move(runtime_configs));
      break;
    case odml::infra::proto::LLM_MODEL_TYPE_GEMMA_2B:
      ABSL_FALLTHROUGH_INTENDED;
    case odml::infra::proto::LLM_MODEL_TYPE_GEMMA_7B:
      builder = std::make_unique<LlmBuilder>(llm_params, std::move(sampler),
                                             std::move(runtime_configs));
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported model type: ", model_type));
  }
  return builder;
}

}  // namespace mediapipe::tasks::genai::xnn_utils
