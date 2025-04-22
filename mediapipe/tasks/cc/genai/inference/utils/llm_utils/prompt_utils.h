// Copyright 2025 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_TASKS_GENAI_INFERENCE_UTILS_LLM_UTILS_PROMPT_UTILS_H_
#define MEDIAPIPE_TASKS_GENAI_INFERENCE_UTILS_LLM_UTILS_PROMPT_UTILS_H_

#include <string>

#include "absl/base/attributes.h"
#include "absl/status/statusor.h"
#include "mediapipe/tasks/cc/genai/inference/proto/prompt_template.pb.h"

namespace mediapipe::tasks::genai::llm_utils {

// Please do not use this function. It is deprecated in favor of supporting
// multiple prompt roles via prompt templates. Please use
// GetPromptPrefixFromPromptTemplates instead.
// Returns the prompt prefix from the single prompt template based on the last
// and current prompt roles.
// prompt_template: The prompt template to reference.
// last_prompt_role: The role of the last prompt called.
// current_prompt_role: The role of the current prompt called.
ABSL_DEPRECATED(
    "Deprecated in favor of supporting multiple prompt roles via prompt "
    "templates. Please use GetPromptPrefixFromPromptTemplates instead.")
absl::StatusOr<std::string> GetPromptPrefixFromSinglePromptTemplate(
    const odml::infra::proto::PromptTemplate& prompt_template,
    odml::infra::proto::PromptRole last_prompt_role,
    odml::infra::proto::PromptRole current_prompt_role);

// Returns the prompt prefix from the prompt templates based on the last and
// current prompt roles.
// prompt_templates: The prompt templates to reference.
// last_prompt_role: The role of the last prompt called.
// current_prompt_role: The role of the current prompt called.
absl::StatusOr<std::string> GetPromptPrefixFromPromptTemplates(
    const odml::infra::proto::PromptTemplates& prompt_templates,
    odml::infra::proto::PromptRole last_prompt_role,
    odml::infra::proto::PromptRole current_prompt_role);

}  // namespace mediapipe::tasks::genai::llm_utils

#endif  // MEDIAPIPE_TASKS_GENAI_INFERENCE_UTILS_LLM_UTILS_PROMPT_UTILS_H_
