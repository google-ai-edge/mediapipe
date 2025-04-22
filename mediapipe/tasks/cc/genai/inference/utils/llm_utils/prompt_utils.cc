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

#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/prompt_utils.h"

#include <string>

#include "absl/status/statusor.h"

namespace mediapipe::tasks::genai::llm_utils {

absl::StatusOr<std::string> GetPromptPrefixFromSinglePromptTemplate(
    const odml::infra::proto::PromptTemplate& prompt_template,
    odml::infra::proto::PromptRole last_prompt_role,
    odml::infra::proto::PromptRole current_prompt_role) {
  odml::infra::proto::PromptTemplates prompt_templates;
  *prompt_templates.mutable_user_template() = prompt_template;
  return GetPromptPrefixFromPromptTemplates(prompt_templates, last_prompt_role,
                                            current_prompt_role);
}

absl::StatusOr<std::string> GetPromptPrefixFromPromptTemplates(
    const odml::infra::proto::PromptTemplates& prompt_templates,
    odml::infra::proto::PromptRole last_prompt_role,
    odml::infra::proto::PromptRole current_prompt_role) {
  std::string prompt_prefix = "";
  if (last_prompt_role != current_prompt_role) {
    switch (last_prompt_role) {
      case odml::infra::proto::PromptRole::PROMPT_ROLE_USER:
        if (prompt_templates.has_user_template()) {
          prompt_prefix = prompt_templates.user_template().prompt_suffix();
        }
        break;
      case odml::infra::proto::PromptRole::PROMPT_ROLE_MODEL:
        if (prompt_templates.has_model_template()) {
          prompt_prefix = prompt_templates.model_template().prompt_suffix();
        }
        break;
      case odml::infra::proto::PromptRole::PROMPT_ROLE_SYSTEM:
        if (prompt_templates.has_system_template()) {
          prompt_prefix = prompt_templates.system_template().prompt_suffix();
        }
        break;
      default:
        break;
    }

    switch (current_prompt_role) {
      case odml::infra::proto::PromptRole::PROMPT_ROLE_USER:
        if (prompt_templates.has_user_template()) {
          prompt_prefix =
              prompt_prefix + prompt_templates.user_template().prompt_prefix();
        }
        break;
      case odml::infra::proto::PromptRole::PROMPT_ROLE_MODEL:
        if (prompt_templates.has_model_template()) {
          prompt_prefix =
              prompt_prefix + prompt_templates.model_template().prompt_prefix();
        }
        break;
      case odml::infra::proto::PromptRole::PROMPT_ROLE_SYSTEM:
        if (prompt_templates.has_system_template()) {
          prompt_prefix = prompt_prefix +
                          prompt_templates.system_template().prompt_prefix();
        }
        break;
      default:
        break;
    }
  }
  return prompt_prefix;
}

}  // namespace mediapipe::tasks::genai::llm_utils
