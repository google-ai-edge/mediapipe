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
#include "absl/strings/match.h"

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

// TODO: b/400470302 - Remove this once the prompt templates are bundled within
// the model.
absl::StatusOr<odml::infra::proto::PromptTemplates>
PredictPromptTemplatesFromPromptTemplate(
    const odml::infra::proto::PromptTemplate& prompt_template) {
  odml::infra::proto::PromptTemplates prompt_templates;
  if (!prompt_template.prompt_prefix().empty()) {
    prompt_templates.mutable_user_template()->set_prompt_prefix(
        prompt_template.prompt_prefix());
  }
  if (!prompt_template.prompt_suffix().empty()) {
    int newline_pos = prompt_template.prompt_suffix().find('\n');
    if (newline_pos != std::string::npos) {
      std::string suffix_until_newline = prompt_template.prompt_suffix().substr(
          0, prompt_template.prompt_suffix().find('\n'));
      suffix_until_newline = suffix_until_newline + "\n";
      std::string suffix_after_newline = prompt_template.prompt_suffix().substr(
          prompt_template.prompt_suffix().find('\n'));
      suffix_after_newline = suffix_after_newline.substr(1);
      prompt_templates.mutable_user_template()->set_prompt_suffix(
          suffix_until_newline);
      prompt_templates.mutable_model_template()->set_prompt_prefix(
          suffix_after_newline);
    } else {
      prompt_templates.mutable_model_template()->set_prompt_prefix(
          prompt_template.prompt_suffix());
    }
  }
  // Predict the model suffix from the provided information.
  if (prompt_templates.has_user_template()) {
    if (absl::StrContains(prompt_templates.user_template().prompt_prefix(),
                          "<start_of_turn>") ||
        absl::StrContains(prompt_templates.user_template().prompt_prefix(),
                          "<ctrl99>")) {
      prompt_templates.mutable_model_template()->set_prompt_suffix(
          prompt_templates.user_template().prompt_suffix());
    }
  }
  return prompt_templates;
}

}  // namespace mediapipe::tasks::genai::llm_utils
