// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/tasks/c/genai/bundler/llm_bundler_utils.h"

#include <string>

#ifdef ENABLE_ODML_CONVERTER
#include "odml/infra/genai/inference/proto/llm_params.pb.h"
#include "odml/infra/genai/inference/proto/prompt_template.pb.h"
#else
#include "absl/log/absl_log.h"
#endif  // ENABLE_ODML_CONVERTER

extern "C" {

const char* MpLlmBundlerGenerateMetadata(
    const LlmBundlerMetadataOptions* options, int* metadata_buffer_size) {
#ifdef ENABLE_ODML_CONVERTER
  odml::infra::proto::LlmParameters params;
  params.set_start_token(options->start_token);
  for (int i = 0; i < options->num_stop_tokens; ++i) {
    params.add_stop_tokens(options->stop_tokens[i]);
  }
  if (options->enable_bytes_to_unicode_mapping) {
    params.add_input_output_normalizations(
        odml::infra::proto::LlmParameters::
            INPUT_OUTPUT_NORMALIZATION_BYTES_TO_UNICODE);
  }
  if (options->system_prompt) {
    params.mutable_prompt_template()->set_session_prefix(
        options->system_prompt);
  } else {
    if (options->prompt_prefix_system) {
      params.mutable_prompt_templates()
          ->mutable_system_template()
          ->set_prompt_prefix(options->prompt_prefix_system);
    }
    if (options->prompt_suffix_system) {
      params.mutable_prompt_templates()
          ->mutable_system_template()
          ->set_prompt_suffix(options->prompt_suffix_system);
    }
  }
  if (options->prompt_prefix_user) {
    params.mutable_prompt_template()->set_prompt_prefix(
        options->prompt_prefix_user);
    params.mutable_prompt_templates()
        ->mutable_user_template()
        ->set_prompt_prefix(options->prompt_prefix_user);
  }
  if (options->prompt_suffix_user) {
    std::string suffix = options->prompt_suffix_user;
    if (options->prompt_prefix_model) {
      suffix += options->prompt_prefix_model;
    }
    params.mutable_prompt_template()->set_prompt_suffix(suffix);
    params.mutable_prompt_templates()
        ->mutable_user_template()
        ->set_prompt_suffix(options->prompt_suffix_user);
  }
  if (options->prompt_prefix_model) {
    params.mutable_prompt_templates()
        ->mutable_model_template()
        ->set_prompt_prefix(options->prompt_prefix_model);
  }
  if (options->prompt_suffix_model) {
    params.mutable_prompt_templates()
        ->mutable_model_template()
        ->set_prompt_suffix(options->prompt_suffix_model);
  }
  if (options->user_role_token) {
    params.set_user_role_token(options->user_role_token);
  }
  if (options->system_role_token) {
    params.set_system_role_token(options->system_role_token);
  }
  if (options->model_role_token) {
    params.set_model_role_token(options->model_role_token);
  }
  if (options->end_role_token) {
    params.set_end_role_token(options->end_role_token);
  }

  std::string s = params.SerializeAsString();
  *metadata_buffer_size = s.length();
  char* metadata_buffer = new char[*metadata_buffer_size];
  s.copy(metadata_buffer, *metadata_buffer_size);
  return metadata_buffer;
#else
  ABSL_LOG(ERROR) << "LLM bundler is not enabled.";
  return nullptr;
#endif  // ENABLE_ODML_CONVERTER
}

void MpLlmBundlerFreeMetadata(const char* metadata_buffer) {
  delete[] metadata_buffer;
}

}  // extern "C"
