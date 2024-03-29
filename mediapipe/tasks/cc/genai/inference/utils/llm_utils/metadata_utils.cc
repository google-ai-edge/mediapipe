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

#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/metadata_utils.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/tasks/cc/genai/inference/proto/llm_params.pb.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe::tasks::genai::llm_utils {

absl::StatusOr<odml::infra::proto::LlmModelType> GetLlmModelType(
    const ::tflite::FlatBufferModel& fb_model) {
  const ::tflite::Model* tflite_model = fb_model.GetModel();
  if (tflite_model->metadata() != nullptr) {
    for (const auto& metadata : *tflite_model->metadata()) {
      if (kLlmModelTypeName == metadata->name()->c_str()) {
        int llm_model_type_index = metadata->buffer();
        odml::infra::proto::LlmModelType llm_model_type =
            static_cast<odml::infra::proto::LlmModelType>(llm_model_type_index);

        return llm_model_type;
      }
    }
  }
  return absl::NotFoundError(
      absl::StrCat("Failed to get LLM model type, missing ", kLlmModelTypeName,
                   " in tflite metadata"));
}

absl::StatusOr<absl::string_view> ExtractSentencePieceToStringView(
    const tflite::FlatBufferModel& model, absl::string_view metadata_key) {
  const std::string key =
      std::string(metadata_key.empty() ? kSpmVocabName : metadata_key);
  for (const auto& metadata : *model.GetModel()->metadata()) {
    if (key == metadata->name()->c_str()) {
      const int spm_vocab_index = metadata->buffer();
      auto spm_vocab_buffer = model.GetModel()->buffers()->Get(spm_vocab_index);
      return absl::string_view(
          static_cast<const char*>(model.allocation()->base()) +
              spm_vocab_buffer->offset(),
          spm_vocab_buffer->size());
    }
  }

  return absl::InvalidArgumentError(
      absl::StrCat(key, " missing in tflite metadata"));
}

}  // namespace mediapipe::tasks::genai::llm_utils
