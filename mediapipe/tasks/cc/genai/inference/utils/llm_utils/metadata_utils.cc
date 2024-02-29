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

#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/metadata_utils.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/genai/inference/proto/llm_params.pb.h"
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/memory_mapped_file.h"
#include "tensorflow/lite/model_builder.h"

namespace mediapipe::tasks::genai::llm_utils {
namespace {

absl::StatusOr<std::shared_ptr<const ::tflite::Model>> GetTfliteModel(
    std::shared_ptr<mediapipe::tasks::genai::llm_utils::MemoryMappedFile>
        mmap_file) {
  if (mmap_file == nullptr) {
    return absl::InvalidArgumentError("LLM model file is null");
  }
  auto tflite_model = std::shared_ptr<const ::tflite::Model>(
      mmap_file, ::tflite::GetModel(mmap_file->data()));

  return tflite_model;
}

}  // namespace

absl::StatusOr<odml::infra::proto::LlmParameters> GetLlmParams(
    std::shared_ptr<mediapipe::tasks::genai::llm_utils::MemoryMappedFile>
        mmap_file) {
  auto tflite_model = GetTfliteModel(mmap_file);
  if (!tflite_model.ok()) {
    return tflite_model.status();
  }

  for (const auto& metadata : *(*tflite_model)->metadata()) {
    if (metadata->name()->c_str() ==
        odml::infra::proto::LlmParameters().GetTypeName()) {
      int llm_params_index = metadata->buffer();
      auto llm_params_buffer =
          (*tflite_model)->buffers()->Get(llm_params_index);
      std::string llm_params_str(
          (char*)mmap_file->data() + llm_params_buffer->offset(),
          llm_params_buffer->size());
      odml::infra::proto::LlmParameters llm_params;
      llm_params.ParseFromString(llm_params_str);

      return llm_params;
    }
  }
  return absl::NotFoundError(
      absl::StrCat("Failed to get LLM params, missing ",
                   odml::infra::proto::LlmParameters().GetTypeName(),
                   " in tflite metadata"));
}

absl::StatusOr<odml::infra::proto::LlmParameters> GetLlmParams(
    const ::tflite::FlatBufferModel& fb_model) {
  const ::tflite::Model* tflite_model = fb_model.GetModel();

  if (tflite_model->metadata() != nullptr) {
    for (const auto& metadata : *tflite_model->metadata()) {
      if (metadata->name()->c_str() ==
          odml::infra::proto::LlmParameters().GetTypeName()) {
        int llm_params_index = metadata->buffer();
        auto llm_params_buffer = tflite_model->buffers()->Get(llm_params_index);
        std::string llm_params_str(
            (char*)fb_model.allocation()->base() + llm_params_buffer->offset(),
            llm_params_buffer->size());
        odml::infra::proto::LlmParameters llm_params;
        llm_params.ParseFromString(llm_params_str);

        return llm_params;
      }
    }
  }
  return absl::NotFoundError(
      absl::StrCat("Failed to get LLM params, missing ",
                   odml::infra::proto::LlmParameters().GetTypeName(),
                   " in tflite metadata"));
}

absl::StatusOr<odml::infra::proto::LlmModelType> GetLlmModelType(
    std::shared_ptr<mediapipe::tasks::genai::llm_utils::MemoryMappedFile>
        mmap_file) {
  auto tflite_model = GetTfliteModel(mmap_file);
  if (!tflite_model.ok()) {
    return tflite_model.status();
  }

  for (const auto& metadata : *(*tflite_model)->metadata()) {
    if (kLlmModelTypeName == metadata->name()->c_str()) {
      int llm_model_type_index = metadata->buffer();
      odml::infra::proto::LlmModelType llm_model_type =
          static_cast<odml::infra::proto::LlmModelType>(llm_model_type_index);

      return llm_model_type;
    }
  }
  return absl::NotFoundError(
      absl::StrCat("Failed to get LLM model type, missing ", kLlmModelTypeName,
                   " in tflite metadata"));
}

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

absl::StatusOr<std::string> GetLlmBackend(
    std::shared_ptr<mediapipe::tasks::genai::llm_utils::MemoryMappedFile>
        mmap_file) {
  auto tflite_model = GetTfliteModel(mmap_file);
  if (!tflite_model.ok()) {
    return tflite_model.status();
  }

  for (const auto& metadata : *(*tflite_model)->metadata()) {
    if (kLlmBackendName == metadata->name()->c_str()) {
      int backend_index = metadata->buffer();
      auto backend_buffer = (*tflite_model)->buffers()->Get(backend_index);
      std::string backend_str(
          (char*)mmap_file->data() + backend_buffer->offset(),
          backend_buffer->size());

      return backend_str;
    }
  }
  return absl::NotFoundError(
      absl::StrCat("Failed to get backend for LLM inference, missing ",
                   kLlmBackendName, " in tflite metadata"));
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

absl::Status ExtractSentencePiece(
    std::shared_ptr<mediapipe::tasks::genai::llm_utils::MemoryMappedFile>
        mmap_file,
    absl::string_view spm_vocab_path) {
  if (spm_vocab_path.empty()) {
    return absl::InvalidArgumentError("SentencePiece model path is empty");
  }

  // File already exists. We don't need to extract it. Note that this might
  // actually be a different model file (though unlikely), but we might want
  // to use a unique path to handle this case.
  if (mediapipe::file::Exists(spm_vocab_path).ok()) {
    ABSL_LOG(WARNING)
        << "Skipped extracting SentencePiece model, SentencePiece "
           "model already exists: "
        << spm_vocab_path;
    return absl::OkStatus();
  }

  auto tflite_model = tflite::FlatBufferModel::BuildFromBuffer(
      static_cast<char*>(mmap_file->data()), mmap_file->length());

  MP_ASSIGN_OR_RETURN(
      absl::string_view spm_vocab_str,
      ExtractSentencePieceToStringView(*tflite_model, kSpmVocabName));
  return mediapipe::file::SetContents(spm_vocab_path, std::move(spm_vocab_str));
}

}  // namespace mediapipe::tasks::genai::llm_utils
