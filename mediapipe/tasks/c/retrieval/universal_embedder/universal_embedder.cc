/* Copyright 2026 The MediaPipe Authors.

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

#include "mediapipe/tasks/c/retrieval/universal_embedder/universal_embedder.h"

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/c/audio/core/common.h"
#include "mediapipe/tasks/c/components/containers/embedding_result_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/c/retrieval/universal_embedder/universal_embedder_internal.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/retrieval/universal_embedder/universal_embedder.h"

namespace mediapipe::tasks::c::retrieval::universal_embedder {

namespace {

using ::mediapipe::tasks::c::components::containers::CppCloseEmbeddingResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToEmbeddingResult;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::components::containers::EmbeddingResult;
using ::mediapipe::tasks::retrieval::universal_embedder::UniversalEmbedder;

const Image& ToImage(const MpImagePtr mp_image) { return mp_image->image; }

UniversalEmbedder* GetCppEmbedder(MpUniversalEmbedderPtr wrapper) {
  ABSL_CHECK(wrapper != nullptr) << "UniversalEmbedder is null.";
  return wrapper->instance.get();
}

}  // namespace

absl::Status CppUniversalEmbedderCreate(
    const MpUniversalEmbedderOptions& options,
    MpUniversalEmbedderPtr* embedder) {
  auto cpp_options =
      std::make_unique<::mediapipe::tasks::retrieval::universal_embedder::
                           UniversalEmbedderOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  cpp_options->l2_normalize = options.l2_normalize;
  if (options.max_input_length > 0) {
    cpp_options->max_input_length = options.max_input_length;
  }
  if (options.vision_tokens_per_image > 0) {
    cpp_options->vision_tokens_per_image = options.vision_tokens_per_image;
  }
  switch (options.activation_data_type) {
    case kMpActivationDataTypeFloat32:
      cpp_options->activation_data_type = ::mediapipe::tasks::retrieval::
          universal_embedder::ActivationDataType::FLOAT32;
      break;
    case kMpActivationDataTypeFloat16:
      cpp_options->activation_data_type = ::mediapipe::tasks::retrieval::
          universal_embedder::ActivationDataType::FLOAT16;
      break;
    case kMpActivationDataTypeInt16:
      cpp_options->activation_data_type = ::mediapipe::tasks::retrieval::
          universal_embedder::ActivationDataType::INT16;
      break;
    case kMpActivationDataTypeInt8:
      cpp_options->activation_data_type = ::mediapipe::tasks::retrieval::
          universal_embedder::ActivationDataType::INT8;
      break;
    case kMpActivationDataTypeDefault:
    default:
      break;
  }
  if (options.cache_dir != nullptr && strlen(options.cache_dir) > 0) {
    cpp_options->cache_dir = std::string(options.cache_dir);
  }

  auto cpp_embedder = UniversalEmbedder::Create(std::move(cpp_options));
  if (!cpp_embedder.ok()) {
    return cpp_embedder.status();
  }
  *embedder =
      new MpUniversalEmbedderInternal{.instance = std::move(*cpp_embedder)};
  return absl::OkStatus();
}

absl::Status CppUniversalEmbedderEmbedText(MpUniversalEmbedderPtr embedder,
                                           absl::string_view utf8_str,
                                           MpUniversalEmbedderResult* result) {
  auto cpp_embedder = GetCppEmbedder(embedder);
  ABSL_ASSIGN_OR_RETURN(EmbeddingResult cpp_result,
                        cpp_embedder->EmbedText(utf8_str));
  CppConvertToEmbeddingResult(cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppUniversalEmbedderEmbedImage(MpUniversalEmbedderPtr embedder,
                                            absl::string_view image_bytes,
                                            MpUniversalEmbedderResult* result) {
  auto cpp_embedder = GetCppEmbedder(embedder);
  ABSL_ASSIGN_OR_RETURN(EmbeddingResult cpp_result,
                        cpp_embedder->EmbedImage(image_bytes));
  CppConvertToEmbeddingResult(cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppUniversalEmbedderEmbedMpImage(
    MpUniversalEmbedderPtr embedder, MpImagePtr image,
    MpUniversalEmbedderResult* result) {
  auto cpp_embedder = GetCppEmbedder(embedder);
  ABSL_ASSIGN_OR_RETURN(EmbeddingResult cpp_result,
                        cpp_embedder->EmbedImage(ToImage(image)));
  CppConvertToEmbeddingResult(cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppUniversalEmbedderEmbedAudio(MpUniversalEmbedderPtr embedder,
                                            const float* audio_data,
                                            int audio_data_size,
                                            MpUniversalEmbedderResult* result) {
  auto cpp_embedder = GetCppEmbedder(embedder);
  std::vector<float> audio_vector(audio_data, audio_data + audio_data_size);
  ABSL_ASSIGN_OR_RETURN(EmbeddingResult cpp_result,
                        cpp_embedder->EmbedAudio(audio_vector));
  CppConvertToEmbeddingResult(cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppUniversalEmbedderEmbedMpAudioData(
    MpUniversalEmbedderPtr embedder, const MpAudioData* audio_data,
    MpUniversalEmbedderResult* result) {
  auto cpp_embedder = GetCppEmbedder(embedder);
  std::vector<float> audio_vector(
      audio_data->audio_data,
      audio_data->audio_data + audio_data->audio_data_size);
  ABSL_ASSIGN_OR_RETURN(EmbeddingResult cpp_result,
                        cpp_embedder->EmbedAudio(audio_vector));
  CppConvertToEmbeddingResult(cpp_result, result);
  return absl::OkStatus();
}

void CppUniversalEmbedderCloseResult(MpUniversalEmbedderResult* result) {
  CppCloseEmbeddingResult(result);
}

absl::Status CppUniversalEmbedderClose(MpUniversalEmbedderPtr embedder) {
  auto cpp_embedder = GetCppEmbedder(embedder);
  auto result = cpp_embedder->Close();
  delete embedder;
  return result;
}

}  // namespace mediapipe::tasks::c::retrieval::universal_embedder

extern "C" {

MP_EXPORT MpStatus
MpUniversalEmbedderCreate(struct MpUniversalEmbedderOptions* options,
                          MpUniversalEmbedderPtr* embedder, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::universal_embedder::
      CppUniversalEmbedderCreate(*options, embedder);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpUniversalEmbedderEmbedText(
    MpUniversalEmbedderPtr embedder, const char* utf8_str,
    MpUniversalEmbedderResult* result, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::universal_embedder::
      CppUniversalEmbedderEmbedText(embedder, utf8_str ? utf8_str : "", result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpUniversalEmbedderEmbedImage(
    MpUniversalEmbedderPtr embedder, const char* image_bytes,
    int image_bytes_size, MpUniversalEmbedderResult* result, char** error_msg) {
  absl::string_view bytes_view(image_bytes, image_bytes_size);
  absl::Status status = mediapipe::tasks::c::retrieval::universal_embedder::
      CppUniversalEmbedderEmbedImage(embedder, bytes_view, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpUniversalEmbedderEmbedMpImage(
    MpUniversalEmbedderPtr embedder, MpImagePtr image,
    MpUniversalEmbedderResult* result, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::universal_embedder::
      CppUniversalEmbedderEmbedMpImage(embedder, image, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpUniversalEmbedderEmbedAudio(
    MpUniversalEmbedderPtr embedder, const float* audio_data,
    int audio_data_size, MpUniversalEmbedderResult* result, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::universal_embedder::
      CppUniversalEmbedderEmbedAudio(embedder, audio_data, audio_data_size,
                                     result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpUniversalEmbedderEmbedMpAudioData(
    MpUniversalEmbedderPtr embedder, const struct MpAudioData* audio_data,
    MpUniversalEmbedderResult* result, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::universal_embedder::
      CppUniversalEmbedderEmbedMpAudioData(embedder, audio_data, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT void MpUniversalEmbedderCloseResult(
    MpUniversalEmbedderResult* result) {
  mediapipe::tasks::c::retrieval::universal_embedder::
      CppUniversalEmbedderCloseResult(result);
}

MP_EXPORT MpStatus MpUniversalEmbedderClose(MpUniversalEmbedderPtr embedder,
                                            char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::universal_embedder::
      CppUniversalEmbedderClose(embedder);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

}  // extern "C"
