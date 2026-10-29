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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/c/audio/core/common.h"
#include "mediapipe/tasks/c/components/containers/embedding_result_converter.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/c/retrieval/universal_embedder/universal_embedder_internal.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/core/base_options.h"
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

std::optional<tasks::core::BaseOptions::Delegate> ToOptionalDelegate(
    MpDelegate delegate) {
  switch (delegate) {
    case MP_DELEGATE_CPU:
      return tasks::core::BaseOptions::CPU;
    case MP_DELEGATE_GPU:
      return tasks::core::BaseOptions::GPU;
    case MP_DELEGATE_NPU:
      return tasks::core::BaseOptions::NPU;
    case MP_DELEGATE_DEFAULT:
    default:
      return std::nullopt;
  }
}

absl::StatusOr<UniversalEmbedder*> GetCppEmbedder(
    MpUniversalEmbedderPtr wrapper) {
  if (wrapper == nullptr || wrapper->instance == nullptr) {
    return absl::InvalidArgumentError(
        "UniversalEmbedder handle is null. The embedder may have failed to "
        "initialize or has already been closed.");
  }
  return wrapper->instance.get();
}

}  // namespace

absl::Status CppUniversalEmbedderCreate(
    const MpUniversalEmbedderOptions& options,
    MpUniversalEmbedderPtr* embedder) {
  if (embedder == nullptr) {
    return absl::InvalidArgumentError("Output embedder pointer is null.");
  }
  *embedder = nullptr;

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
  cpp_options->text_delegate = ToOptionalDelegate(options.text_delegate);
  cpp_options->vision_delegate = ToOptionalDelegate(options.vision_delegate);
  cpp_options->audio_delegate = ToOptionalDelegate(options.audio_delegate);
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
  ABSL_ASSIGN_OR_RETURN(UniversalEmbedder * cpp_embedder,
                        GetCppEmbedder(embedder));
  if (result == nullptr) {
    return absl::InvalidArgumentError("Output result pointer is null.");
  }
  ABSL_ASSIGN_OR_RETURN(EmbeddingResult cpp_result,
                        cpp_embedder->EmbedText(utf8_str));
  CppConvertToEmbeddingResult(cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppUniversalEmbedderEmbedImage(MpUniversalEmbedderPtr embedder,
                                            absl::string_view image_bytes,
                                            MpUniversalEmbedderResult* result) {
  ABSL_ASSIGN_OR_RETURN(UniversalEmbedder * cpp_embedder,
                        GetCppEmbedder(embedder));
  if (result == nullptr) {
    return absl::InvalidArgumentError("Output result pointer is null.");
  }
  ABSL_ASSIGN_OR_RETURN(EmbeddingResult cpp_result,
                        cpp_embedder->EmbedImage(image_bytes));
  CppConvertToEmbeddingResult(cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppUniversalEmbedderEmbedMpImage(
    MpUniversalEmbedderPtr embedder, MpImagePtr image,
    MpUniversalEmbedderResult* result) {
  ABSL_ASSIGN_OR_RETURN(UniversalEmbedder * cpp_embedder,
                        GetCppEmbedder(embedder));
  if (image == nullptr) {
    return absl::InvalidArgumentError("Input image is null.");
  }
  if (result == nullptr) {
    return absl::InvalidArgumentError("Output result pointer is null.");
  }
  ABSL_ASSIGN_OR_RETURN(EmbeddingResult cpp_result,
                        cpp_embedder->EmbedImage(ToImage(image)));
  CppConvertToEmbeddingResult(cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppUniversalEmbedderEmbedAudio(MpUniversalEmbedderPtr embedder,
                                            const float* audio_data,
                                            int audio_data_size,
                                            MpUniversalEmbedderResult* result) {
  ABSL_ASSIGN_OR_RETURN(UniversalEmbedder * cpp_embedder,
                        GetCppEmbedder(embedder));
  if (audio_data == nullptr && audio_data_size > 0) {
    return absl::InvalidArgumentError("Input audio data is null.");
  }
  if (audio_data_size < 0) {
    return absl::InvalidArgumentError("Input audio data size is negative.");
  }
  if (result == nullptr) {
    return absl::InvalidArgumentError("Output result pointer is null.");
  }
  std::vector<float> audio_vector(audio_data, audio_data + audio_data_size);
  ABSL_ASSIGN_OR_RETURN(EmbeddingResult cpp_result,
                        cpp_embedder->EmbedAudio(audio_vector));
  CppConvertToEmbeddingResult(cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppUniversalEmbedderEmbedMpAudioData(
    MpUniversalEmbedderPtr embedder, const MpAudioData* audio_data,
    MpUniversalEmbedderResult* result) {
  if (audio_data == nullptr) {
    return absl::InvalidArgumentError("Input audio data is null.");
  }
  return CppUniversalEmbedderEmbedAudio(embedder, audio_data->audio_data,
                                        audio_data->audio_data_size, result);
}

absl::Status CppUniversalEmbedderEmbedContent(
    MpUniversalEmbedderPtr embedder, const MpUniversalEmbedderPart* parts,
    int parts_count, MpUniversalEmbedderResult* result) {
  ABSL_ASSIGN_OR_RETURN(UniversalEmbedder * cpp_embedder,
                        GetCppEmbedder(embedder));
  if (result == nullptr) {
    return absl::InvalidArgumentError("Output result pointer is null.");
  }
  if (parts == nullptr && parts_count > 0) {
    return absl::InvalidArgumentError(
        "Parts array is null but parts_count > 0.");
  }
  if (parts_count < 0) {
    return absl::InvalidArgumentError("parts_count is negative.");
  }
  std::vector<UniversalEmbedder::Part> cpp_parts;
  cpp_parts.reserve(parts_count);
  for (int i = 0; i < parts_count; ++i) {
    const auto& part = parts[i];
    switch (part.type) {
      case kMpUniversalEmbedderPartTypeText: {
        if (part.value.text == nullptr) {
          return absl::InvalidArgumentError("Text part string is null.");
        }
        cpp_parts.push_back(
            UniversalEmbedder::TextPart{.text = part.value.text});
        break;
      }
      case kMpUniversalEmbedderPartTypeImage: {
        if (part.value.image.size < 0) {
          return absl::InvalidArgumentError("Image part size is negative.");
        }
        if (part.value.image.bytes == nullptr && part.value.image.size > 0) {
          return absl::InvalidArgumentError("Image part bytes is null.");
        }
        cpp_parts.push_back(UniversalEmbedder::ImagePart{
            .image_bytes = std::string(
                part.value.image.bytes ? part.value.image.bytes : "",
                part.value.image.size)});
        break;
      }
      case kMpUniversalEmbedderPartTypeAudio: {
        if (part.value.audio.size < 0) {
          return absl::InvalidArgumentError("Audio part size is negative.");
        }
        if (part.value.audio.samples == nullptr && part.value.audio.size > 0) {
          return absl::InvalidArgumentError("Audio part samples is null.");
        }
        std::vector<float> audio;
        if (part.value.audio.samples != nullptr && part.value.audio.size > 0) {
          audio.assign(part.value.audio.samples,
                       part.value.audio.samples + part.value.audio.size);
        }
        cpp_parts.push_back(
            UniversalEmbedder::AudioPart{.audio_data = std::move(audio)});
        break;
      }
      default:
        return absl::InvalidArgumentError(
            "Unknown MpUniversalEmbedderPartType.");
    }
  }
  ABSL_ASSIGN_OR_RETURN(EmbeddingResult cpp_result,
                        cpp_embedder->EmbedContent(cpp_parts));
  CppConvertToEmbeddingResult(cpp_result, result);
  return absl::OkStatus();
}

void CppUniversalEmbedderCloseResult(MpUniversalEmbedderResult* result) {
  if (result == nullptr) {
    return;
  }
  CppCloseEmbeddingResult(result);
}

absl::Status CppUniversalEmbedderClose(MpUniversalEmbedderPtr embedder) {
  // Closing a null handle (e.g. after a failed Create) is a no-op.
  if (embedder == nullptr) {
    return absl::OkStatus();
  }
  absl::Status result = absl::OkStatus();
  if (embedder->instance != nullptr) {
    result = embedder->instance->Close();
  }
  delete embedder;
  return result;
}

}  // namespace mediapipe::tasks::c::retrieval::universal_embedder

extern "C" {

MP_EXPORT MpStatus
MpUniversalEmbedderCreate(struct MpUniversalEmbedderOptions* options,
                          MpUniversalEmbedderPtr* embedder, char** error_msg) {
  if (options == nullptr) {
    return mediapipe::tasks::c::core::HandleStatus(
        absl::InvalidArgumentError("MpUniversalEmbedderOptions is null."),
        error_msg);
  }
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

MP_EXPORT MpStatus MpUniversalEmbedderEmbedContent(
    MpUniversalEmbedderPtr embedder,
    const struct MpUniversalEmbedderPart* parts, int parts_count,
    MpUniversalEmbedderResult* result, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::universal_embedder::
      CppUniversalEmbedderEmbedContent(embedder, parts, parts_count, result);
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
