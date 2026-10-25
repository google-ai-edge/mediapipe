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

#include "mediapipe/tasks/c/text/text_proofreader/text_proofreader.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/cc/text/text_proofreader/text_proofreader.h"

struct MpTextProofreaderInternal {
  std::unique_ptr<::mediapipe::tasks::text::text_proofreader::TextProofreader>
      instance;
};

namespace mediapipe::tasks::c::text::text_proofreader {
namespace {
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::text::text_proofreader::TextProofreader;
using ::mediapipe::tasks::text::text_proofreader::TextProofreaderResult;
using ::mediapipe::tasks::text::text_proofreader::TextProofreaderStreamResult;

TextProofreader* GetCppProofreader(MpTextProofreaderPtr wrapper) {
  ABSL_CHECK(wrapper != nullptr) << "TextProofreader is null.";
  return wrapper->instance.get();
}

void PopulateResult(const TextProofreaderResult& cpp_result,
                    MpTextProofreaderResult* c_result) {
  c_result->proofread_text = strdup(cpp_result.proofread_text.c_str());
  c_result->corrections_count = cpp_result.corrections.size();
  if (c_result->corrections_count > 0) {
    c_result->corrections = new MpCorrection[c_result->corrections_count];
    for (int i = 0; i < c_result->corrections_count; ++i) {
      c_result->corrections[i].type =
          static_cast<MpCorrectionType>(cpp_result.corrections[i].type);
      c_result->corrections[i].text =
          strdup(cpp_result.corrections[i].text.c_str());
    }
  } else {
    c_result->corrections = nullptr;
  }
}

void PopulateStreamingResult(const TextProofreaderStreamResult& cpp_result,
                             MpTextProofreaderStreamResult* c_result,
                             std::vector<MpCorrection>* c_corrections) {
  c_result->chunk =
      cpp_result.chunk.empty() ? nullptr : cpp_result.chunk.c_str();

  // The C API continues to own the memory of MpTextProofreaderStreamResult
  // pointees.
  if (cpp_result.corrections.has_value()) {
    c_corrections->reserve(cpp_result.corrections->size());
    for (const auto& c : *cpp_result.corrections) {
      MpCorrection c_corr;
      c_corr.type = static_cast<MpCorrectionType>(c.type);
      c_corr.text = const_cast<char*>(c.text.c_str());
      c_corrections->push_back(c_corr);
    }
    c_result->corrections_count = c_corrections->size();
    c_result->corrections = c_corrections->data();
  }

  c_result->done = cpp_result.done;
}
}  // namespace

absl::Status CppTextProofreaderCreate(const MpTextProofreaderOptions* options,
                                      MpTextProofreaderPtr* proofreader) {
  if (options == nullptr) {
    return absl::InvalidArgumentError(
        "Failed to create TextProofreader: Options must not be null.");
  }

  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::text::text_proofreader::TextProofreaderOptions>();
  CppConvertToBaseOptions(options->base_options, &cpp_options->base_options);
  cpp_options->max_num_tokens = options->max_num_tokens;
  if (options->cache_dir) {
    cpp_options->cache_dir = options->cache_dir;
  }
  cpp_options->min_log_severity = options->min_log_severity;

  ABSL_ASSIGN_OR_RETURN(auto cpp_proofreader,
                        TextProofreader::Create(std::move(cpp_options)));
  *proofreader =
      new MpTextProofreaderInternal{.instance = std::move(cpp_proofreader)};
  return absl::OkStatus();
}

absl::Status CppTextProofreaderProofread(MpTextProofreaderPtr proofreader,
                                         const char* input_text,
                                         MpTextProofreaderResult** result) {
  if (!proofreader) {
    return absl::InvalidArgumentError(
        "Failed to proofread: Proofreader must not be null.");
  }
  if (!input_text) {
    return absl::InvalidArgumentError(
        "Failed to proofread: Input text must not be null.");
  }
  if (!result) {
    return absl::InvalidArgumentError(
        "Failed to proofread: Result pointer must not be null.");
  }

  TextProofreader* cpp_proofreader = GetCppProofreader(proofreader);
  ABSL_ASSIGN_OR_RETURN(TextProofreaderResult cpp_result,
                        cpp_proofreader->Proofread(input_text));
  *result = new MpTextProofreaderResult();
  PopulateResult(cpp_result, *result);
  return absl::OkStatus();
}

absl::Status CppTextProofreaderProofreadStreaming(
    MpTextProofreaderPtr proofreader, const char* input_text,
    MpTextProofreaderCallback callback, void* user_data) {
  if (!proofreader) {
    return absl::InvalidArgumentError(
        "Failed to proofread streaming: Proofreader must not be null.");
  }
  if (!input_text) {
    return absl::InvalidArgumentError(
        "Failed to proofread streaming: Input text must not be null.");
  }
  if (!callback) {
    return absl::InvalidArgumentError(
        "Failed to proofread streaming: Callback must not be null.");
  }

  TextProofreader* cpp_proofreader = GetCppProofreader(proofreader);

  return cpp_proofreader->ProofreadStreaming(
      input_text,
      [callback,
       user_data](absl::StatusOr<TextProofreaderStreamResult> stream_result) {
        if (!stream_result.ok()) {
          std::string error_str = stream_result.status().ToString();
          callback(user_data, nullptr, error_str.c_str());
          return;
        }

        MpTextProofreaderStreamResult c_result = {};
        std::vector<MpCorrection> c_corrections;
        PopulateStreamingResult(*stream_result, &c_result, &c_corrections);
        callback(user_data, &c_result, nullptr);
      });
}

void CppTextProofreaderCloseResult(MpTextProofreaderResult* result) {
  if (!result) {
    return;
  }
  if (result->proofread_text) {
    free(const_cast<char*>(result->proofread_text));
    result->proofread_text = nullptr;
  }
  if (result->corrections) {
    for (int i = 0; i < result->corrections_count; ++i) {
      if (result->corrections[i].text) {
        free(const_cast<char*>(result->corrections[i].text));
      }
    }
    delete[] result->corrections;
    result->corrections = nullptr;
    result->corrections_count = 0;
  }
  delete result;
}

absl::Status CppTextProofreaderClose(MpTextProofreaderPtr proofreader) {
  delete proofreader;
  return absl::OkStatus();
}

}  // namespace mediapipe::tasks::c::text::text_proofreader

extern "C" {

MP_EXPORT MpStatus
MpTextProofreaderCreate(const struct MpTextProofreaderOptions* options,
                        MpTextProofreaderPtr* proofreader, char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::text::text_proofreader::CppTextProofreaderCreate(
          options, proofreader);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpTextProofreaderProofread(
    MpTextProofreaderPtr proofreader, const char* input_text,
    struct MpTextProofreaderResult** result, char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::text::text_proofreader::CppTextProofreaderProofread(
          proofreader, input_text, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpTextProofreaderProofreadStreaming(
    MpTextProofreaderPtr proofreader, const char* input_text,
    MpTextProofreaderCallback callback, void* user_data, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::text::text_proofreader::
      CppTextProofreaderProofreadStreaming(proofreader, input_text, callback,
                                           user_data);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT void MpTextProofreaderCloseResult(
    struct MpTextProofreaderResult* result) {
  mediapipe::tasks::c::text::text_proofreader::CppTextProofreaderCloseResult(
      result);
}

MP_EXPORT MpStatus MpTextProofreaderClose(MpTextProofreaderPtr proofreader,
                                          char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::text::text_proofreader::CppTextProofreaderClose(
          proofreader);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}
}  // extern "C"
