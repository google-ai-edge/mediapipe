/* Copyright 2023 The MediaPipe Authors.

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

#include "mediapipe/tasks/c/text/text_embedder/text_embedder.h"

#include <memory>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/c/components/containers/embedding_result.h"
#include "mediapipe/tasks/c/components/containers/embedding_result_converter.h"
#include "mediapipe/tasks/c/components/processors/embedder_options_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/text/text_embedder/text_embedder.h"

struct MpTextEmbedderInternal {
  std::unique_ptr<::mediapipe::tasks::text::text_embedder::TextEmbedder>
      instance;
};

namespace mediapipe::tasks::c::text::text_embedder {

namespace {

using ::mediapipe::tasks::c::components::containers::CppCloseEmbeddingResult;
using ::mediapipe::tasks::c::components::containers::CppConvertToCppEmbedding;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToEmbeddingResult;
using ::mediapipe::tasks::c::components::processors::
    CppConvertToEmbedderOptions;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::components::containers::EmbeddingResult;
using ::mediapipe::tasks::text::text_embedder::EmbeddingType;
using ::mediapipe::tasks::text::text_embedder::TextEmbedder;
using ::mediapipe::tasks::text::text_embedder::TextFormatContext;
using ::mediapipe::tasks::text::text_embedder::TextRole;

typedef ::mediapipe::tasks::components::containers::Embedding CppEmbedding;

TextEmbedder* GetCppEmbedder(MpTextEmbedderPtr wrapper) {
  ABSL_CHECK(wrapper != nullptr) << "TextEmbedder is null.";
  return wrapper->instance.get();
}

absl::StatusOr<EmbeddingType> CppConvertToEmbeddingType(
    MpTextEmbedderEmbeddingType embedding_type) {
  switch (embedding_type) {
    case MP_TEXT_EMBEDDER_EMBEDDING_TYPE_RETRIEVAL_QUERY:
      return EmbeddingType::RETRIEVAL_QUERY;
    case MP_TEXT_EMBEDDER_EMBEDDING_TYPE_RETRIEVAL_DOCUMENT:
      return EmbeddingType::RETRIEVAL_DOCUMENT;
    case MP_TEXT_EMBEDDER_EMBEDDING_TYPE_SEMANTIC_SIMILARITY:
      return EmbeddingType::SEMANTIC_SIMILARITY;
    case MP_TEXT_EMBEDDER_EMBEDDING_TYPE_CLASSIFICATION:
      return EmbeddingType::CLASSIFICATION;
    case MP_TEXT_EMBEDDER_EMBEDDING_TYPE_CLUSTERING:
      return EmbeddingType::CLUSTERING;
    case MP_TEXT_EMBEDDER_EMBEDDING_TYPE_QUESTION_ANSWERING:
      return EmbeddingType::QUESTION_ANSWERING;
    case MP_TEXT_EMBEDDER_EMBEDDING_TYPE_FACT_CHECKING:
      return EmbeddingType::FACT_CHECKING;
    case MP_TEXT_EMBEDDER_EMBEDDING_TYPE_CODE_RETRIEVAL:
      return EmbeddingType::CODE_RETRIEVAL;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unhandled MpTextEmbedderEmbeddingType: ", embedding_type));
}

absl::StatusOr<TextRole> CppConvertToTextRole(MpTextEmbedderRole text_role) {
  switch (text_role) {
    case MP_TEXT_EMBEDDER_ROLE_QUERY:
      return TextRole::kQuery;
    case MP_TEXT_EMBEDDER_ROLE_DOCUMENT:
      return TextRole::kDocument;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unhandled MpTextEmbedderRole: ", text_role));
}

absl::StatusOr<TextFormatContext> CppConvertToTextFormatContext(
    const struct MpTextEmbedderFormatContext* c_format_context) {
  TextFormatContext cpp_format_context;
  MP_ASSIGN_OR_RETURN(cpp_format_context.task_type,
                      CppConvertToEmbeddingType(c_format_context->task_type));
  MP_ASSIGN_OR_RETURN(cpp_format_context.role,
                      CppConvertToTextRole(c_format_context->role));
  if (c_format_context->title) {
    cpp_format_context.title = c_format_context->title;
  }
  return cpp_format_context;
}

}  // namespace

absl::Status CppTextEmbedderCreate(const MpTextEmbedderOptions& options,
                                   MpTextEmbedderPtr* embedder) {
  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::text::text_embedder::TextEmbedderOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToEmbedderOptions(options.embedder_options,
                              &cpp_options->embedder_options);

  auto cpp_embedder = TextEmbedder::Create(std::move(cpp_options));
  if (!cpp_embedder.ok()) {
    return cpp_embedder.status();
  }
  *embedder = new MpTextEmbedderInternal{.instance = std::move(*cpp_embedder)};
  return absl::OkStatus();
}

absl::Status CppTextEmbedderEmbed(
    MpTextEmbedderPtr embedder, const char* utf8_str,
    const struct MpTextEmbedderFormatContext* format_context,
    MpTextEmbedderResult* result) {
  auto cpp_embedder = GetCppEmbedder(embedder);
  EmbeddingResult cpp_result;
  if (format_context) {
    MP_ASSIGN_OR_RETURN(auto cpp_format_context,
                        CppConvertToTextFormatContext(format_context));
    MP_ASSIGN_OR_RETURN(cpp_result,
                        cpp_embedder->Embed(utf8_str, cpp_format_context));
  } else {
    MP_ASSIGN_OR_RETURN(cpp_result, cpp_embedder->Embed(utf8_str));
  }

  CppConvertToEmbeddingResult(cpp_result, result);
  return absl::OkStatus();
}

void CppTextEmbedderCloseResult(MpTextEmbedderResult* result) {
  CppCloseEmbeddingResult(result);
}

absl::Status CppTextEmbedderClose(MpTextEmbedderPtr embedder) {
  auto cpp_embedder = GetCppEmbedder(embedder);
  auto result = cpp_embedder->Close();
  if (!result.ok()) {
    return result;
  }
  delete embedder;
  return absl::OkStatus();
}

absl::Status CppTextEmbedderCosSimilarity(const MpEmbedding* u,
                                          const MpEmbedding* v,
                                          double* similarity) {
  CppEmbedding cpp_u;
  CppConvertToCppEmbedding(*u, &cpp_u);
  CppEmbedding cpp_v;
  CppConvertToCppEmbedding(*v, &cpp_v);
  MP_ASSIGN_OR_RETURN(
      *similarity,
      mediapipe::tasks::text::text_embedder::TextEmbedder::CosineSimilarity(
          cpp_u, cpp_v));
  return absl::OkStatus();
}

}  // namespace mediapipe::tasks::c::text::text_embedder

extern "C" {

MP_EXPORT MpStatus MpTextEmbedderCreate(struct MpTextEmbedderOptions* options,
                                        MpTextEmbedderPtr* embedder,
                                        char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::text::text_embedder::CppTextEmbedderCreate(*options,
                                                                      embedder);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus
MpTextEmbedderEmbed(MpTextEmbedderPtr embedder, const char* utf8_str,
                    const struct MpTextEmbedderFormatContext* format_context,
                    MpTextEmbedderResult* result, char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::text::text_embedder::CppTextEmbedderEmbed(
          embedder, utf8_str, format_context, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT void MpTextEmbedderCloseResult(MpTextEmbedderResult* result) {
  mediapipe::tasks::c::text::text_embedder::CppTextEmbedderCloseResult(result);
}

MP_EXPORT MpStatus MpTextEmbedderClose(MpTextEmbedderPtr embedder,
                                       char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::text::text_embedder::CppTextEmbedderClose(embedder);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpTextEmbedderCosSimilarity(const MpEmbedding* u,
                                               const MpEmbedding* v,
                                               double* similarity,
                                               char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::text::text_embedder::CppTextEmbedderCosSimilarity(
          u, v, similarity);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

}  // extern "C"
