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
using ::mediapipe::tasks::c::core::ToMpStatus;
using ::mediapipe::tasks::text::text_embedder::TextEmbedder;
typedef ::mediapipe::tasks::components::containers::Embedding CppEmbedding;

TextEmbedder* GetCppEmbedder(MpTextEmbedderPtr wrapper) {
  ABSL_CHECK(wrapper != nullptr) << "TextEmbedder is null.";
  return wrapper->instance.get();
}

}  // namespace

absl::Status CppTextEmbedderCreate(const TextEmbedderOptions& options,
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

absl::Status CppTextEmbedderEmbed(MpTextEmbedderPtr embedder,
                                  const char* utf8_str,
                                  TextEmbedderResult* result) {
  auto cpp_embedder = GetCppEmbedder(embedder);
  auto cpp_result = cpp_embedder->Embed(utf8_str);
  if (!cpp_result.ok()) {
    return cpp_result.status();
  }
  CppConvertToEmbeddingResult(*cpp_result, result);
  return absl::OkStatus();
}

void CppTextEmbedderCloseResult(TextEmbedderResult* result) {
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

absl::Status CppTextEmbedderCosSimilarity(const Embedding* u,
                                          const Embedding* v,
                                          double* similarity) {
  CppEmbedding cpp_u;
  CppConvertToCppEmbedding(*u, &cpp_u);
  CppEmbedding cpp_v;
  CppConvertToCppEmbedding(*v, &cpp_v);
  auto status_or_similarity =
      mediapipe::tasks::text::text_embedder::TextEmbedder::CosineSimilarity(
          cpp_u, cpp_v);
  if (status_or_similarity.ok()) {
    *similarity = status_or_similarity.value();
  }
  return status_or_similarity.status();
}

}  // namespace mediapipe::tasks::c::text::text_embedder

extern "C" {

MP_EXPORT MpStatus MpTextEmbedderCreate(struct TextEmbedderOptions* options,
                                        MpTextEmbedderPtr* embedder,
                                        char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::text::text_embedder::CppTextEmbedderCreate(*options,
                                                                      embedder);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpTextEmbedderEmbed(MpTextEmbedderPtr embedder,
                                       const char* utf8_str,
                                       TextEmbedderResult* result,
                                       char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::text::text_embedder::CppTextEmbedderEmbed(
          embedder, utf8_str, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT void MpTextEmbedderCloseResult(TextEmbedderResult* result) {
  mediapipe::tasks::c::text::text_embedder::CppTextEmbedderCloseResult(result);
}

MP_EXPORT MpStatus MpTextEmbedderClose(MpTextEmbedderPtr embedder,
                                       char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::text::text_embedder::CppTextEmbedderClose(embedder);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpTextEmbedderCosSimilarity(const Embedding* u,
                                               const Embedding* v,
                                               double* similarity,
                                               char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::text::text_embedder::CppTextEmbedderCosSimilarity(
          u, v, similarity);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

}  // extern "C"
