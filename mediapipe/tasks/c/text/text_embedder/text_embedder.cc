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

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "mediapipe/tasks/c/components/containers/embedding_result.h"
#include "mediapipe/tasks/c/components/containers/embedding_result_converter.h"
#include "mediapipe/tasks/c/components/processors/embedder_options_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/text/text_embedder/text_embedder.h"

namespace mediapipe::tasks::c::text::text_embedder {

namespace {

using ::mediapipe::tasks::c::components::containers::CppCloseEmbeddingResult;
using ::mediapipe::tasks::c::components::containers::CppConvertToCppEmbedding;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToEmbeddingResult;
using ::mediapipe::tasks::c::components::processors::
    CppConvertToEmbedderOptions;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::text::text_embedder::TextEmbedder;
typedef ::mediapipe::tasks::components::containers::Embedding CppEmbedding;

int CppProcessError(absl::Status status, char** error_msg) {
  if (error_msg) {
    *error_msg = strdup(status.ToString().c_str());
  }
  return status.raw_code();
}

}  // namespace

TextEmbedder* CppTextEmbedderCreate(const TextEmbedderOptions& options,
                                    char** error_msg) {
  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::text::text_embedder::TextEmbedderOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToEmbedderOptions(options.embedder_options,
                              &cpp_options->embedder_options);

  auto embedder = TextEmbedder::Create(std::move(cpp_options));
  if (!embedder.ok()) {
    ABSL_LOG(ERROR) << "Failed to create TextEmbedder: " << embedder.status();
    CppProcessError(embedder.status(), error_msg);
    return nullptr;
  }
  return embedder->release();
}

int CppTextEmbedderEmbed(void* embedder, const char* utf8_str,
                         TextEmbedderResult* result, char** error_msg) {
  auto cpp_embedder = static_cast<TextEmbedder*>(embedder);
  auto cpp_result = cpp_embedder->Embed(utf8_str);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Embedding extraction failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToEmbeddingResult(*cpp_result, result);
  return 0;
}

void CppTextEmbedderCloseResult(TextEmbedderResult* result) {
  CppCloseEmbeddingResult(result);
}

int CppTextEmbedderClose(void* embedder, char** error_msg) {
  auto cpp_embedder = static_cast<TextEmbedder*>(embedder);
  auto result = cpp_embedder->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close TextEmbedder: " << result;
    return CppProcessError(result, error_msg);
  }
  delete cpp_embedder;
  return 0;
}

int CppTextEmbedderCosineSimilarity(const Embedding* u, const Embedding* v,
                                    double* similarity, char** error_msg) {
  CppEmbedding cpp_u;
  CppConvertToCppEmbedding(*u, &cpp_u);
  CppEmbedding cpp_v;
  CppConvertToCppEmbedding(*v, &cpp_v);
  auto status_or_similarity =
      mediapipe::tasks::text::text_embedder::TextEmbedder::CosineSimilarity(
          cpp_u, cpp_v);
  if (status_or_similarity.ok()) {
    *similarity = status_or_similarity.value();
  } else {
    ABSL_LOG(ERROR) << "Cannot compute cosine similarity.";
    return CppProcessError(status_or_similarity.status(), error_msg);
  }
  return 0;
}

}  // namespace mediapipe::tasks::c::text::text_embedder

extern "C" {

void* text_embedder_create(struct TextEmbedderOptions* options,
                           char** error_msg) {
  return mediapipe::tasks::c::text::text_embedder::CppTextEmbedderCreate(
      *options, error_msg);
}

int text_embedder_embed(void* embedder, const char* utf8_str,
                        TextEmbedderResult* result, char** error_msg) {
  return mediapipe::tasks::c::text::text_embedder::CppTextEmbedderEmbed(
      embedder, utf8_str, result, error_msg);
}

void text_embedder_close_result(TextEmbedderResult* result) {
  mediapipe::tasks::c::text::text_embedder::CppTextEmbedderCloseResult(result);
}

int text_embedder_close(void* embedder, char** error_ms) {
  return mediapipe::tasks::c::text::text_embedder::CppTextEmbedderClose(
      embedder, error_ms);
}

int text_embedder_cosine_similarity(const Embedding* u, const Embedding* v,
                                    double* similarity, char** error_msg) {
  return mediapipe::tasks::c::text::text_embedder::
      CppTextEmbedderCosineSimilarity(u, v, similarity, error_msg);
}

}  // extern "C"
