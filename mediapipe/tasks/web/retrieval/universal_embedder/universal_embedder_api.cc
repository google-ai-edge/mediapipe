/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

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

#ifndef __EMSCRIPTEN__
#error "This is web-only code, but was built for a non-web target platform."
#endif  // __EMSCRIPTEN__

#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/em_macros.h>
#include <emscripten/val.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/retrieval/universal_embedder/universal_embedder.h"

namespace mediapipe::tasks::retrieval::universal_embedder {
namespace {

void DisableLogging() {
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfinity);
}

// Throws a JavaScript Error exception across the Wasm boundary.
[[noreturn]] void ReportAndThrowError(absl::string_view message) {
  emscripten::val::global("Error").new_(std::string(message)).throw_();
#if defined(__GNUC__) || defined(__clang__)
  __builtin_unreachable();
#endif
}

// Validates the opaque handle and casts it back to the UniversalEmbedder
// pointer.
UniversalEmbedder* GetEmbedderFromHandleOrThrow(intptr_t handle) {
  auto* embedder = reinterpret_cast<UniversalEmbedder*>(handle);
  if (!embedder) {
    ReportAndThrowError("Invalid or closed UniversalEmbedder handle.");
  }
  return embedder;
}

// Validates the opaque builder handle and casts it back to the Part vector.
std::vector<UniversalEmbedder::Part>* GetContentBuilderFromHandleOrThrow(
    intptr_t handle) {
  auto* parts = reinterpret_cast<std::vector<UniversalEmbedder::Part>*>(handle);
  if (!parts) {
    ReportAndThrowError("Invalid ContentBuilder handle.");
  }
  return parts;
}

// Converts C++ EmbeddingResult to a JavaScript Float32Array.
emscripten::val EmbeddingResultToJs(
    const components::containers::EmbeddingResult& result) {
  if (result.embeddings.empty()) {
    return emscripten::val::array();
  }
  const auto& vec = result.embeddings[0].float_embedding;
  return emscripten::val::global("Float32Array")
      .new_(emscripten::typed_memory_view(vec.size(), vec.data()));
}

// Extracts raw bytes from a JavaScript string or Uint8Array.
std::string ExtractBytesFromJs(emscripten::val bytes_val) {
  if (bytes_val.isString()) {
    return bytes_val.as<std::string>();
  }
  const size_t length = bytes_val["byteLength"].as<size_t>();
  if (length == 0) {
    return "";
  }
  std::string bytes;
  bytes.resize(length);
  emscripten::val memory_view = emscripten::val(emscripten::typed_memory_view(
      length, reinterpret_cast<uint8_t*>(bytes.data())));
  memory_view.call<void>("set", bytes_val);
  return bytes;
}

// Extracts float samples from a JavaScript array or Float32Array.
std::vector<float> ExtractFloatArrayFromJs(emscripten::val float_array_val) {
  if (float_array_val.isArray()) {
    return emscripten::vecFromJSArray<float>(float_array_val);
  }
  const size_t length = float_array_val["length"].as<size_t>();
  if (length == 0) {
    return {};
  }
  std::vector<float> float_samples(length);
  emscripten::val memory_view = emscripten::val(
      emscripten::typed_memory_view(length, float_samples.data()));
  memory_view.call<void>("set", float_array_val);
  return float_samples;
}

}  // namespace

// Creates the native UniversalEmbedder instance and returns its pointer as an
// opaque handle.
intptr_t CreateUniversalEmbedder(const std::string& model_path,
                                 bool l2_normalize, bool use_gpu_backend,
                                 int max_input_length,
                                 int vision_tokens_per_image,
                                 int activation_data_type) {
  DisableLogging();
  auto options = std::make_unique<UniversalEmbedderOptions>();
  options->base_options.model_asset_path = model_path;
  if (use_gpu_backend) {
    options->base_options.delegate = tasks::core::BaseOptions::GPU;
  } else {
    options->base_options.delegate = tasks::core::BaseOptions::CPU;
  }
  options->l2_normalize = l2_normalize;
  if (max_input_length > 0) {
    options->max_input_length = max_input_length;
  }
  if (vision_tokens_per_image > 0) {
    options->vision_tokens_per_image = vision_tokens_per_image;
  }
  switch (activation_data_type) {
    case 0:
      // Default: do not set activation_data_type.
      break;
    case 1:
      options->activation_data_type = ActivationDataType::FLOAT32;
      break;
    case 2:
      options->activation_data_type = ActivationDataType::FLOAT16;
      break;
    case 3:
      options->activation_data_type = ActivationDataType::INT16;
      break;
    case 4:
      options->activation_data_type = ActivationDataType::INT8;
      break;
    default:
      ReportAndThrowError(absl::StrCat("Unsupported activation data type: ",
                                       activation_data_type));
  }

  auto embedder_or = UniversalEmbedder::Create(std::move(options));
  if (!embedder_or.ok()) {
    ReportAndThrowError(embedder_or.status().message());
  }
  return reinterpret_cast<intptr_t>(embedder_or.value().release());
}

emscripten::val UniversalEmbedderEmbedText(intptr_t handle,
                                           const std::string& text) {
  DisableLogging();
  auto* embedder = GetEmbedderFromHandleOrThrow(handle);
  auto res_or = embedder->EmbedText(text);
  if (!res_or.ok()) {
    ReportAndThrowError(res_or.status().message());
  }
  return EmbeddingResultToJs(*res_or);
}

emscripten::val UniversalEmbedderEmbedImage(intptr_t handle,
                                            emscripten::val image_val) {
  DisableLogging();
  auto* embedder = GetEmbedderFromHandleOrThrow(handle);
  auto res_or = embedder->EmbedImage(ExtractBytesFromJs(image_val));
  if (!res_or.ok()) {
    ReportAndThrowError(res_or.status().message());
  }
  return EmbeddingResultToJs(*res_or);
}

emscripten::val UniversalEmbedderEmbedAudio(intptr_t handle,
                                            emscripten::val audio_samples_val) {
  DisableLogging();
  auto* embedder = GetEmbedderFromHandleOrThrow(handle);
  auto res_or =
      embedder->EmbedAudio(ExtractFloatArrayFromJs(audio_samples_val));
  if (!res_or.ok()) {
    ReportAndThrowError(res_or.status().message());
  }
  return EmbeddingResultToJs(*res_or);
}

intptr_t UniversalEmbedderCreateContentBuilder() {
  return reinterpret_cast<intptr_t>(new std::vector<UniversalEmbedder::Part>());
}

void UniversalEmbedderFreeContentBuilder(intptr_t builder_handle) {
  if (builder_handle != 0) {
    delete reinterpret_cast<std::vector<UniversalEmbedder::Part>*>(
        builder_handle);
  }
}

void UniversalEmbedderBuilderAddText(intptr_t builder_handle,
                                     const std::string& text) {
  DisableLogging();
  auto* parts = GetContentBuilderFromHandleOrThrow(builder_handle);
  parts->push_back(UniversalEmbedder::TextPart{text});
}

void UniversalEmbedderBuilderAddImage(intptr_t builder_handle,
                                      emscripten::val image_val) {
  DisableLogging();
  auto* parts = GetContentBuilderFromHandleOrThrow(builder_handle);
  parts->push_back(UniversalEmbedder::ImagePart{ExtractBytesFromJs(image_val)});
}

void UniversalEmbedderBuilderAddAudio(intptr_t builder_handle,
                                      emscripten::val audio_samples_val) {
  DisableLogging();
  auto* parts = GetContentBuilderFromHandleOrThrow(builder_handle);
  parts->push_back(
      UniversalEmbedder::AudioPart{ExtractFloatArrayFromJs(audio_samples_val)});
}

emscripten::val UniversalEmbedderExecuteEmbedContent(intptr_t handle,
                                                     intptr_t builder_handle) {
  DisableLogging();
  auto* embedder = GetEmbedderFromHandleOrThrow(handle);
  auto* parts = GetContentBuilderFromHandleOrThrow(builder_handle);

  auto res_or = embedder->EmbedContent(*parts);
  if (!res_or.ok()) {
    ReportAndThrowError(res_or.status().message());
  }
  return EmbeddingResultToJs(*res_or);
}

void UniversalEmbedderClose(intptr_t handle) {
  DisableLogging();
  auto* embedder = reinterpret_cast<UniversalEmbedder*>(handle);
  if (embedder) {
    embedder->Close().IgnoreError();
    delete embedder;
  }
}

EMSCRIPTEN_BINDINGS(universal_embedder_api) {
  emscripten::function("createUniversalEmbedder", &CreateUniversalEmbedder);
  emscripten::function("universalEmbedder_embedText",
                       &UniversalEmbedderEmbedText);
  emscripten::function("universalEmbedder_embedImage",
                       &UniversalEmbedderEmbedImage);
  emscripten::function("universalEmbedder_embedAudio",
                       &UniversalEmbedderEmbedAudio);
  emscripten::function("universalEmbedder_createContentBuilder",
                       &UniversalEmbedderCreateContentBuilder);
  emscripten::function("universalEmbedder_builderAddText",
                       &UniversalEmbedderBuilderAddText);
  emscripten::function("universalEmbedder_builderAddImage",
                       &UniversalEmbedderBuilderAddImage);
  emscripten::function("universalEmbedder_builderAddAudio",
                       &UniversalEmbedderBuilderAddAudio);
  emscripten::function("universalEmbedder_executeEmbedContent",
                       &UniversalEmbedderExecuteEmbedContent);
  emscripten::function("universalEmbedder_freeContentBuilder",
                       &UniversalEmbedderFreeContentBuilder);
  emscripten::function("universalEmbedder_close", &UniversalEmbedderClose);
}

}  // namespace mediapipe::tasks::retrieval::universal_embedder
