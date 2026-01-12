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

#ifndef MEDIAPIPE_TASKS_C_GENAI_BUNDLER_LLM_BUNDLER_UTILS_H_
#define MEDIAPIPE_TASKS_C_GENAI_BUNDLER_LLM_BUNDLER_UTILS_H_

#include <stdbool.h>

#ifndef MP_EXPORT
#if defined(_MSC_VER)
#define MP_EXPORT __declspec(dllexport)
#else
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // _MSC_VER
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

// Options for generating LLM bundler metadata.
// All char* are expected to be null-terminated UTF-8 strings.
struct LlmBundlerMetadataOptions {
  const char* start_token;
  const char** stop_tokens;
  int num_stop_tokens;
  bool enable_bytes_to_unicode_mapping;
  const char* system_prompt;
  const char* prompt_prefix_user;
  const char* prompt_suffix_user;
  const char* prompt_prefix_model;
  const char* prompt_suffix_model;
  const char* prompt_prefix_system;
  const char* prompt_suffix_system;
  const char* user_role_token;
  const char* system_role_token;
  const char* model_role_token;
  const char* end_role_token;
};

// Creates LlmParameters proto message and serializes it to bytes.
// The caller must free the returned buffer using LlmBundlerFreeMetadata.
//
// The options struct contains pointers to strings that are used to populate
// the metadata. The function returns a pointer to a buffer containing the
// serialized proto message, and metadata_buffer_size will be populated with
// the size of this buffer.
MP_EXPORT const char* MpLlmBundlerGenerateMetadata(
    const struct LlmBundlerMetadataOptions* options, int* metadata_buffer_size);

// Frees the metadata buffer returned by LlmBundlerGenerateMetadata.
MP_EXPORT void MpLlmBundlerFreeMetadata(const char* metadata_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MEDIAPIPE_TASKS_C_GENAI_BUNDLER_LLM_BUNDLER_UTILS_H_
