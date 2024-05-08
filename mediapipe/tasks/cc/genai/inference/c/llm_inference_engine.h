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

#ifndef MEDIAPIPE_TASKS_GENAI_INFERENCE_C_LLM_INFERENCE_ENGINE_H_
#define MEDIAPIPE_TASKS_GENAI_INFERENCE_C_LLM_INFERENCE_ENGINE_H_

#ifdef __cplusplus
#include <cstddef>
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#endif

#ifndef ODML_EXPORT
#define ODML_EXPORT __attribute__((visibility("default")))
#endif  // ODML_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

typedef void LlmInferenceEngine_Session;

// LlmSessionConfig configures how to execute the model.
typedef struct {
  // Path to the model artifact.
  const char* model_path;

  // Directory path for storing model related tokenizer and cache weights. the
  // user is responsible for providing the directory that can be writable by the
  // program.
  const char* cache_dir;

  // Sequence batch size for encoding. Used by GPU only. Number of input tokens
  // to process at a time for batch processing. Setting this value to 1 means
  // both the encoding and decoding share the same graph of sequence length
  // of 1. Setting this value to 0 means the batch size will be optimized
  // programmatically.
  size_t sequence_batch_size;

  // Number of decode steps per sync. Used by GPU only. The default value is 3.
  size_t num_decode_steps_per_sync;

  // Maximum number of tokens for input and output.
  size_t max_tokens;

  // Top K number of tokens to be sampled from for each decoding step.
  size_t topk;

  // Maximum cumulative probability over the tokens to sample from in each
  // decoding step for top-p / nucleus sampling.
  float topp;

  // Randomness when decoding the next token, 0.0f means greedy decoding.
  float temperature;

  // Random seed for sampling tokens.
  size_t random_seed;

  // Path to the LoRA tflite flatbuffer file. Optional.
  // This is only compatible with GPU models.
  const char* lora_path;
} LlmSessionConfig;

// LlmResponseContext is the return type for
// LlmInferenceEngine_Session_PredictSync.
typedef struct {
  // An array of string. The size of the array depends on the number of
  // responses.
  char** response_array;

  // Number of responses.
  int response_count;

  // Done all outputs for this session.
  bool done;
} LlmResponseContext;

// Frees all context within the LlmResponseContext.
ODML_EXPORT void LlmInferenceEngine_CloseResponseContext(
    LlmResponseContext* response_context);

// Create a LlmInferenceEngine session for executing a query.
ODML_EXPORT int LlmInferenceEngine_CreateSession(
    const LlmSessionConfig* session_config,
    LlmInferenceEngine_Session** session_out, char** error_msg);

// Free the session, will wait until graph is done executing.
ODML_EXPORT void LlmInferenceEngine_Session_Delete(
    LlmInferenceEngine_Session* session);

// Return the generated output in sync mode.
ODML_EXPORT LlmResponseContext LlmInferenceEngine_Session_PredictSync(
    LlmInferenceEngine_Session* session, const char* input);

// Run callback function in async mode.
// The callback will be invoked multiple times until `response_context.done`
// is `true`. You need to invoke `LlmInferenceEngine_CloseResponseContext` after
// each invocation to free memory.
// The callback context can be a pointer to any user defined data structure as
// it is passed to the callback unmodified.
ODML_EXPORT void LlmInferenceEngine_Session_PredictAsync(
    LlmInferenceEngine_Session* session, void* callback_context,
    const char* input,
    void (*callback)(void* callback_context,
                     LlmResponseContext* response_context));

// Tokenizes an input prompt using a pre-existing processor and returns its
// length in tokens. Returns -1 if tokenization fails.
ODML_EXPORT int LlmInferenceEngine_Session_SizeInTokens(
    LlmInferenceEngine_Session* session, const char* input, char** error_msg);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_GENAI_INFERENCE_C_LLM_INFERENCE_ENGINE_H_
