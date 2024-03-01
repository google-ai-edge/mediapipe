#ifndef MEDIAPIPE_TASKS_JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_CORE_JNI_LLM_INFERENCE_ENGINE_H_
#define MEDIAPIPE_TASKS_JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_CORE_JNI_LLM_INFERENCE_ENGINE_H_

#include <cstddef>
#include <cstdint>

#ifndef ODML_EXPORT
#define ODML_EXPORT __attribute__((visibility("default")))
#endif  // ODML_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

typedef void LlmInferenceEngine_Session;

// LlmSessionConfig configures how to execute the model.
typedef struct {
  // Path to the tflite flatbuffer file.
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

  // Randomness when decoding the next token, 0.0f means greedy decoding.
  float temperature;

  // Random seed for sampling tokens.
  size_t random_seed;
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
ODML_EXPORT LlmInferenceEngine_Session* LlmInferenceEngine_CreateSession(
    const LlmSessionConfig* session_config);

// Free the session, will wait until graph is done executing.
ODML_EXPORT void LlmInferenceEngine_Session_Delete(
    LlmInferenceEngine_Session* session);

// Return the generated output in sync mode.
ODML_EXPORT LlmResponseContext LlmInferenceEngine_Session_PredictSync(
    LlmInferenceEngine_Session* session, const char* input);

// Run callback function in async mode.
// The callback context can be a pointer to any user defined data structure as
// it is passed to the callback unmodified.
ODML_EXPORT void LlmInferenceEngine_Session_PredictAsync(
    LlmInferenceEngine_Session* session, void* callback_context,
    const char* input,
    void (*callback)(void* callback_context,
                     const LlmResponseContext response_context));

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_CORE_JNI_LLM_INFERENCE_ENGINE_H_
