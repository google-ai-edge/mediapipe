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

// Backend to execute the large language model.
enum LlmBackend {
  // CPU
  kCPU,

  // GPU
  kGPU,
};

// LlmSessionConfig configures how to execute the model.
typedef struct {
  // Path to the tflite model file.
  const char* model_path;

  // Select a supported backend.
  enum LlmBackend backend;

  // Sequence batch size for encoding. Used by GPU only.
  size_t sequence_batch_size;

  // Number of decode steps per sync. Used by GPU only.
  size_t num_decode_steps_per_sync;

  // Maximum sequence length stands for the total number of tokens from input
  // and output.
  size_t max_sequence_length;
} LlmSessionConfig;

// LlmResponseContext is the return type for
// LlmInferenceEngine_Session_PredictSync.
typedef struct {
  // An array of string. The size of the array depends on the number of
  // responses.
  char** response_array;

  // Number of responses.
  int response_count;
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
