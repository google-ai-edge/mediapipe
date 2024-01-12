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

// Supported model types.
enum LlmModelType {
  // Unknown
  kUNKNOWN_MODEL_TYPE,

  // Falcon with 1B parameters.
  kFalcon1B,

  // GMini with 2B parameters.
  kGMini2B,
};

// Attention types.
enum LlmAttentionType {
  // Multi-head Attention.
  kMHA,

  // Multi-query Attention.
  kMQA,
};

// Backend to execute the large language model.
enum LlmBackend {
  // CPU
  kCPU,

  // GPU
  kGPU,
};

// LlmModelParameters should accurately describe the model used.
typedef struct {
  // Set a supported model types.
  enum LlmModelType model_type;

  // Path to the directory that contains spm.model and the weight directory.
  const char* model_path;

  // MHA or MQA.
  enum LlmAttentionType attention_type;

  // Start token id will be appended to the query before feeding into the model.
  int start_token_id;

  // Stop token/word that indicates the response is completed.
  const char** stop_tokens;

  // Number of stop tokens.
  size_t stop_tokens_size;
} LlmModelParameters;

// LlmSessionConfig configures how to execute the model.
typedef struct {
  // Select a supported backend.
  enum LlmBackend backend;

  // Sequence batch size for encoding.
  size_t sequence_batch_size;

  // Output batch size for decoding.(for gpu)
  size_t num_decode_tokens;

  // Maximum sequence length stands for the total number of tokens from input
  // and output.
  size_t max_sequence_length;

  // Use fake weights instead of loading from file.
  bool use_fake_weights;
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

// Frees all context within the LlmResponseContext including itself.
ODML_EXPORT void LlmInferenceEngine_CloseResponseContext(
    LlmResponseContext response_context);

// Create a LlmInferenceEngine session for executing a query.
ODML_EXPORT LlmInferenceEngine_Session* LlmInferenceEngine_CreateSession(
    const LlmModelParameters* model_parameters,
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
