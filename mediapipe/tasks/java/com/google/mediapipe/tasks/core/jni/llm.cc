// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni/llm.h"

#include <jni.h>

#include <string>

#include "absl/status/status.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/jni_util.h"
#include "mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni/proto/llm_options.pb.h"
#include "mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni/proto/llm_response_context.pb.h"
#include "odml/infra/genai/inference/c/llm_inference_engine.h"

namespace {

using LlmModelParametersProto = mediapipe::tasks::core::jni::LlmModelParameters;
using LlmSessionConfigProto = mediapipe::tasks::core::jni::LlmSessionConfig;
using LlmResponseContextProto = mediapipe::tasks::core::jni::LlmResponseContext;
using mediapipe::android::JStringToStdString;
using mediapipe::android::ThrowIfError;

LlmModelParameters ParseModelParameters(void* bytes, int size) {
  LlmModelParametersProto input;
  input.ParseFromArray(bytes, size);

  LlmModelParameters output;

  switch (input.model_type()) {
    case kFalcon1B:
      output.model_type = kFalcon1B;
      break;
    case kGMini2B:
      output.model_type = kGMini2B;
      break;
    default:
      output.model_type = kUNKNOWN_MODEL_TYPE;
  }

  output.model_path = strdup(input.model_directory().c_str());

  switch (input.attention_type()) {
    case kMHA:
      output.attention_type = kMHA;
      break;
    case kMQA:
      output.attention_type = kMQA;
      break;
    default:
      output.attention_type = kMHA;
  }

  output.start_token_id = input.start_token_id();

  const char** stop_tokens = new const char*[input.stop_tokens_size()];
  for (int i = 0; i < input.stop_tokens_size(); ++i) {
    stop_tokens[i] = strdup(input.stop_tokens(i).c_str());
  }
  output.stop_tokens = stop_tokens;
  output.stop_tokens_size = input.stop_tokens_size();

  return output;
}

void CloseModelParameters(LlmModelParameters* model_parameters) {
  delete model_parameters->model_path;
  model_parameters->model_path = nullptr;

  for (int i = 0; i < model_parameters->stop_tokens_size; ++i) {
    delete model_parameters->stop_tokens[i];
  }

  delete[] model_parameters->stop_tokens;
  model_parameters->stop_tokens = nullptr;
  model_parameters->stop_tokens_size = 0;
}

LlmSessionConfig ParseSessionConfig(void* bytes, int size) {
  LlmSessionConfigProto input;
  input.ParseFromArray(bytes, size);

  LlmSessionConfig output;

  switch (input.backend()) {
    case LlmSessionConfigProto::CPU:
      output.backend = kCPU;
      break;
    case LlmSessionConfigProto::GPU:
      output.backend = kGPU;
      break;
    default:
      output.backend = kCPU;
  }

  output.sequence_batch_size = input.sequence_batch_size();
  output.num_decode_tokens = input.num_decode_tokens();
  output.max_sequence_length = input.max_sequence_length();
  output.use_fake_weights = input.use_fake_weights();

  return output;
}

jbyteArray ToByteArray(JNIEnv* env, const LlmResponseContext& context) {
  LlmResponseContextProto output;
  for (int i = 0; i < context.response_count; ++i) {
    output.add_responses(context.response_array[i]);
  }
  std::string output_str = output.SerializeAsString();

  jbyteArray data = env->NewByteArray(output_str.size());
  env->SetByteArrayRegion(data, 0, output_str.size(),
                          reinterpret_cast<const jbyte*>(output_str.data()));
  return data;
}

// A context object that is passed to the callback so that global state can be
// recovered.
typedef struct {
  JNIEnv* env;
  jobject global_callback_ref;
  jmethodID callback_method_id;
} CallbackContext;

void ProcessAsyncResponse(void* callback_context_handle,
                          const LlmResponseContext respone_context) {
  CallbackContext* callback_context =
      reinterpret_cast<CallbackContext*>(callback_context_handle);
  JNIEnv* env = callback_context->env;

  const jbyteArray response_context_bytes = ToByteArray(env, respone_context);

  env->CallVoidMethod(callback_context->global_callback_ref,
                      callback_context->callback_method_id,
                      response_context_bytes);
}

}  // namespace

JNIEXPORT jlong JNICALL JNI_METHOD(nativeCreateSession)(
    JNIEnv* env, jclass thiz, jbyteArray model_parameters_bytes,
    jbyteArray session_config_bytes) {
  // Retrieve the LLM model parameters.
  jbyte* model_parameters_ref =
      env->GetByteArrayElements(model_parameters_bytes, nullptr);
  int model_parameters_size = env->GetArrayLength(model_parameters_bytes);
  LlmModelParameters model_parameters = ParseModelParameters(
      reinterpret_cast<void*>(model_parameters_ref), model_parameters_size);
  env->ReleaseByteArrayElements(model_parameters_bytes, model_parameters_ref,
                                JNI_ABORT);

  // Retrieve the LLM session configuration.
  jbyte* session_config_ref =
      env->GetByteArrayElements(session_config_bytes, nullptr);
  int session_config_size = env->GetArrayLength(session_config_bytes);
  LlmSessionConfig session_config = ParseSessionConfig(
      reinterpret_cast<void*>(session_config_ref), session_config_size);
  env->ReleaseByteArrayElements(session_config_bytes, session_config_ref,
                                JNI_ABORT);

  void* session =
      LlmInferenceEngine_CreateSession(&model_parameters, &session_config);
  CloseModelParameters(&model_parameters);

  return reinterpret_cast<jlong>(session);
}

JNIEXPORT void JNICALL JNI_METHOD(nativeDeleteSession)(JNIEnv* env, jclass thiz,
                                                       jlong session_handle) {
  LlmInferenceEngine_Session_Delete(reinterpret_cast<void*>(session_handle));
}

JNIEXPORT jbyteArray JNICALL JNI_METHOD(nativePredictSync)(JNIEnv* env,
                                                           jclass thiz,
                                                           jlong session_handle,
                                                           jstring input) {
  std::string input_str = JStringToStdString(env, input);
  LlmResponseContext response_context = LlmInferenceEngine_Session_PredictSync(
      reinterpret_cast<void*>(session_handle), input_str.c_str());
  const jbyteArray response_bytes = ToByteArray(env, response_context);
  LlmInferenceEngine_CloseResponseContext(&response_context);
  return response_bytes;
}

JNIEXPORT jlong JNICALL JNI_METHOD(nativeRegisterCallback)(JNIEnv* env,
                                                           jclass thiz,
                                                           jobject callback) {
  auto callback_class = env->GetObjectClass(callback);
  auto global_callback_ref = env->NewGlobalRef(callback);
  if (!global_callback_ref) {
    ThrowIfError(env, absl::InternalError("Failed to allocate callback"));
    return 0;
  }
  auto callback_method_id =
      env->GetMethodID(callback_class, "onAsyncResponse", "([B)V");

  CallbackContext* callback_context =
      new CallbackContext{env, global_callback_ref, callback_method_id};
  return reinterpret_cast<jlong>(callback_context);
}

JNIEXPORT void JNICALL JNI_METHOD(nativeRemoveCallback)(
    JNIEnv* env, jclass thiz, jlong callback_context_handle) {
  CallbackContext* callback_context =
      reinterpret_cast<CallbackContext*>(callback_context_handle);
  env->DeleteGlobalRef(callback_context->global_callback_ref);
  delete reinterpret_cast<CallbackContext*>(callback_context);
}

JNIEXPORT void JNICALL
JNI_METHOD(nativePredictAsync)(JNIEnv* env, jclass thiz, jlong session_handle,
                               jlong callback_context_handle, jstring input) {
  std::string input_str = JStringToStdString(env, input);
  LlmInferenceEngine_Session_PredictAsync(
      reinterpret_cast<void*>(session_handle),
      reinterpret_cast<void*>(callback_context_handle), input_str.c_str(),
      &ProcessAsyncResponse);
}
