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

#include <cstdlib>
#include <string>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/class_registry.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/jni_util.h"
#include "mediapipe/tasks/cc/genai/inference/c/llm_inference_engine.h"
#include "mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni/proto/llm_options.pb.h"
#include "mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni/proto/llm_response_context.pb.h"

namespace {

using LlmSessionConfigProto = mediapipe::tasks::core::jni::LlmSessionConfig;
using LlmResponseContextProto = mediapipe::tasks::core::jni::LlmResponseContext;
using mediapipe::android::JStringToStdString;
using mediapipe::android::ThrowIfError;
using mediapipe::java::GetJNIEnv;

LlmSessionConfig ParseSessionConfig(void* bytes, int size) {
  LlmSessionConfigProto input;
  input.ParseFromArray(bytes, size);

  LlmSessionConfig output;
  output.model_path = strdup(input.model_path().c_str());
  output.cache_dir = strdup(input.cache_dir().c_str());
  output.sequence_batch_size = input.sequence_batch_size();
  output.num_decode_steps_per_sync = input.num_decode_steps_per_sync();
  output.max_tokens = input.max_tokens();
  output.temperature = input.temperature();
  output.topk = input.topk();
  output.topp = 1.0f;
  output.random_seed = input.random_seed();
  if (input.has_lora_path()) {
    output.lora_path = strdup(input.lora_path().c_str());
  }
  return output;
}

void FreeSessionConfig(LlmSessionConfig* session_config) {
  delete session_config->model_path;
  delete session_config->cache_dir;
  session_config->model_path = nullptr;
  session_config->cache_dir = nullptr;
}

jbyteArray ToByteArray(JNIEnv* env, const LlmResponseContext& context) {
  LlmResponseContextProto output;
  for (int i = 0; i < context.response_count; ++i) {
    output.add_responses(context.response_array[i]);
  }
  output.set_done(context.done);

  std::string serialized_str = output.SerializeAsString();
  jbyteArray data = env->NewByteArray(serialized_str.size());
  env->SetByteArrayRegion(
      data, 0, serialized_str.size(),
      reinterpret_cast<const jbyte*>(serialized_str.data()));
  return data;
}

void ProcessAsyncResponse(void* callback_ref,
                          LlmResponseContext* response_context) {
  jobject object_ref = reinterpret_cast<jobject>(callback_ref);
  JNIEnv* env = GetJNIEnv();
  if (env == nullptr) {
    ABSL_LOG(ERROR)
        << "Failed to retrieve JNI environment. Cannot invoke callback.";
    return;
  }

  jclass class_ref = env->GetObjectClass(object_ref);
  auto& class_registry = mediapipe::android::ClassRegistry::GetInstance();
  std::string method_name = class_registry.GetMethodName(
      "com/google/mediapipe/tasks/core/LlmTaskRunner", "onAsyncResponse");
  jmethodID method_id =
      env->GetMethodID(class_ref, method_name.c_str(), "([B)V");

  const jbyteArray response_context_bytes = ToByteArray(env, *response_context);
  LlmInferenceEngine_CloseResponseContext(response_context);

  env->CallVoidMethod(object_ref, method_id, response_context_bytes);
}

}  // namespace

JNIEXPORT jlong JNICALL JNI_METHOD(nativeCreateSession)(
    JNIEnv* env, jclass thiz, jbyteArray session_config_bytes) {
  // Retrieve the LLM session configuration.
  jbyte* session_config_ref =
      env->GetByteArrayElements(session_config_bytes, nullptr);
  int session_config_size = env->GetArrayLength(session_config_bytes);
  LlmSessionConfig session_config = ParseSessionConfig(
      reinterpret_cast<void*>(session_config_ref), session_config_size);
  env->ReleaseByteArrayElements(session_config_bytes, session_config_ref,
                                JNI_ABORT);

  void* session = nullptr;
  char* error_msg = nullptr;
  int error_code =
      LlmInferenceEngine_CreateSession(&session_config, &session, &error_msg);
  if (error_code) {
    ThrowIfError(env, absl::InternalError(absl::StrCat(
                          "Failed to initialize session: %s", error_msg)));
    free(error_msg);
  }
  FreeSessionConfig(&session_config);
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

JNIEXPORT jobject JNICALL JNI_METHOD(nativeRegisterCallback)(JNIEnv* env,
                                                             jclass thiz,
                                                             jobject callback) {
  if (mediapipe::java::SetJavaVM(env)) {
    auto callback_ref = env->NewGlobalRef(callback);
    if (callback_ref) return callback_ref;
  }
  ThrowIfError(env, absl::InternalError("Failed to allocate callback"));
  return nullptr;
}

JNIEXPORT void JNICALL JNI_METHOD(nativeRemoveCallback)(JNIEnv* env,
                                                        jclass thiz,
                                                        jobject callback_ref) {
  env->DeleteGlobalRef(callback_ref);
}

JNIEXPORT void JNICALL JNI_METHOD(nativePredictAsync)(JNIEnv* env, jclass thiz,
                                                      jlong session_handle,
                                                      jobject callback_ref,
                                                      jstring input) {
  std::string input_str = JStringToStdString(env, input);
  LlmInferenceEngine_Session_PredictAsync(
      reinterpret_cast<LlmInferenceEngine_Session*>(session_handle),
      reinterpret_cast<void*>(callback_ref), input_str.c_str(),
      &ProcessAsyncResponse);
}

JNIEXPORT jint JNICALL JNI_METHOD(nativeSizeInTokens)(JNIEnv* env, jclass thiz,
                                                      jlong session_handle,
                                                      jstring input) {
  std::string input_str = JStringToStdString(env, input);
  char* error_msg = nullptr;
  int size = LlmInferenceEngine_Session_SizeInTokens(
      reinterpret_cast<void*>(session_handle), input_str.c_str(), &error_msg);
  if (size == -1) {
    ThrowIfError(env, absl::InternalError(absl::StrCat(
                          "Failed to compute size: %s", error_msg)));
    free(error_msg);
  }
  return size;
}
