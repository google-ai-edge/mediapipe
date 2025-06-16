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

#include "mediapipe/tasks/java/com/google/mediapipe/tasks/genai/llminference/jni/llm.h"

#include <jni.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "include/core/SkAlphaType.h"  // from @skia
#include "include/core/SkBitmap.h"     // from @skia
#include "include/core/SkImage.h"      // from @skia
#include "include/core/SkImageInfo.h"  // from @skia
#include "mediapipe/java/com/google/mediapipe/framework/jni/class_registry.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/jni_util.h"
#include "mediapipe/tasks/cc/genai/inference/c/llm_inference_engine.h"
#include "mediapipe/tasks/java/com/google/mediapipe/tasks/genai/llminference/jni/proto/llm_options.pb.h"
#include "mediapipe/tasks/java/com/google/mediapipe/tasks/genai/llminference/jni/proto/llm_response_context.pb.h"

namespace {

using LlmModelSettingsProto =
    mediapipe::tasks::genai::llminference::jni::LlmModelSettings;
using LlmSessionConfigProto =
    mediapipe::tasks::genai::llminference::jni::LlmSessionConfig;
using LlmResponseContextProto =
    mediapipe::tasks::genai::llminference::jni::LlmResponseContext;
using mediapipe::android::JStringToStdString;
using mediapipe::java::GetJNIEnv;

const bool kDefaultIncludeTokenCostCalculator = true;

void ThrowIllegalStateException(JNIEnv* env, const std::string& message) {
  jclass exceptionClass = env->FindClass("java/lang/IllegalStateException");
  jmethodID exceptionInit =
      env->GetMethodID(exceptionClass, "<init>", "(Ljava/lang/String;)V");
  jstring exceptionMessage = env->NewStringUTF(message.c_str());
  jthrowable exception = static_cast<jthrowable>(
      env->NewObject(exceptionClass, exceptionInit, exceptionMessage));
  env->Throw(exception);
}

bool ThrowIfError(JNIEnv* env, absl::Status status) {
  if (!status.ok()) {
    std::string error_message = std::string(status.message());
    ThrowIllegalStateException(env, error_message);
    return true;
  }
  return false;
}

LlmModelSettings ParseModelSettings(void* bytes, int size) {
  LlmModelSettingsProto input;
  input.ParseFromArray(bytes, size);

  LlmModelSettings output;
  output.model_path = strdup(input.model_path().c_str());
  output.vision_encoder_path =
      input.vision_model_settings().has_encoder_path()
          ? strdup(input.vision_model_settings().encoder_path().c_str())
          : nullptr;
  output.vision_adapter_path =
      input.vision_model_settings().has_adapter_path()
          ? strdup(input.vision_model_settings().adapter_path().c_str())
          : nullptr;
  output.cache_dir = strdup(input.cache_dir().c_str());
  output.sequence_batch_size = input.sequence_batch_size();
  output.num_decode_steps_per_sync = input.num_decode_steps_per_sync();
  output.max_num_tokens = input.max_tokens();
  output.max_num_images = input.max_num_images();
  output.max_top_k = input.max_top_k();
  output.number_of_supported_lora_ranks =
      input.number_of_supported_lora_ranks();
  if (input.supported_lora_ranks_size() > 0) {
    output.supported_lora_ranks = new size_t[input.supported_lora_ranks_size()];
    for (int i = 0; i < input.supported_lora_ranks_size(); ++i) {
      output.supported_lora_ranks[i] = input.supported_lora_ranks(i);
    }
  } else {
    output.supported_lora_ranks = nullptr;
  }
  output.llm_activation_data_type = kLlmActivationDataTypeDefault;
  output.num_draft_tokens = 0;
  output.wait_for_weight_uploads = false;
  output.use_submodel = false;
  switch (input.llm_preferred_backend()) {
    case LlmModelSettingsProto::DEFAULT:
      output.preferred_backend = kLlmPreferredBackendDefault;
      break;
    case LlmModelSettingsProto::GPU:
      output.preferred_backend = kLlmPreferredBackendGpu;
      break;
    case LlmModelSettingsProto::CPU:
      output.preferred_backend = kLlmPreferredBackendCpu;
      break;
    default:
      output.preferred_backend = kLlmPreferredBackendDefault;
      break;
  }
  if (input.has_audio_model_settings()) {
    output.enable_audio_modality = true;
    output.max_audio_sequence_length =
        input.audio_model_settings().max_audio_sequence_length();
  } else {
    output.enable_audio_modality = false;
  }
  return output;
}

LlmSessionConfig ParseSessionConfig(void* bytes, int size) {
  LlmSessionConfigProto input;
  input.ParseFromArray(bytes, size);

  LlmSessionConfig output;
  output.temperature = input.temperature();
  output.topk = input.topk();
  output.topp = input.topp();
  output.random_seed = input.random_seed();
  if (input.has_lora_path()) {
    output.lora_path = strdup(input.lora_path().c_str());
  }
  output.include_token_cost_calculator =
      input.graph_config().has_include_token_cost_calculator()
          ? input.graph_config().include_token_cost_calculator()
          : kDefaultIncludeTokenCostCalculator;
  output.enable_vision_modality = input.graph_config().enable_vision_modality();
  output.enable_audio_modality = input.graph_config().enable_audio_modality();
  if (input.has_prompt_templates()) {
    LlmPromptTemplates* prompt_templates = new LlmPromptTemplates();
    if (input.prompt_templates().has_user_prefix()) {
      prompt_templates->user_prefix =
          strdup(input.prompt_templates().user_prefix().c_str());
    }
    if (input.prompt_templates().has_user_suffix()) {
      prompt_templates->user_suffix =
          strdup(input.prompt_templates().user_suffix().c_str());
    }
    if (input.prompt_templates().has_model_prefix()) {
      prompt_templates->model_prefix =
          strdup(input.prompt_templates().model_prefix().c_str());
    }
    if (input.prompt_templates().has_model_suffix()) {
      prompt_templates->model_suffix =
          strdup(input.prompt_templates().model_suffix().c_str());
    }
    if (input.prompt_templates().has_system_prefix()) {
      prompt_templates->system_prefix =
          strdup(input.prompt_templates().system_prefix().c_str());
    }
    if (input.prompt_templates().has_system_suffix()) {
      prompt_templates->system_suffix =
          strdup(input.prompt_templates().system_suffix().c_str());
    }
    output.prompt_templates = prompt_templates;
  } else {
    output.prompt_templates = nullptr;
  }
  return output;
}

void FreeModelSettings(LlmModelSettings* model_settings) {
  delete model_settings->model_path;
  delete model_settings->vision_adapter_path;
  delete model_settings->vision_encoder_path;
  delete model_settings->cache_dir;
  delete[] model_settings->supported_lora_ranks;
  model_settings->model_path = nullptr;
  model_settings->cache_dir = nullptr;
}

void FreeSessionConfig(LlmSessionConfig* session_config) {
  // Release optional resources because they are initialized with strdup or new.
  if (session_config->lora_path != nullptr) {
    delete session_config->lora_path;
  }
  if (session_config->prompt_templates != nullptr) {
    if (session_config->prompt_templates->user_prefix != nullptr) {
      delete session_config->prompt_templates->user_prefix;
    }
    if (session_config->prompt_templates->user_suffix != nullptr) {
      delete session_config->prompt_templates->user_suffix;
    }
    if (session_config->prompt_templates->model_prefix != nullptr) {
      delete session_config->prompt_templates->model_prefix;
    }
    if (session_config->prompt_templates->model_suffix != nullptr) {
      delete session_config->prompt_templates->model_suffix;
    }
    if (session_config->prompt_templates->system_prefix != nullptr) {
      delete session_config->prompt_templates->system_prefix;
    }
    if (session_config->prompt_templates->system_suffix != nullptr) {
      delete session_config->prompt_templates->system_suffix;
    }
    delete session_config->prompt_templates;
  }
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
      "com/google/mediapipe/tasks/genai/llminference/LlmTaskRunner",
      "onAsyncResponse");
  jmethodID method_id =
      env->GetMethodID(class_ref, method_name.c_str(), "([B)V");

  const jbyteArray response_context_bytes = ToByteArray(env, *response_context);
  LlmInferenceEngine_CloseResponseContext(response_context);

  env->CallVoidMethod(object_ref, method_id, response_context_bytes);
}

}  // namespace

JNIEXPORT jlong JNICALL JNI_METHOD(nativeCreateEngine)(
    JNIEnv* env, jclass thiz, jbyteArray model_settings_bytes) {
  // Retrieve the LLM model settings.
  jbyte* model_settings_ref =
      env->GetByteArrayElements(model_settings_bytes, nullptr);
  int model_settings_size = env->GetArrayLength(model_settings_bytes);
  LlmModelSettings model_settings = ParseModelSettings(
      reinterpret_cast<void*>(model_settings_ref), model_settings_size);
  env->ReleaseByteArrayElements(model_settings_bytes, model_settings_ref,
                                JNI_ABORT);

  void* engine = nullptr;
  char* error_msg = nullptr;
  int error_code =
      LlmInferenceEngine_CreateEngine(&model_settings, &engine, &error_msg);
  if (error_code) {
    ThrowIfError(env, absl::InternalError(absl::StrCat(
                          "Failed to initialize engine: %s", error_msg)));
    free(error_msg);
  }
  FreeModelSettings(&model_settings);
  return reinterpret_cast<jlong>(engine);
}

JNIEXPORT void JNICALL JNI_METHOD(nativeDeleteEngine)(JNIEnv* env, jclass thiz,
                                                      jlong engine_handle) {
  LlmInferenceEngine_Engine_Delete(reinterpret_cast<void*>(engine_handle));
}

JNIEXPORT jlong JNICALL JNI_METHOD(nativeCreateSession)(
    JNIEnv* env, jclass thiz, jbyteArray session_config_bytes,
    jlong engine_handle) {
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
      LlmInferenceEngine_CreateSession(reinterpret_cast<void*>(engine_handle),
                                       &session_config, &session, &error_msg);
  if (error_code) {
    ThrowIfError(env, absl::InternalError(absl::StrCat(
                          "Failed to initialize session: %s", error_msg)));
    free(error_msg);
  }
  FreeSessionConfig(&session_config);
  return reinterpret_cast<jlong>(session);
}

JNIEXPORT jlong JNICALL JNI_METHOD(nativeCloneSession)(JNIEnv* env, jclass thiz,
                                                       jlong session_handle) {
  void* session = nullptr;
  char* error_msg = nullptr;
  int error_code = LlmInferenceEngine_Session_Clone(
      reinterpret_cast<void*>(session_handle), &session, &error_msg);
  if (error_code) {
    ThrowIfError(env, absl::InternalError(absl::StrCat(
                          "Failed to clone session: %s", error_msg)));
    free(error_msg);
  }
  return reinterpret_cast<jlong>(session);
}

JNIEXPORT void JNICALL JNI_METHOD(nativeDeleteSession)(JNIEnv* env, jclass thiz,
                                                       jlong session_handle) {
  LlmInferenceEngine_Session_Delete(reinterpret_cast<void*>(session_handle));
}

JNIEXPORT void JNICALL JNI_METHOD(nativeAddQueryChunk)(JNIEnv* env, jclass thiz,
                                                       jlong session_handle,
                                                       jstring input) {
  std::string input_str = JStringToStdString(env, input);
  char* error_msg = nullptr;
  int error_code = LlmInferenceEngine_Session_AddQueryChunk(
      reinterpret_cast<void*>(session_handle), input_str.c_str(), &error_msg);
  if (error_code) {
    ThrowIfError(
        env, absl::InternalError(absl::StrCat(
                 "Failed to add query chunk: %s, %s", input_str, error_msg)));
    free(error_msg);
  }
}

JNIEXPORT void JNICALL JNI_METHOD(nativeAddImage)(JNIEnv* env, jclass thiz,
                                                  jlong session_handle,
                                                  jlong image_handle) {
  char* error_msg = nullptr;
  int error_code = LlmInferenceEngine_Session_AddImage(
      reinterpret_cast<void*>(session_handle),
      reinterpret_cast<void*>(image_handle), &error_msg);
  if (error_code) {
    ThrowIfError(env, absl::InternalError(
                          absl::StrCat("Failed to add image:, %s", error_msg)));
    free(error_msg);
  }
}

JNIEXPORT void JNICALL JNI_METHOD(nativeAddAudio)(JNIEnv* env, jclass thiz,
                                                  jlong engine_handle,
                                                  jlong session_handle,
                                                  jbyteArray audio_bytes) {
  char* error_msg = nullptr;

  jbyte* audio_elements_ptr = env->GetByteArrayElements(audio_bytes, nullptr);
  if (audio_elements_ptr == nullptr) {
    ThrowIfError(env, absl::InternalError(
                          "Failed to get byte array elements for audio."));
    return;
  }
  jsize array_len_bytes = env->GetArrayLength(audio_bytes);

  int error_code = LlmInferenceEngine_Session_AddAudio(
      reinterpret_cast<void*>(engine_handle),
      reinterpret_cast<void*>(session_handle),
      reinterpret_cast<const char*>(audio_elements_ptr),
      static_cast<int>(array_len_bytes), &error_msg);

  env->ReleaseByteArrayElements(audio_bytes, audio_elements_ptr,
                                JNI_ABORT);  // Release after C API call

  if (error_code) {
    ThrowIfError(env, absl::InternalError(absl::StrCat(
                          "Failed to add audio spectrum: %s", error_msg)));
    free(error_msg);
  }
}

JNIEXPORT jbyteArray JNICALL
JNI_METHOD(nativePredictSync)(JNIEnv* env, jclass thiz, jlong session_handle) {
  char* error_msg = nullptr;
  LlmResponseContext response_context;
  int error_code = LlmInferenceEngine_Session_PredictSync(
      reinterpret_cast<void*>(session_handle), &response_context, &error_msg);
  if (error_code) {
    ThrowIfError(env, absl::InternalError(absl::StrCat(
                          "Failed to predict sync: %s", error_msg)));
    free(error_msg);
  }
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
                                                      jobject callback_ref) {
  char* error_msg = nullptr;
  int error_code = LlmInferenceEngine_Session_PredictAsync(
      reinterpret_cast<LlmInferenceEngine_Session*>(session_handle),
      reinterpret_cast<void*>(callback_ref), &error_msg, &ProcessAsyncResponse);
  if (error_code) {
    ThrowIfError(env, absl::InternalError(absl::StrCat(
                          "Failed to predict async: %s", error_msg)));
    free(error_msg);
  }
}

JNIEXPORT void JNICALL JNI_METHOD(nativePendingProcessCancellation)(
    JNIEnv* env, jclass, jlong session_handle) {
  char* error_msg = nullptr;
  int error_code = LlmInferenceEngine_Session_PendingProcessCancellation(
      reinterpret_cast<LlmInferenceEngine_Session*>(session_handle),
      &error_msg);
  if (error_code) {
    ThrowIfError(env,
                 absl::InternalError(absl::StrCat(
                     "Failed to cancel pending processes: %s", error_msg)));
    free(error_msg);
  }
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

JNIEXPORT jlong JNICALL JNI_METHOD(nativeCreateSkBitmap)(
    JNIEnv* env, jclass thiz, jobject byte_buffer, jint width, jint height,
    jint color_type, jint alpha_type) {
  const int64_t buffer_size = env->GetDirectBufferCapacity(byte_buffer);
  void* buffer_data = env->GetDirectBufferAddress(byte_buffer);
  if (buffer_data == nullptr || buffer_size < 0) {
    ThrowIfError(env, absl::InternalError("Cannot get direct access to the "
                                          "input buffer. It should be created "
                                          "using allocateDirect."));
  }

  SkColorType sk_color_type = static_cast<SkColorType>(color_type);
  SkAlphaType sk_alpha_type = static_cast<SkAlphaType>(alpha_type);
  SkImageInfo imageInfo =
      SkImageInfo::Make(width, height, sk_color_type, sk_alpha_type);

  auto bitmap = std::make_unique<SkBitmap>();
  bool success =
      bitmap->installPixels(imageInfo, buffer_data, imageInfo.minRowBytes());
  if (!success) {
    ThrowIfError(env, absl::InternalError("Cannot initialize SkBitmap."));
  }

  return reinterpret_cast<jlong>(bitmap.release());
}

JNIEXPORT void JNICALL JNI_METHOD(nativeDeleteSkBitmap)(JNIEnv*, jclass,
                                                        jlong bitmap_handle) {
  delete reinterpret_cast<SkBitmap*>(bitmap_handle);
}

JNIEXPORT jlong JNICALL JNI_METHOD(nativeGetSentencePieceProcessor)(
    JNIEnv* env, jclass thiz, jlong engine_handle) {
  const void* processor = nullptr;
  char* error_msg = nullptr;
  int error_code = LlmInferenceEngine_GetSentencePieceProcessor(
      reinterpret_cast<void*>(engine_handle), &processor, &error_msg);
  if (error_code) {
    ThrowIfError(env,
                 absl::InternalError(absl::StrCat(
                     "Failed to get SentencePieceProcessor: %s", error_msg)));
    free(error_msg);
    return 0;  // Return 0 on failure.
  }
  return reinterpret_cast<jlong>(processor);
}

JNIEXPORT void JNICALL JNI_METHOD(nativeUpdateSessionConfig)(
    JNIEnv* env, jclass thiz, jlong session_handle, jbyteArray config_bytes) {
  if (session_handle == 0) {
    ThrowIfError(env, absl::InvalidArgumentError("Invalid session handle."));
    return;
  }

  auto session = reinterpret_cast<LlmInferenceEngine_Session*>(session_handle);

  // Get the byte array data.
  jbyte* config_data = env->GetByteArrayElements(config_bytes, nullptr);
  jsize config_length = env->GetArrayLength(config_bytes);

  // Parse the byte array into an LlmSessionConfig proto.
  LlmSessionConfigProto session_config_proto;
  if (!session_config_proto.ParseFromArray(config_data, config_length)) {
    env->ReleaseByteArrayElements(config_bytes, config_data, JNI_ABORT);
    ThrowIfError(env, absl::InvalidArgumentError("Invalid config bytes."));
    return;
  }
  env->ReleaseByteArrayElements(config_bytes, config_data, JNI_ABORT);

  // Convert the proto to the C struct.
  SessionRuntimeConfig config = {};
  size_t topk = 0;
  float topp = 0.0f;
  float temperature = 0.0f;
  size_t random_seed = 0;
  if (session_config_proto.has_topk()) {
    topk = session_config_proto.topk();
    config.topk = &topk;
  }
  if (session_config_proto.has_topp()) {
    topp = session_config_proto.topp();
    config.topp = &topp;
  }
  if (session_config_proto.has_temperature()) {
    temperature = session_config_proto.temperature();
    config.temperature = &temperature;
  }
  if (session_config_proto.has_random_seed()) {
    random_seed = session_config_proto.random_seed();
    config.random_seed = &random_seed;
  }
  if (session_config_proto.has_constraint_handle()) {
    config.constraint =
        reinterpret_cast<Constraint*>(session_config_proto.constraint_handle());
  }

  char* error_msg = nullptr;
  int error_code =
      LlmInferenceEngine_UpdateRuntimeConfig(session, &config, &error_msg);
  if (error_code) {
    ThrowIfError(env, absl::InternalError(absl::StrCat(
                          "Failed to update runtime config: %s", error_msg)));
    free(error_msg);
  }
}
