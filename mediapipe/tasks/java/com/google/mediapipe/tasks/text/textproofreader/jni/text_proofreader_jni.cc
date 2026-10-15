// Copyright 2026 The MediaPipe Authors.
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

#include "mediapipe/tasks/java/com/google/mediapipe/tasks/text/textproofreader/jni/text_proofreader_jni.h"

#include <jni.h>

#include <cstdlib>
#include <string>

#include "mediapipe/java/com/google/mediapipe/framework/jni/jni_util.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/text/text_proofreader/text_proofreader.h"

namespace {
using mediapipe::android::JStringToStdString;

void ThrowRuntimeException(JNIEnv* env, const std::string& message) {
  jclass exception_cls = env->FindClass("java/lang/RuntimeException");
  env->ThrowNew(exception_cls, message.c_str());
}

bool ThrowIfError(JNIEnv* env, MpStatus status, char* error_msg) {
  if (status != kMpOk) {
    std::string error_str = error_msg ? error_msg : "Unknown error";
    ThrowRuntimeException(env, error_str);
    if (error_msg) free(error_msg);
    return true;
  }
  return false;
}

}  // namespace

JNIEXPORT jlong JNICALL JNI_METHOD(nativeCreateHandle)(
    JNIEnv* env, jclass clazz, jint model_fd, jstring model_path,
    jint max_num_tokens, jstring app_id, jstring app_version,
    jint host_environment, jint host_system) {
  mediapipe::java::SetJavaVM(env);
  MpTextProofreaderOptions options = {};
  options.base_options.file_descriptor = model_fd;

  std::string cpp_model_path;
  if (model_path != nullptr) {
    cpp_model_path = JStringToStdString(env, model_path);
    if (!cpp_model_path.empty()) {
      options.base_options.model_asset_path = cpp_model_path.c_str();
    }
  }

  options.base_options.delegate = MP_DELEGATE_CPU;

  options.base_options.host_environment =
      mediapipe::tasks::c::core::ToMpHostEnvironment(host_environment);
  options.base_options.host_system =
      mediapipe::tasks::c::core::ToMpHostSystem(host_system);

  std::string c_app_id;
  if (app_id != nullptr) {
    c_app_id = JStringToStdString(env, app_id);
    options.base_options.app_id = c_app_id.c_str();
  }

  std::string c_app_version;
  if (app_version != nullptr) {
    c_app_version = JStringToStdString(env, app_version);
    options.base_options.app_version = c_app_version.c_str();
  }

  if (max_num_tokens > 0) {
    options.max_num_tokens = max_num_tokens;
  }

  MpTextProofreaderPtr proofreader = nullptr;
  char* error_msg = nullptr;
  MpStatus status = MpTextProofreaderCreate(&options, &proofreader, &error_msg);

  if (ThrowIfError(env, status, error_msg)) {
    return 0;
  }

  return reinterpret_cast<jlong>(proofreader);
}

struct StreamingContext {
  JavaVM* jvm;
  jobject callback;
  jmethodID on_next_method;
  jmethodID on_error_method;
  jmethodID on_done_method;
  jclass runtime_exception_cls;
  jmethodID runtime_exception_constructor;
  jclass result_cls;
  jmethodID result_constructor;

  jclass array_list_cls;
  jmethodID array_list_init;
  jmethodID correction_obj_append;
  jclass correction_type_cls;
  jobjectArray enum_values;
  jclass correction_cls;
  jmethodID correction_init;
};

namespace {
void CleanupStreamingContext(JNIEnv* env, StreamingContext* ctx) {
  env->DeleteGlobalRef(ctx->callback);
  env->DeleteGlobalRef(ctx->runtime_exception_cls);
  env->DeleteGlobalRef(ctx->result_cls);
  env->DeleteGlobalRef(ctx->array_list_cls);
  env->DeleteGlobalRef(ctx->correction_type_cls);
  env->DeleteGlobalRef(ctx->enum_values);
  env->DeleteGlobalRef(ctx->correction_cls);
  delete ctx;
}

jobject CreateCorrectionListWithContext(JNIEnv* env,
                                        const MpCorrection* corrections_array,
                                        int corrections_count,
                                        StreamingContext* ctx) {
  jobject corrections_list_obj =
      env->NewObject(ctx->array_list_cls, ctx->array_list_init);

  for (int i = 0; i < corrections_count; ++i) {
    int enum_index = static_cast<int>(corrections_array[i].type);
    jobject enum_obj = env->GetObjectArrayElement(ctx->enum_values, enum_index);
    jstring text_str = env->NewStringUTF(
        corrections_array[i].text ? corrections_array[i].text : "");

    jobject correction_obj = env->NewObject(
        ctx->correction_cls, ctx->correction_init, enum_obj, text_str);
    env->CallBooleanMethod(corrections_list_obj, ctx->correction_obj_append,
                           correction_obj);

    env->DeleteLocalRef(text_str);
    env->DeleteLocalRef(enum_obj);
    env->DeleteLocalRef(correction_obj);
  }

  return corrections_list_obj;
}

jobject CreateCorrectionList(JNIEnv* env, const MpCorrection* corrections_array,
                             int corrections_count) {
  jclass array_list_cls = env->FindClass("java/util/ArrayList");
  jmethodID array_list_init = env->GetMethodID(array_list_cls, "<init>", "()V");
  jmethodID correction_obj_append =
      env->GetMethodID(array_list_cls, "add", "(Ljava/lang/Object;)Z");
  jobject corrections_list_obj =
      env->NewObject(array_list_cls, array_list_init);

  jclass correction_type_cls = env->FindClass(
      "com/google/mediapipe/tasks/text/textproofreader/"
      "TextProofreaderResult$CorrectionType");
  jmethodID correction_type_values = env->GetStaticMethodID(
      correction_type_cls, "values",
      "()[Lcom/google/mediapipe/tasks/text/textproofreader/"
      "TextProofreaderResult$CorrectionType;");
  jobjectArray enum_values = static_cast<jobjectArray>(
      env->CallStaticObjectMethod(correction_type_cls, correction_type_values));

  jclass correction_cls = env->FindClass(
      "com/google/mediapipe/tasks/text/textproofreader/"
      "TextProofreaderResult$Correction");
  jmethodID correction_init = env->GetMethodID(
      correction_cls, "<init>",
      "(Lcom/google/mediapipe/tasks/text/textproofreader/"
      "TextProofreaderResult$CorrectionType;Ljava/lang/String;)V");

  for (int i = 0; i < corrections_count; ++i) {
    int enum_index = static_cast<int>(corrections_array[i].type);
    jobject enum_obj = env->GetObjectArrayElement(enum_values, enum_index);
    jstring text_str = env->NewStringUTF(
        corrections_array[i].text ? corrections_array[i].text : "");

    jobject correction_obj =
        env->NewObject(correction_cls, correction_init, enum_obj, text_str);
    env->CallBooleanMethod(corrections_list_obj, correction_obj_append,
                           correction_obj);

    env->DeleteLocalRef(text_str);
    env->DeleteLocalRef(enum_obj);
    env->DeleteLocalRef(correction_obj);
  }

  env->DeleteLocalRef(array_list_cls);
  env->DeleteLocalRef(correction_type_cls);
  env->DeleteLocalRef(enum_values);
  env->DeleteLocalRef(correction_cls);

  return corrections_list_obj;
}
}  // namespace

void StreamingCallback(void* user_data,
                       const MpTextProofreaderStreamResult* result,
                       const char* error_msg) {
  StreamingContext* ctx = static_cast<StreamingContext*>(user_data);
  JNIEnv* env = mediapipe::java::GetJNIEnv();
  if (!env) {
    return;
  }

  if (error_msg) {
    jstring error_str = env->NewStringUTF(error_msg);
    jobject exception_obj =
        env->NewObject(ctx->runtime_exception_cls,
                       ctx->runtime_exception_constructor, error_str);
    env->CallVoidMethod(ctx->callback, ctx->on_error_method, exception_obj);
    env->DeleteLocalRef(error_str);
    env->DeleteLocalRef(exception_obj);

    CleanupStreamingContext(env, ctx);
    return;
  }

  if (result) {
    jstring chunk_str = env->NewStringUTF(result->chunk ? result->chunk : "");
    jobject corrections_list_obj = nullptr;
    if (result->done && result->corrections) {
      corrections_list_obj = CreateCorrectionListWithContext(
          env, result->corrections, result->corrections_count, ctx);
    }

    jobject result_obj =
        env->NewObject(ctx->result_cls, ctx->result_constructor, chunk_str,
                       corrections_list_obj, result->done);

    env->CallVoidMethod(ctx->callback, ctx->on_next_method, result_obj);

    env->DeleteLocalRef(chunk_str);
    if (corrections_list_obj) {
      env->DeleteLocalRef(corrections_list_obj);
    }
    env->DeleteLocalRef(result_obj);

    if (result->done) {
      env->CallVoidMethod(ctx->callback, ctx->on_done_method);
      CleanupStreamingContext(env, ctx);
    }
  }
}

JNIEXPORT void JNICALL JNI_METHOD(nativeProofreadStreaming)(
    JNIEnv* env, jclass clazz, jlong handle, jstring text, jobject callback) {
  mediapipe::java::SetJavaVM(env);
  MpTextProofreaderPtr proofreader =
      reinterpret_cast<MpTextProofreaderPtr>(handle);
  std::string cpp_text = JStringToStdString(env, text);

  jclass callback_cls = env->GetObjectClass(callback);
  StreamingContext* ctx = new StreamingContext();
  env->GetJavaVM(&(ctx->jvm));
  ctx->callback = env->NewGlobalRef(callback);
  ctx->on_next_method =
      env->GetMethodID(callback_cls, "onNext",
                       "(Lcom/google/mediapipe/tasks/text/textproofreader/"
                       "TextProofreaderStreamingResult;)V");
  ctx->on_error_method =
      env->GetMethodID(callback_cls, "onError", "(Ljava/lang/Throwable;)V");
  ctx->on_done_method = env->GetMethodID(callback_cls, "onDone", "()V");

  jclass runtime_exception_cls = env->FindClass("java/lang/RuntimeException");
  ctx->runtime_exception_cls =
      reinterpret_cast<jclass>(env->NewGlobalRef(runtime_exception_cls));
  ctx->runtime_exception_constructor = env->GetMethodID(
      runtime_exception_cls, "<init>", "(Ljava/lang/String;)V");

  jclass result_cls = env->FindClass(
      "com/google/mediapipe/tasks/text/textproofreader/"
      "TextProofreaderStreamingResult");
  ctx->result_cls = reinterpret_cast<jclass>(env->NewGlobalRef(result_cls));
  ctx->result_constructor = env->GetMethodID(
      result_cls, "<init>", "(Ljava/lang/String;Ljava/util/List;Z)V");

  jclass array_list_cls = env->FindClass("java/util/ArrayList");
  ctx->array_list_cls =
      reinterpret_cast<jclass>(env->NewGlobalRef(array_list_cls));
  ctx->array_list_init = env->GetMethodID(array_list_cls, "<init>", "()V");
  ctx->correction_obj_append =
      env->GetMethodID(array_list_cls, "add", "(Ljava/lang/Object;)Z");

  jclass correction_type_cls = env->FindClass(
      "com/google/mediapipe/tasks/text/textproofreader/"
      "TextProofreaderResult$CorrectionType");
  ctx->correction_type_cls =
      reinterpret_cast<jclass>(env->NewGlobalRef(correction_type_cls));
  jmethodID correction_type_values = env->GetStaticMethodID(
      correction_type_cls, "values",
      "()[Lcom/google/mediapipe/tasks/text/textproofreader/"
      "TextProofreaderResult$CorrectionType;");
  jobjectArray enum_values = static_cast<jobjectArray>(
      env->CallStaticObjectMethod(correction_type_cls, correction_type_values));
  ctx->enum_values =
      reinterpret_cast<jobjectArray>(env->NewGlobalRef(enum_values));

  jclass correction_cls = env->FindClass(
      "com/google/mediapipe/tasks/text/textproofreader/"
      "TextProofreaderResult$Correction");
  ctx->correction_cls =
      reinterpret_cast<jclass>(env->NewGlobalRef(correction_cls));
  ctx->correction_init = env->GetMethodID(
      correction_cls, "<init>",
      "(Lcom/google/mediapipe/tasks/text/textproofreader/"
      "TextProofreaderResult$CorrectionType;Ljava/lang/String;)V");

  char* error_msg = nullptr;
  MpStatus status = MpTextProofreaderProofreadStreaming(
      proofreader, cpp_text.c_str(), StreamingCallback, ctx, &error_msg);

  // We should delete GlobalRefs if there is an error.
  // Since MpTextProofreaderProofreadStreaming is async, cleanup for the non
  // -error case happens in StreamingCallback when result->done is true (or an
  // async error is reported).
  if (ThrowIfError(env, status, error_msg)) {
    CleanupStreamingContext(env, ctx);
    return;
  }
}

JNIEXPORT jobject JNICALL JNI_METHOD(nativeProofread)(JNIEnv* env, jclass clazz,
                                                      jlong handle,
                                                      jstring text) {
  mediapipe::java::SetJavaVM(env);
  MpTextProofreaderPtr proofreader =
      reinterpret_cast<MpTextProofreaderPtr>(handle);
  std::string cpp_text = JStringToStdString(env, text);

  MpTextProofreaderResult* result = nullptr;
  char* error_msg = nullptr;
  MpStatus status = MpTextProofreaderProofread(proofreader, cpp_text.c_str(),
                                               &result, &error_msg);

  if (ThrowIfError(env, status, error_msg)) {
    return nullptr;
  }

  jstring proofread_str =
      env->NewStringUTF(result->proofread_text ? result->proofread_text : "");
  jobject corrections_list_obj =
      CreateCorrectionList(env, result->corrections, result->corrections_count);

  jclass result_cls = env->FindClass(
      "com/google/mediapipe/tasks/text/textproofreader/"
      "TextProofreaderResult");
  jmethodID constructor = env->GetMethodID(
      result_cls, "<init>", "(Ljava/lang/String;Ljava/util/List;)V");
  jobject result_obj = env->NewObject(result_cls, constructor, proofread_str,
                                      corrections_list_obj);

  MpTextProofreaderCloseResult(result);
  return result_obj;
}

JNIEXPORT void JNICALL JNI_METHOD(nativeClose)(JNIEnv* env, jclass clazz,
                                               jlong handle) {
  MpTextProofreaderPtr proofreader =
      reinterpret_cast<MpTextProofreaderPtr>(handle);
  char* error_msg = nullptr;
  MpTextProofreaderClose(proofreader, &error_msg);
  if (error_msg) free(error_msg);
}
