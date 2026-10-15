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

#include "mediapipe/tasks/java/com/google/mediapipe/tasks/text/textsummarizer/jni/text_summarizer_jni.h"

#include <jni.h>

#include <cstdlib>
#include <string>

#include "mediapipe/java/com/google/mediapipe/framework/jni/jni_util.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/text/text_summarizer/text_summarizer.h"

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
    jint max_num_tokens, jint mode, jstring app_id, jstring app_version,
    jint host_environment, jint host_system) {
  MpTextSummarizerOptions options = {};
  options.base_options.file_descriptor = model_fd;

  std::string c_model_path;
  if (model_path != nullptr) {
    c_model_path = JStringToStdString(env, model_path);
    if (!c_model_path.empty()) {
      options.base_options.model_asset_path = c_model_path.c_str();
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

  if (mode != MP_TEXT_SUMMARIZER_MODE_TLDR &&
      mode != MP_TEXT_SUMMARIZER_MODE_KEYPOINTS) {
    ThrowRuntimeException(env, "Invalid TextSummarizerMode.");
    return 0;
  }
  options.mode = static_cast<MpTextSummarizerMode>(mode);

  MpTextSummarizerPtr summarizer = nullptr;
  char* error_msg = nullptr;
  MpStatus status = MpTextSummarizerCreate(&options, &summarizer, &error_msg);

  if (ThrowIfError(env, status, error_msg)) {
    return 0;
  }

  return reinterpret_cast<jlong>(summarizer);
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
};

void StreamingCallback(void* user_data,
                       const MpTextSummarizerStreamResult* result,
                       const char* error_msg) {
  StreamingContext* ctx = static_cast<StreamingContext*>(user_data);
  JNIEnv* env;
  bool attached = false;
  // Whether current thread is attached to JVM. If not, attach it.
  // On Android: jint AttachCurrentThread(JNIEnv** env, void* args) matches
  // &env.
  //
  // On other platforms: jint AttachCurrentThread(void** env,
  // void* args) matches &env from JNIEnv** to void** to prevent compiler
  // warnings.
  //
  // Note: JNI_VERSION_1_6 is used here to be compatible with older Android
  // versions.
  if (ctx->jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) !=
      JNI_OK) {
#ifdef __ANDROID__
    ctx->jvm->AttachCurrentThread(&env, nullptr);
#else
    ctx->jvm->AttachCurrentThread(reinterpret_cast<void**>(&env), nullptr);
#endif
    attached = true;
  }

  JavaVM* jvm = ctx->jvm;

  if (error_msg) {
    jstring error_str = env->NewStringUTF(error_msg);
    jobject exception_obj =
        env->NewObject(ctx->runtime_exception_cls,
                       ctx->runtime_exception_constructor, error_str);
    env->CallVoidMethod(ctx->callback, ctx->on_error_method, exception_obj);
    env->DeleteLocalRef(error_str);
    env->DeleteLocalRef(exception_obj);

    env->DeleteGlobalRef(ctx->callback);
    env->DeleteGlobalRef(ctx->runtime_exception_cls);
    env->DeleteGlobalRef(ctx->result_cls);
    delete ctx;

    if (attached) jvm->DetachCurrentThread();
    return;
  }

  if (result) {
    jstring summary_str = env->NewStringUTF(result->chunk ? result->chunk : "");
    jobject result_obj = env->NewObject(
        ctx->result_cls, ctx->result_constructor, summary_str, result->done);

    env->CallVoidMethod(ctx->callback, ctx->on_next_method, result_obj);

    env->DeleteLocalRef(summary_str);
    env->DeleteLocalRef(result_obj);

    if (result->done) {
      env->CallVoidMethod(ctx->callback, ctx->on_done_method);
      env->DeleteGlobalRef(ctx->callback);
      env->DeleteGlobalRef(ctx->runtime_exception_cls);
      env->DeleteGlobalRef(ctx->result_cls);
      delete ctx;
    }
  }

  if (attached) {
    // Actually detaching here is a bit tricky if we call it multiple times from
    // a background thread. However since the background thread isn't attached
    // to JVM, we must detach when done.
    jvm->DetachCurrentThread();
  }
}

JNIEXPORT void JNICALL JNI_METHOD(nativeSummarizeStreaming)(
    JNIEnv* env, jobject thiz, jlong handle, jstring text, jobject callback) {
  MpTextSummarizerPtr summarizer =
      reinterpret_cast<MpTextSummarizerPtr>(handle);
  std::string c_text = JStringToStdString(env, text);

  jclass callback_cls = env->GetObjectClass(callback);
  StreamingContext* ctx = new StreamingContext();
  env->GetJavaVM(&(ctx->jvm));
  ctx->callback = env->NewGlobalRef(callback);
  ctx->on_next_method =
      env->GetMethodID(callback_cls, "onNext",
                       "(Lcom/google/mediapipe/tasks/text/textsummarizer/"
                       "TextSummarizerStreamingResult;)V");
  ctx->on_error_method =
      env->GetMethodID(callback_cls, "onError", "(Ljava/lang/Throwable;)V");
  ctx->on_done_method = env->GetMethodID(callback_cls, "onDone", "()V");

  jclass runtime_exception_cls = env->FindClass("java/lang/RuntimeException");
  ctx->runtime_exception_cls =
      reinterpret_cast<jclass>(env->NewGlobalRef(runtime_exception_cls));
  ctx->runtime_exception_constructor = env->GetMethodID(
      runtime_exception_cls, "<init>", "(Ljava/lang/String;)V");

  jclass result_cls = env->FindClass(
      "com/google/mediapipe/tasks/text/textsummarizer/"
      "TextSummarizerStreamingResult");
  ctx->result_cls = reinterpret_cast<jclass>(env->NewGlobalRef(result_cls));
  ctx->result_constructor =
      env->GetMethodID(result_cls, "<init>", "(Ljava/lang/String;Z)V");

  char* error_msg = nullptr;
  MpStatus status = MpTextSummarizerSummarizeStreaming(
      summarizer, c_text.c_str(), StreamingCallback, ctx, &error_msg);

  // We should delete GlobalRefs if there is an error.
  // Since MpTextSummarizerSummarizeStreaming is async, cleanup for the non
  // -error case happens in StreamingCallback when result->done is true (or an
  // async error is reported).
  if (ThrowIfError(env, status, error_msg)) {
    env->DeleteGlobalRef(ctx->callback);
    env->DeleteGlobalRef(ctx->runtime_exception_cls);
    env->DeleteGlobalRef(ctx->result_cls);
    delete ctx;
    return;
  }
}

JNIEXPORT jobject JNICALL JNI_METHOD(nativeSummarize)(JNIEnv* env, jobject thiz,
                                                      jlong handle,
                                                      jstring text) {
  MpTextSummarizerPtr summarizer =
      reinterpret_cast<MpTextSummarizerPtr>(handle);
  std::string c_text = JStringToStdString(env, text);

  MpTextSummarizerResult result = {};
  char* error_msg = nullptr;
  MpStatus status = MpTextSummarizerSummarize(summarizer, c_text.c_str(),
                                              &result, &error_msg);

  if (ThrowIfError(env, status, error_msg)) {
    return nullptr;
  }

  // Convert result to Java object
  jclass result_cls = env->FindClass(
      "com/google/mediapipe/tasks/text/textsummarizer/TextSummarizerResult");
  jmethodID constructor =
      env->GetMethodID(result_cls, "<init>", "(Ljava/lang/String;)V");
  jstring summary_str = env->NewStringUTF(result.summary ? result.summary : "");
  jobject result_obj = env->NewObject(result_cls, constructor, summary_str);

  MpTextSummarizerCloseResult(&result);
  return result_obj;
}

JNIEXPORT void JNICALL JNI_METHOD(nativeClose)(JNIEnv* env, jobject thiz,
                                               jlong handle) {
  MpTextSummarizerPtr summarizer =
      reinterpret_cast<MpTextSummarizerPtr>(handle);
  char* error_msg = nullptr;
  MpTextSummarizerClose(summarizer, &error_msg);
  if (error_msg) free(error_msg);
}
