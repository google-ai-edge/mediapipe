// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/java/com/google/mediapipe/framework/jni/jni_util.h"

#include <pthread.h>

#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/class_registry.h"

namespace {

ABSL_CONST_INIT absl::Mutex g_jvm_mutex(absl::kConstInit);
JavaVM* g_jvm ABSL_GUARDED_BY(g_jvm_mutex);

class JvmThread {
 public:
  explicit JvmThread(JavaVM* jvm) {
    jvm_ = jvm;
    attached_ = false;
    jni_env_ = nullptr;
    int get_env_stat =
        jvm_->GetEnv(reinterpret_cast<void**>(&jni_env_), JNI_VERSION_1_6);
    // TODO: report the error back to Java layer.
    switch (get_env_stat) {
      case JNI_OK:
        break;
      case JNI_EDETACHED:
        LOG(INFO) << "GetEnv: not attached";
        if (jvm_->AttachCurrentThread(
#ifdef __ANDROID__
                &jni_env_,
#else
                reinterpret_cast<void**>(&jni_env_),
#endif  // __ANDROID__
                nullptr) != 0) {
          LOG(ERROR) << "Failed to attach to java thread.";
          break;
        }
        attached_ = true;
        break;
      case JNI_EVERSION:
        LOG(ERROR) << "GetEnv: jni version not supported.";
        break;
      default:
        LOG(ERROR) << "GetEnv: unknown status.";
        break;
    }
  }

  ~JvmThread() {
    if (attached_) {
      jvm_->DetachCurrentThread();
    }
  }

  JNIEnv* GetEnv() const { return jni_env_; }

 private:
  bool attached_;
  JavaVM* jvm_;
  JNIEnv* jni_env_;
};

// Since current android abi doesn't have pthread_local, we have to rely on
// pthread functions to achieve the detachment of java thread when native thread
// exits (see: http://developer.android.com/training/articles/perf-jni.html).
static pthread_key_t jvm_thread_key;
static pthread_once_t key_once = PTHREAD_ONCE_INIT;

static void ThreadExitCallback(void* key_value) {
  JvmThread* jvm_thread = reinterpret_cast<JvmThread*>(key_value);
  // Detach the thread when thread exits.
  LOG(INFO) << "Exiting thread. Detach thread.";
  delete jvm_thread;
}

void MakeKey() { pthread_key_create(&jvm_thread_key, ThreadExitCallback); }

// Returns the global Java VM instance.
JavaVM* GetJavaVM() {
  absl::MutexLock lock(&g_jvm_mutex);
  return g_jvm;
}

}  // namespace

namespace mediapipe {

namespace android {

std::string JStringToStdString(JNIEnv* env, jstring jstr) {
  const char* s = env->GetStringUTFChars(jstr, 0);
  if (!s) {
    return std::string();
  }
  std::string str(s);
  env->ReleaseStringUTFChars(jstr, s);
  return str;
}

// Converts a `java.util.List<String>` to a `std::vector<std::string>`.
std::vector<std::string> JavaListToStdStringVector(JNIEnv* env, jobject from) {
  jclass cls = env->FindClass("java/util/List");
  int size = env->CallIntMethod(from, env->GetMethodID(cls, "size", "()I"));
  std::vector<std::string> result;
  result.reserve(size);
  for (int i = 0; i < size; i++) {
    jobject element = env->CallObjectMethod(
        from, env->GetMethodID(cls, "get", "(I)Ljava/lang/Object;"), i);
    result.push_back(JStringToStdString(env, static_cast<jstring>(element)));
    env->DeleteLocalRef(element);
  }
  env->DeleteLocalRef(cls);
  return result;
}

jthrowable CreateMediaPipeException(JNIEnv* env, absl::Status status) {
  auto& class_registry = mediapipe::android::ClassRegistry::GetInstance();
  std::string mpe_class_name = class_registry.GetClassName(
      mediapipe::android::ClassRegistry::kMediaPipeExceptionClassName);
  std::string mpe_constructor_name = class_registry.GetMethodName(
      mediapipe::android::ClassRegistry::kMediaPipeExceptionClassName,
      "<init>");

  jclass status_cls = env->FindClass(mpe_class_name.c_str());
  jmethodID status_ctr =
      env->GetMethodID(status_cls, mpe_constructor_name.c_str(), "(I[B)V");
  int length = status.message().length();
  jbyteArray message_bytes = env->NewByteArray(length);
  env->SetByteArrayRegion(message_bytes, 0, length,
                          reinterpret_cast<jbyte*>(const_cast<char*>(
                              std::string(status.message()).c_str())));
  jthrowable result = reinterpret_cast<jthrowable>(
      env->NewObject(status_cls, status_ctr, status.code(), message_bytes));
  env->DeleteLocalRef(status_cls);
  return result;
}

bool ThrowIfError(JNIEnv* env, absl::Status status) {
  if (!status.ok()) {
    env->Throw(mediapipe::android::CreateMediaPipeException(env, status));
    return true;
  }
  return false;
}

SerializedMessageIds::SerializedMessageIds(JNIEnv* env, jobject data) {
  auto& class_registry = mediapipe::android::ClassRegistry::GetInstance();
  std::string serialized_message(
      mediapipe::android::ClassRegistry::kProtoUtilSerializedMessageClassName);
  std::string serialized_message_obfuscated =
      class_registry.GetClassName(serialized_message);
  std::string type_name_obfuscated =
      class_registry.GetFieldName(serialized_message, "typeName");
  std::string value_obfuscated =
      class_registry.GetFieldName(serialized_message, "value");
  jclass j_class = env->FindClass(serialized_message_obfuscated.c_str());
  type_name_id = env->GetFieldID(j_class, type_name_obfuscated.c_str(),
                                 "Ljava/lang/String;");
  value_id = env->GetFieldID(j_class, value_obfuscated.c_str(), "[B");
  env->DeleteLocalRef(j_class);
}

}  // namespace android

namespace java {

bool HasJavaVM() {
  absl::MutexLock lock(&g_jvm_mutex);
  return g_jvm != nullptr;
}

bool SetJavaVM(JNIEnv* env) {
  absl::MutexLock lock(&g_jvm_mutex);
  if (!g_jvm) {
    if (env->GetJavaVM(&g_jvm) != JNI_OK) {
      LOG(ERROR) << "Can not get the Java VM instance!";
      g_jvm = nullptr;
      return false;
    }
  }
  return true;
}

JNIEnv* GetJNIEnv() {
  pthread_once(&key_once, MakeKey);
  JvmThread* jvm_thread =
      reinterpret_cast<JvmThread*>(pthread_getspecific(jvm_thread_key));
  if (jvm_thread == nullptr) {
    jvm_thread = new JvmThread(GetJavaVM());
    pthread_setspecific(jvm_thread_key, jvm_thread);
  }
  return jvm_thread->GetEnv();
}

}  // namespace java

}  // namespace mediapipe
