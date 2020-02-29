// Copyright 2020 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_UTIL_ANDROID_JNI_HELPER_H_
#define MEDIAPIPE_UTIL_ANDROID_JNI_HELPER_H_

#include <jni.h>

namespace mediapipe {
namespace jni_common {

inline bool ExceptionPrintClear(JNIEnv* env) {
  if (env->ExceptionCheck()) {
    env->ExceptionDescribe();
    env->ExceptionClear();
    return true;
  }
  return false;
}

class JniHelper {
 public:
  // This constructor should be used when a JavaVM pointer is available, and the
  // JNIEnv needs to be obtained using AttachCurrentThread. This will also push
  // a local stack frame, and pop it when this object is destroyed. If
  // enable_logging is true, it will log verbosely in the constructor and
  // destructor.
  JniHelper(JavaVM* vm, jint version, int caller_line,
            bool enable_logging = true);

  // This constructor should be used then the JNIEnv pointer itself is
  // available, and the only thing that needs to be taken care of is pushing and
  // popping the stack frames. If enable_logging is true, it will log verbosely
  // in the constructor and destructor.
  JniHelper(JNIEnv* env, int caller_line, bool enable_logging = true);

  // Detaches the current thread, if necessary, and pops the local stack frame
  // that was pushed during construction.
  ~JniHelper();

  // Copy and assignment are disallowed because it could cause double-detaching.
  JniHelper(const JniHelper& other) = delete;
  JniHelper& operator=(const JniHelper& other) = delete;

  JNIEnv* GetEnv() const;

 private:
  JavaVM* vm_;
  JNIEnv* env_;
  bool need_to_detach_;
  const int caller_line_;
  const bool enable_logging_;
};

}  // namespace jni_common
}  // namespace mediapipe

#endif  // MEDIAPIPE_UTIL_ANDROID_JNI_HELPER_H_
