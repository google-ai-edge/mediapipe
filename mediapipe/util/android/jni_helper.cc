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

#include "mediapipe/util/android/jni_helper.h"

#include "mediapipe/util/android/logging.h"

namespace mediapipe {
namespace jni_common {

JniHelper::JniHelper(JavaVM* vm, jint version, int caller_line,
                     bool enable_logging)
    : vm_(vm),
      env_(nullptr),
      need_to_detach_(false),
      caller_line_(caller_line),
      enable_logging_(enable_logging) {
  JNI_COMMON_CHECK(vm_);
  const int code = vm_->GetEnv(reinterpret_cast<void**>(&env_), version);
  if (code == JNI_OK) {
    if (0 != env_->PushLocalFrame(0)) {  // Create new stack frame.
      ExceptionPrintClear(env_);
      if (enable_logging_) {
        JNI_COMMON_LOG(VERBOSE, "JniHelper: failed to push local frame.");
      }
      env_ = nullptr;
    }
  } else if (code == JNI_EDETACHED) {
    if (vm_->AttachCurrentThread(&env_, nullptr) == JNI_OK) {
      if (enable_logging_) {
        JNI_COMMON_LOG(VERBOSE,
                       "JniHelper: attached thread (Called from line %d).",
                       caller_line_);
      }
      need_to_detach_ = true;
    } else {
      if (enable_logging_) {
        JNI_COMMON_LOG(
            ERROR,
            "JniHelper: couldn't attach current thread (Called from line %d).",
            caller_line_);
      }
      env_ = nullptr;
    }
  } else {
    if (enable_logging_) {
      JNI_COMMON_LOG(ERROR,
                     "JniHelper: couldn't get env (Called from line %d).",
                     caller_line_);
    }
    env_ = nullptr;
  }
}

JniHelper::JniHelper(JNIEnv* env, int caller_line, bool enable_logging)
    : vm_(nullptr),
      env_(env),
      need_to_detach_(false),
      caller_line_(caller_line),
      enable_logging_(enable_logging) {
  JNI_COMMON_CHECK(env_);
  if (0 != env_->PushLocalFrame(0)) {  // Create new stack frame.
    ExceptionPrintClear(env_);
    if (enable_logging_) {
      JNI_COMMON_LOG(
          VERBOSE,
          "JniHelper: failed to push local frame (Called from line %d).",
          caller_line_);
    }
    env_ = nullptr;
  }
}

JniHelper::~JniHelper() {
  if (need_to_detach_) {
    if (enable_logging_) {
      JNI_COMMON_LOG(
          VERBOSE, "~JniHelper: about to detach thread (Called from line %d).",
          caller_line_);
    }
    if (vm_->DetachCurrentThread() == JNI_OK) {
      if (enable_logging_) {
        JNI_COMMON_LOG(VERBOSE,
                       "~JniHelper: detached thread (Called from line %d).",
                       caller_line_);
      }
    } else {
      if (enable_logging_) {
        JNI_COMMON_LOG(
            ERROR, "~JniHelper: couldn't detach thread (Called from line %d).",
            caller_line_);
      }
    }
  } else {
    if (env_ != nullptr) {
      env_->PopLocalFrame(nullptr);  // Clean up local references.
    }
  }
}

JNIEnv* JniHelper::GetEnv() const { return env_; }

}  // namespace jni_common
}  // namespace mediapipe
