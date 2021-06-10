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

#ifndef JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_JNI_UTIL_H_
#define JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_JNI_UTIL_H_

#include <jni.h>

#include <string>

#include "mediapipe/framework/port/status.h"

namespace mediapipe {

namespace android {

std::string JStringToStdString(JNIEnv* env, jstring jstr);

std::vector<std::string> JavaListToStdStringVector(JNIEnv* env, jobject from);

// Creates a java MediaPipeException object for a absl::Status.
jthrowable CreateMediaPipeException(JNIEnv* env, absl::Status status);

// Throws a MediaPipeException for any non-ok absl::Status.
// Note that the exception is thrown after execution returns to Java.
bool ThrowIfError(JNIEnv* env, absl::Status status);

// The Jni ids for Java class SerializedMessage.
class SerializedMessageIds {
 public:
  SerializedMessageIds(JNIEnv* env, jobject data);
  jclass j_class;
  jfieldID type_name_id;
  jfieldID value_id;
};

}  // namespace android

namespace java {

// Sets the global Java VM instance, if it is not set yet.
// Returns true on success.
bool SetJavaVM(JNIEnv* env);

// Determines if the global Java VM instance is available.
bool HasJavaVM();

// Returns the current JNI environment.
JNIEnv* GetJNIEnv();

}  // namespace java

}  // namespace mediapipe

#endif  // JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_JNI_UTIL_H_
