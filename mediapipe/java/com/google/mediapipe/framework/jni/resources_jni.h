// Copyright 2025 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_RESOURCES_JNI_H_
#define MEDIAPIPE_JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_RESOURCES_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define MEDIAPIPE_RESOURCES_SERVICE_METHOD(METHOD_NAME) \
  Java_com_google_mediapipe_framework_ResourcesService_##METHOD_NAME

JNIEXPORT void JNICALL MEDIAPIPE_RESOURCES_SERVICE_METHOD(
    nativeInstallServiceObject)(JNIEnv* env, jobject thiz, jlong context,
                                jobject resourcesMapping);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // MEDIAPIPE_JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_RESOURCES_JNI_H_
