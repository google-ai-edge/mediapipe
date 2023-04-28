// Copyright 2022 The MediaPipe Authors.
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

#ifndef JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_CORE_JNI_MODEL_RESOURCES_CACHE_JNI_H_
#define JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_CORE_JNI_MODEL_RESOURCES_CACHE_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define MODEL_RESOURCES_CACHE_METHOD(METHOD_NAME) \
  Java_com_google_mediapipe_tasks_core_ModelResourcesCache_##METHOD_NAME

#define MODEL_RESOURCES_CACHE_SERVICE_METHOD(METHOD_NAME) \
  Java_com_google_mediapipe_tasks_core_ModelResourcesCacheService_##METHOD_NAME

JNIEXPORT jlong JNICALL MODEL_RESOURCES_CACHE_METHOD(
    nativeCreateModelResourcesCache)(JNIEnv* env, jobject thiz);

JNIEXPORT void JNICALL MODEL_RESOURCES_CACHE_METHOD(
    nativeReleaseModelResourcesCache)(JNIEnv* env, jobject thiz,
                                      jlong nativeHandle);

JNIEXPORT void JNICALL MODEL_RESOURCES_CACHE_SERVICE_METHOD(
    nativeInstallServiceObject)(JNIEnv* env, jobject thiz, jlong contextHandle,
                                jlong objectHandle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_CORE_JNI_MODEL_RESOURCES_CACHE_JNI_H_
