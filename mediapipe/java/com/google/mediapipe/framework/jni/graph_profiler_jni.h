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

#ifndef JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_GRAPH_PROFILER_JNI_H_
#define JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_GRAPH_PROFILER_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define GRAPH_METHOD(METHOD_NAME) \
  Java_com_google_mediapipe_framework_GraphProfiler_##METHOD_NAME

JNIEXPORT void JNICALL GRAPH_METHOD(nativeReset)(JNIEnv* env, jobject thiz,
                                                 jlong profiling_context);

JNIEXPORT void JNICALL GRAPH_METHOD(nativeResume)(JNIEnv* env, jobject thiz,
                                                  jlong profiling_context);

JNIEXPORT void JNICALL GRAPH_METHOD(nativePause)(JNIEnv* env, jobject thiz,
                                                 jlong profiling_context);

JNIEXPORT jobjectArray JNICALL GRAPH_METHOD(nativeGetCalculatorProfiles)(
    JNIEnv* env, jobject thiz, jlong profiling_context);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // JAVA_COM_GOOGLE_MEDIAPIPE_FRAMEWORK_JNI_GRAPH_PROFILER_JNI_H_
