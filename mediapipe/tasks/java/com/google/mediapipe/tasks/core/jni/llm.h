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

#ifndef JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_CORE_JNI_LLM_H_
#define JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_CORE_JNI_LLM_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define JNI_METHOD(METHOD_NAME) \
  Java_com_google_mediapipe_tasks_core_LlmTaskRunner_##METHOD_NAME

/*
 * Class:     com_google_mediapipe_tasks_core_LlmTaskRunner
 * Method:    nativeCreateSession
 * Signature: ([B)J
 */
JNIEXPORT jlong JNICALL JNI_METHOD(nativeCreateSession)(JNIEnv *, jclass,
                                                        jbyteArray);

/*
 * Class:     com_google_mediapipe_tasks_core_LlmTaskRunner
 * Method:    nativeDeleteSession
 * Signature: (J)V
 */
JNIEXPORT void JNICALL JNI_METHOD(nativeDeleteSession)(JNIEnv *, jclass, jlong);

/*
 * Class:     com_google_mediapipe_tasks_core_LlmTaskRunner
 * Method:    nativePredictSync
 * Signature: (JLjava/lang/String;)[B
 */
JNIEXPORT jbyteArray JNICALL JNI_METHOD(nativePredictSync)(JNIEnv *, jclass,
                                                           jlong, jstring);

/*
 * Class:     com_google_mediapipe_tasks_core_LlmTaskRunner
 * Method:    nativeRegisterCallback
 * Signature: (Ljava/lang/Object;)Ljava/lang/Object
 */
JNIEXPORT jobject JNICALL JNI_METHOD(nativeRegisterCallback)(JNIEnv *, jclass,
                                                             jobject);

/*
 * Class:     com_google_mediapipe_tasks_core_LlmTaskRunner
 * Method:    nativeRemoveCallback
 * Signature: (Ljava/lang/Object;)V
 */
JNIEXPORT void JNICALL JNI_METHOD(nativeRemoveCallback)(JNIEnv *, jclass,
                                                        jobject);

/*
 * Class:     com_google_mediapipe_tasks_core_LlmTaskRunner
 * Method:    nativePredictAsync
 * Signature: (JLjava/lang/Object;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL JNI_METHOD(nativePredictAsync)(JNIEnv *, jclass, jlong,
                                                      jobject, jstring);

/*
 * Class:     com_google_mediapipe_tasks_core_LlmTaskRunner
 * Method:    nativeSizeInTokens
 * Signature: (JLjava/lang/String;)I
 */
JNIEXPORT jint JNICALL JNI_METHOD(nativeSizeInTokens)(JNIEnv *, jclass, jlong,
                                                      jstring);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_CORE_JNI_LLM_H_
