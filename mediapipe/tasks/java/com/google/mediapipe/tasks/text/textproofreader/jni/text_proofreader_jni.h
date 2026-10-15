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

#ifndef MEDIAPIPE_TASKS_JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_TEXT_TEXTPROOFREADER_JNI_TEXT_PROOFREADER_JNI_H_
#define MEDIAPIPE_TASKS_JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_TEXT_TEXTPROOFREADER_JNI_TEXT_PROOFREADER_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

#define JNI_METHOD(METHOD_NAME) \
  Java_com_google_mediapipe_tasks_text_textproofreader_TextProofreader_##METHOD_NAME

// Creates a TextProofreader and returns a Handle (pointer).
JNIEXPORT jlong JNICALL JNI_METHOD(nativeCreateHandle)(
    JNIEnv* env, jclass clazz, jint model_fd, jstring model_path,
    jint max_num_tokens, jstring app_id, jstring app_version,
    jint host_environment, jint host_system);

// Performs proofreading synchronously.
JNIEXPORT jobject JNICALL JNI_METHOD(nativeProofread)(JNIEnv* env, jclass clazz,
                                                      jlong handle,
                                                      jstring text);

// Performs proofreading asynchronously and streams the results.
JNIEXPORT void JNICALL JNI_METHOD(nativeProofreadStreaming)(
    JNIEnv* env, jclass clazz, jlong handle, jstring text, jobject callback);

// Closes and cleans up a TextProofreader.
JNIEXPORT void JNICALL JNI_METHOD(nativeClose)(JNIEnv* env, jclass clazz,
                                               jlong handle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MEDIAPIPE_TASKS_JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_TEXT_TEXTPROOFREADER_JNI_TEXT_PROOFREADER_JNI_H_
