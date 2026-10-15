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

#ifndef JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_TEXT_TEXTSUMMARIZER_JNI_TEXT_SUMMARIZER_JNI_H_
#define JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_TEXT_TEXTSUMMARIZER_JNI_TEXT_SUMMARIZER_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define JNI_METHOD(METHOD_NAME) \
  Java_com_google_mediapipe_tasks_text_textsummarizer_TextSummarizer_##METHOD_NAME

JNIEXPORT jlong JNICALL JNI_METHOD(nativeCreateHandle)(
    JNIEnv* env, jclass clazz, jint model_fd, jstring model_path,
    jint max_num_tokens, jint mode, jstring app_id, jstring app_version,
    jint host_environment, jint host_system);

JNIEXPORT jobject JNICALL JNI_METHOD(nativeSummarize)(JNIEnv* env, jobject thiz,
                                                      jlong handle,
                                                      jstring text);

JNIEXPORT void JNICALL JNI_METHOD(nativeSummarizeStreaming)(
    JNIEnv* env, jobject thiz, jlong handle, jstring text, jobject callback);

JNIEXPORT void JNICALL JNI_METHOD(nativeClose)(JNIEnv* env, jobject thiz,
                                               jlong handle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_TEXT_TEXTSUMMARIZER_JNI_TEXT_SUMMARIZER_JNI_H_
