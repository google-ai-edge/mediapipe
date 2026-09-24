/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_RETRIEVAL_JNI_TEXT_CHUNKER_JNI_H_
#define MEDIAPIPE_TASKS_JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_RETRIEVAL_JNI_TEXT_CHUNKER_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jobjectArray JNICALL
Java_com_google_mediapipe_tasks_retrieval_chunking_DefaultTextChunker_nativeChunkText(
    JNIEnv* env, jclass clazz, jstring text, jint chunk_size,
    jint chunk_overlap, jint chunking_mode);

#ifdef __cplusplus
}
#endif

#endif  // MEDIAPIPE_TASKS_JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_RETRIEVAL_JNI_TEXT_CHUNKER_JNI_H_
