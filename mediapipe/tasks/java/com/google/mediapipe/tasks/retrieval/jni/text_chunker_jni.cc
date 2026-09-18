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

#include <jni.h>

#include <cstddef>
#include <string>
#include <vector>

#include "mediapipe/tasks/cc/text/utils/genai_utils.h"

using ::mediapipe::tasks::text::utils::ChunkText;
using ::mediapipe::tasks::text::utils::ChunkTextByCountingCharacters;
using ::mediapipe::tasks::text::utils::TextChunk;

enum JniChunkingMode {
  CHUNK_MODE_CHARACTER = 0,
  CHUNK_MODE_WORD = 1,
};

extern "C" {

JNIEXPORT jobjectArray JNICALL
Java_com_google_mediapipe_tasks_retrieval_chunking_DefaultTextChunker_nativeChunkText(  // NOLINT(whitespace/line_length)
    JNIEnv* env, jclass clazz, jstring j_text, jint chunk_size,
    jint chunk_overlap, jint chunking_mode) {
  if (j_text == nullptr) {
    return nullptr;
  }

  const char* text_chars = env->GetStringUTFChars(j_text, nullptr);
  if (text_chars == nullptr) {
    return nullptr;
  }
  std::string text_str(text_chars);
  env->ReleaseStringUTFChars(j_text, text_chars);

  jclass string_class = env->FindClass("java/lang/String");
  if (string_class == nullptr) {
    return nullptr;
  }

  if (chunking_mode == CHUNK_MODE_CHARACTER) {
    std::vector<std::string> chunks =
        ChunkTextByCountingCharacters(text_str, chunk_size, chunk_overlap);
    jobjectArray j_chunks =
        env->NewObjectArray(chunks.size(), string_class, nullptr);
    if (j_chunks == nullptr) {
      return nullptr;
    }
    for (size_t i = 0; i < chunks.size(); ++i) {
      jstring j_chunk = env->NewStringUTF(chunks[i].c_str());
      env->SetObjectArrayElement(j_chunks, i, j_chunk);
      env->DeleteLocalRef(j_chunk);
    }
    return j_chunks;
  } else if (chunking_mode == CHUNK_MODE_WORD) {
    int word_chunk_threshold = chunk_size;
    int word_chunk_size = chunk_size - chunk_overlap;
    std::vector<TextChunk> chunks =
        ChunkText(text_str, word_chunk_threshold, word_chunk_size);
    jobjectArray j_chunks =
        env->NewObjectArray(chunks.size(), string_class, nullptr);
    if (j_chunks == nullptr) {
      return nullptr;
    }
    for (size_t i = 0; i < chunks.size(); ++i) {
      std::string temp_str(chunks[i].text);
      jstring j_chunk = env->NewStringUTF(temp_str.c_str());
      env->SetObjectArrayElement(j_chunks, i, j_chunk);
      env->DeleteLocalRef(j_chunk);
    }
    return j_chunks;
  }

  return nullptr;
}

}  // extern "C"
