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

#include "mediapipe/java/com/google/mediapipe/framework/jni/graph_texture_frame_jni.h"

#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_texture_buffer.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/jni_util.h"

using mediapipe::GlTextureBufferSharedPtr;

JNIEXPORT void JNICALL GRAPH_TEXTURE_FRAME_METHOD(nativeReleaseBuffer)(
    JNIEnv* env, jobject thiz, jlong nativeHandle) {
  GlTextureBufferSharedPtr* buffer =
      reinterpret_cast<GlTextureBufferSharedPtr*>(nativeHandle);
  delete buffer;
}

JNIEXPORT jint JNICALL GRAPH_TEXTURE_FRAME_METHOD(nativeGetTextureName)(
    JNIEnv* env, jobject thiz, jlong nativeHandle) {
  GlTextureBufferSharedPtr* buffer =
      reinterpret_cast<GlTextureBufferSharedPtr*>(nativeHandle);
  return (*buffer)->name();
}

JNIEXPORT void JNICALL GRAPH_TEXTURE_FRAME_METHOD(nativeGpuWait)(
    JNIEnv* env, jobject thiz, jlong nativeHandle) {
  GlTextureBufferSharedPtr* buffer =
      reinterpret_cast<GlTextureBufferSharedPtr*>(nativeHandle);
  (*buffer)->WaitOnGpu();
}

JNIEXPORT jint JNICALL GRAPH_TEXTURE_FRAME_METHOD(nativeGetWidth)(
    JNIEnv* env, jobject thiz, jlong nativeHandle) {
  GlTextureBufferSharedPtr* buffer =
      reinterpret_cast<GlTextureBufferSharedPtr*>(nativeHandle);
  return (*buffer)->width();
}

JNIEXPORT jint JNICALL GRAPH_TEXTURE_FRAME_METHOD(nativeGetHeight)(
    JNIEnv* env, jobject thiz, jlong nativeHandle) {
  GlTextureBufferSharedPtr* buffer =
      reinterpret_cast<GlTextureBufferSharedPtr*>(nativeHandle);
  return (*buffer)->height();
}
