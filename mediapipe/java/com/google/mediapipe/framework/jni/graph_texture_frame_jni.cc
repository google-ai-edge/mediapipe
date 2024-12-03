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

#include "absl/log/absl_log.h"
#include "absl/strings/str_format.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_context.h"
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

JNIEXPORT jlong JNICALL GRAPH_TEXTURE_FRAME_METHOD(
    nativeCreateSyncTokenForCurrentExternalContext)(JNIEnv* env, jobject thiz,
                                                    jlong nativeHandle) {
  GlTextureBufferSharedPtr* buffer =
      reinterpret_cast<GlTextureBufferSharedPtr*>(nativeHandle);
  mediapipe::GlSyncToken* token = nullptr;
  auto context_for_deletion = (*buffer)->GetProducerContext();
  // A GlTextureBuffer won't have a producer context if the contents haven't
  // been produced by MediaPipe. In that case we won't have a context to use
  // to release the sync fence.
  // TODO: get the graph's main context from the packet context?
  // Or clean up in some other way?
  if (context_for_deletion) {
    auto sync = mediapipe::GlContext::CreateSyncTokenForCurrentExternalContext(
        context_for_deletion);
    // A Java handle to a token is a raw pointer to a std::shared_ptr on the
    // heap, cast to a long. If the shared_ptr itself is null, leave the token
    // null too.
    if (sync) {
      token = new mediapipe::GlSyncToken(std::move(sync));
    }
  }
  return reinterpret_cast<jlong>(token);
}

JNIEXPORT jlong JNICALL GRAPH_TEXTURE_FRAME_METHOD(
    nativeGetCurrentExternalContextHandle)(JNIEnv* env, jobject thiz) {
  return reinterpret_cast<jlong>(
      mediapipe::GlContext::GetCurrentNativeContext());
}

JNIEXPORT void JNICALL GRAPH_TEXTURE_FRAME_METHOD(nativeDidRead)(
    JNIEnv* env, jobject thiz, jlong nativeHandle, jlong consumerSyncToken) {
  if (!consumerSyncToken) return;

  GlTextureBufferSharedPtr* buffer =
      reinterpret_cast<GlTextureBufferSharedPtr*>(nativeHandle);
  mediapipe::GlSyncToken& token =
      *reinterpret_cast<mediapipe::GlSyncToken*>(consumerSyncToken);
  // The below check attempts to detect when an invalid or already deleted
  // `consumerSyncToken` is passed. (That results in undefined behavior.
  // However, `DidRead` may succeed resulting in a later crash and masking the
  // actual problem.)
  if (token.use_count() == 0) {
    ABSL_LOG_FIRST_N(ERROR, 5)
        << absl::StrFormat("invalid sync token ref: %d", consumerSyncToken);
    return;
  }
  (*buffer)->DidRead(token);
}
