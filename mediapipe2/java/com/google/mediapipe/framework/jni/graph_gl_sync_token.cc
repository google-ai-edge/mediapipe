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

#include "mediapipe/java/com/google/mediapipe/framework/jni/graph_gl_sync_token.h"

#include <memory>

#include "mediapipe/framework/port/logging.h"
#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/jni_util.h"

JNIEXPORT void JNICALL GRAPH_GL_SYNC_TOKEN_METHOD(nativeWaitOnCpu)(
    JNIEnv* env, jclass cls, jlong syncToken) {
  mediapipe::GlSyncToken& token =
      *reinterpret_cast<mediapipe::GlSyncToken*>(syncToken);
  token->Wait();
}

JNIEXPORT void JNICALL GRAPH_GL_SYNC_TOKEN_METHOD(nativeWaitOnGpu)(
    JNIEnv* env, jclass cls, jlong syncToken) {
  mediapipe::GlSyncToken& token =
      *reinterpret_cast<mediapipe::GlSyncToken*>(syncToken);
  token->WaitOnGpu();
}

JNIEXPORT void JNICALL GRAPH_GL_SYNC_TOKEN_METHOD(nativeRelease)(
    JNIEnv* env, jclass cls, jlong syncToken) {
  delete reinterpret_cast<mediapipe::GlSyncToken*>(syncToken);
}
