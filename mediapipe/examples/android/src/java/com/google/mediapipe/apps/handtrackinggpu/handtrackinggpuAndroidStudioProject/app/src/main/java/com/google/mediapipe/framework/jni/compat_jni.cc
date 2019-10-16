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

#include "mediapipe/java/com/google/mediapipe/framework/jni/compat_jni.h"

#include <EGL/egl.h>

JNIEXPORT jlong JNICALL COMPAT_METHOD(getCurrentNativeEGLContext)(JNIEnv* env,
                                                                  jclass clz) {
  return reinterpret_cast<jlong>(eglGetCurrentContext());
}

JNIEXPORT jlong JNICALL COMPAT_METHOD(getCurrentNativeEGLSurface)(
    JNIEnv* env, jclass clz, jint readdraw) {
  return reinterpret_cast<jlong>(eglGetCurrentSurface(readdraw));
}
