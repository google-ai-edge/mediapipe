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
//
// Defines symbols and methods related to portable portions of the framework.
//

#ifndef MEDIAPIPE_FRAMEWORK_PORT_H_
#define MEDIAPIPE_FRAMEWORK_PORT_H_

// Note: some defines, e.g. MEDIAPIPE_LITE, can only be set in the bazel rules.
// For consistency, we now set MEDIAPIPE_MOBILE there too. However, for the sake
// of projects that may want to build MediaPipe using alternative build systems,
// we also try to set platform-specific defines in this header if missing.
#if !defined(MEDIAPIPE_MOBILE) &&                                      \
    (defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX) || \
     defined(__EMSCRIPTEN__))
#define MEDIAPIPE_MOBILE
#endif

#if !defined(MEDIAPIPE_ANDROID) && defined(__ANDROID__)
#define MEDIAPIPE_ANDROID
#endif

#if defined(__APPLE__)
#include "TargetConditionals.h"  // for TARGET_OS_*
#if !defined(MEDIAPIPE_IOS) && !TARGET_OS_OSX
#define MEDIAPIPE_IOS
#endif
#endif

// These platforms do not support OpenGL ES Compute Shaders (v3.1 and up),
// but can still run OpenGL ES 3.0 and below.
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE) && \
    (defined(__APPLE__) || defined(__EMSCRIPTEN__))
#define MEDIAPIPE_DISABLE_GL_COMPUTE
#endif

#endif  // MEDIAPIPE_FRAMEWORK_PORT_H_
