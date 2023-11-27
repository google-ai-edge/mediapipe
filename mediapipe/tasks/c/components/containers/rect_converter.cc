/* Copyright 2023 The MediaPipe Authors.

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

#include "mediapipe/tasks/c/components/containers/rect_converter.h"

#include "mediapipe/tasks/c/components/containers/rect.h"
#include "mediapipe/tasks/cc/components/containers/rect.h"

namespace mediapipe::tasks::c::components::containers {

// Converts a C++ Rect to a C Rect.
void CppConvertToRect(const mediapipe::tasks::components::containers::Rect& in,
                      struct MPRect* out) {
  out->left = in.left;
  out->top = in.top;
  out->right = in.right;
  out->bottom = in.bottom;
}

// Converts a C++ RectF to a C RectF.
void CppConvertToRectF(
    const mediapipe::tasks::components::containers::RectF& in, MPRectF* out) {
  out->left = in.left;
  out->top = in.top;
  out->right = in.right;
  out->bottom = in.bottom;
}

}  // namespace mediapipe::tasks::c::components::containers
