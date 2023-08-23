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

#include "mediapipe/tasks/c/components/containers/category.h"

namespace mediapie::tasks::c::components::containers {

void CppConvertToCategory(mediapipe::tasks::components::containers::Category in,
                          Category* out) {
  out->index = in.index;
  out->score = in.score;
  out->category_name =
      in.category_name.has_value() ? in.category_name->c_str() : nullptr;
  out->display_name =
      in.display_name.has_value() ? in.display_name->c_str() : nullptr;
}

}  // namespace mediapie::tasks::c::components::containers
