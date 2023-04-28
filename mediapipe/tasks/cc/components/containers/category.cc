/* Copyright 2022 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/components/containers/category.h"

#include <optional>
#include <string>

#include "mediapipe/framework/formats/classification.pb.h"

namespace mediapipe::tasks::components::containers {

Category ConvertToCategory(const mediapipe::Classification& proto) {
  Category category;
  category.index = proto.index();
  category.score = proto.score();
  if (proto.has_label()) {
    category.category_name = proto.label();
  }
  if (proto.has_display_name()) {
    category.display_name = proto.display_name();
  }
  return category;
}

}  // namespace mediapipe::tasks::components::containers
