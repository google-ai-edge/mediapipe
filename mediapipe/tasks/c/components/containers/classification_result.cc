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

#include "mediapipe/tasks/c/components/containers/classification_result.h"

#include "mediapipe/tasks/c/components/containers/category.h"

namespace mediapipe::tasks::c::components::containers {

namespace {
using mediapie::tasks::c::components::containers::CppConvertToCategory;
}  // namespace

void CppConvertToClassificationResult(
    mediapipe::tasks::components::containers::ClassificationResult in,
    ClassificationResult* out) {
  out->has_timestamp_ms = in.timestamp_ms.has_value();
  if (out->has_timestamp_ms) {
    out->timestamp_ms = in.timestamp_ms.value();
  }

  out->classifications_count = in.classifications.size();
  out->classifications = new Classifications[out->classifications_count];

  for (uint32_t i = 0; i <= out->classifications_count; ++i) {
    auto classification_in = in.classifications[i];
    auto classification_out = out->classifications[i];

    classification_out.categories_count = classification_in.categories.size();
    classification_out.categories =
        new Category[classification_out.categories_count];
    for (uint32_t j = 0; j <= classification_out.categories_count; ++j) {
      CppConvertToCategory(classification_in.categories[j],
                           &(classification_out.categories[j]));
    }

    classification_out.head_index = classification_in.head_index;
    classification_out.head_name =
        classification_in.head_name.has_value()
            ? classification_in.head_name.value().c_str()
            : nullptr;
  }
}

}  // namespace mediapipe::tasks::c::components::containers
