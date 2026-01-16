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

#include "mediapipe/tasks/c/components/containers/classification_result_converter.h"

#include <cstdint>
#include <cstdlib>

#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/components/containers/category_converter.h"
#include "mediapipe/tasks/c/components/containers/classification_result.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"

namespace mediapipe::tasks::c::components::containers {

void CppConvertToClassificationResult(
    const mediapipe::tasks::components::containers::ClassificationResult& in,
    ClassificationResult* out) {
  out->has_timestamp_ms = in.timestamp_ms.has_value();
  out->timestamp_ms = out->has_timestamp_ms ? in.timestamp_ms.value() : 0;

  out->classifications_count = in.classifications.size();
  out->classifications = out->classifications_count
                             ? new Classifications[out->classifications_count]
                             : nullptr;

  for (uint32_t i = 0; i < out->classifications_count; ++i) {
    auto classification_in = in.classifications[i];
    auto& classification_out = out->classifications[i];

    classification_out.categories_count = classification_in.categories.size();
    classification_out.categories =
        classification_out.categories_count
            ? new Category[classification_out.categories_count]
            : nullptr;
    for (uint32_t j = 0; j < classification_out.categories_count; ++j) {
      CppConvertToCategory(classification_in.categories[j],
                           &(classification_out.categories[j]));
    }

    classification_out.head_index = classification_in.head_index;
    classification_out.head_name =
        classification_in.head_name.has_value()
            ? strdup(classification_in.head_name->c_str())
            : nullptr;
  }
}

void CppCloseClassificationResult(ClassificationResult* in) {
  for (uint32_t i = 0; i < in->classifications_count; ++i) {
    auto& classification_in = in->classifications[i];

    for (uint32_t j = 0; j < classification_in.categories_count; ++j) {
      CppCloseCategory(&classification_in.categories[j]);
    }
    delete[] classification_in.categories;
    classification_in.categories = nullptr;

    free(classification_in.head_name);
    classification_in.head_name = nullptr;
  }

  delete[] in->classifications;
  in->classifications = nullptr;
}

}  // namespace mediapipe::tasks::c::components::containers
