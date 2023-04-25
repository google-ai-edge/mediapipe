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

#include "mediapipe/tasks/cc/components/containers/classification_result.h"

#include <optional>
#include <string>
#include <vector>

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/tasks/cc/components/containers/category.h"
#include "mediapipe/tasks/cc/components/containers/proto/classifications.pb.h"

namespace mediapipe::tasks::components::containers {

Classifications ConvertToClassifications(const proto::Classifications& proto) {
  Classifications classifications;
  classifications.categories.reserve(
      proto.classification_list().classification_size());
  for (const auto& classification :
       proto.classification_list().classification()) {
    classifications.categories.push_back(ConvertToCategory(classification));
  }
  classifications.head_index = proto.head_index();
  if (proto.has_head_name()) {
    classifications.head_name = proto.head_name();
  }
  return classifications;
}

Classifications ConvertToClassifications(
    const mediapipe::ClassificationList& proto, int head_index,
    std::optional<std::string> head_name) {
  Classifications classifications;
  classifications.categories.reserve(proto.classification_size());
  for (const auto& classification : proto.classification()) {
    classifications.categories.push_back(ConvertToCategory(classification));
  }
  classifications.head_index = head_index;
  classifications.head_name = head_name;
  return classifications;
}

ClassificationResult ConvertToClassificationResult(
    const proto::ClassificationResult& proto) {
  ClassificationResult classification_result;
  classification_result.classifications.reserve(proto.classifications_size());
  for (const auto& classifications : proto.classifications()) {
    classification_result.classifications.push_back(
        ConvertToClassifications(classifications));
  }
  if (proto.has_timestamp_ms()) {
    classification_result.timestamp_ms = proto.timestamp_ms();
  }
  return classification_result;
}

}  // namespace mediapipe::tasks::components::containers
