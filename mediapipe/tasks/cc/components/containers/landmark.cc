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

#include "mediapipe/tasks/cc/components/containers/landmark.h"

#include <optional>
#include <utility>

#include "mediapipe/framework/formats/landmark.pb.h"

namespace mediapipe::tasks::components::containers {

Landmark ConvertToLandmark(const mediapipe::Landmark& proto) {
  return {/*x=*/proto.x(), /*y=*/proto.y(), /*z=*/proto.z(),
          /*visibility=*/proto.has_visibility()
              ? std::optional<float>(proto.visibility())
              : std::nullopt,
          /*presence=*/proto.has_presence()
              ? std::optional<float>(proto.presence())
              : std::nullopt};
}

NormalizedLandmark ConvertToNormalizedLandmark(
    const mediapipe::NormalizedLandmark& proto) {
  return {/*x=*/proto.x(), /*y=*/proto.y(), /*z=*/proto.z(),
          /*visibility=*/proto.has_visibility()
              ? std::optional<float>(proto.visibility())
              : std::nullopt,
          /*presence=*/proto.has_presence()
              ? std::optional<float>(proto.presence())
              : std::nullopt};
}

Landmarks ConvertToLandmarks(const mediapipe::LandmarkList& proto) {
  Landmarks landmarks;
  landmarks.landmarks.reserve(proto.landmark_size());
  for (const auto& landmark : proto.landmark()) {
    landmarks.landmarks.push_back(ConvertToLandmark(landmark));
  }
  return landmarks;
}

NormalizedLandmarks ConvertToNormalizedLandmarks(
    const mediapipe::NormalizedLandmarkList& proto) {
  NormalizedLandmarks landmarks;
  landmarks.landmarks.reserve(proto.landmark_size());
  for (const auto& landmark : proto.landmark()) {
    landmarks.landmarks.push_back(ConvertToNormalizedLandmark(landmark));
  }
  return landmarks;
}

}  // namespace mediapipe::tasks::components::containers
