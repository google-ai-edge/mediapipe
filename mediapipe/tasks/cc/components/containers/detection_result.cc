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

#include "mediapipe/tasks/cc/components/containers/detection_result.h"

#include <optional>
#include <string>
#include <vector>

#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/tasks/cc/components/containers/category.h"
#include "mediapipe/tasks/cc/components/containers/rect.h"

namespace mediapipe::tasks::components::containers {

constexpr int kDefaultCategoryIndex = -1;

Detection ConvertToDetection(const mediapipe::Detection& detection_proto) {
  Detection detection;
  for (int idx = 0; idx < detection_proto.score_size(); ++idx) {
    detection.categories.push_back(
        {/* index= */ detection_proto.label_id_size() > idx
             ? detection_proto.label_id(idx)
             : kDefaultCategoryIndex,
         /* score= */ detection_proto.score(idx),
         /* category_name */ detection_proto.label_size() > idx
             ? detection_proto.label(idx)
             : "",
         /* display_name */ detection_proto.display_name_size() > idx
             ? detection_proto.display_name(idx)
             : ""});
  }
  Rect bounding_box;
  if (detection_proto.location_data().has_bounding_box()) {
    mediapipe::LocationData::BoundingBox bounding_box_proto =
        detection_proto.location_data().bounding_box();
    bounding_box.left = bounding_box_proto.xmin();
    bounding_box.top = bounding_box_proto.ymin();
    bounding_box.right = bounding_box_proto.xmin() + bounding_box_proto.width();
    bounding_box.bottom =
        bounding_box_proto.ymin() + bounding_box_proto.height();
  }
  detection.bounding_box = bounding_box;
  if (!detection_proto.location_data().relative_keypoints().empty()) {
    detection.keypoints = std::vector<NormalizedKeypoint>();
    detection.keypoints->reserve(
        detection_proto.location_data().relative_keypoints_size());
    for (const auto& keypoint :
         detection_proto.location_data().relative_keypoints()) {
      detection.keypoints->push_back(
          {keypoint.x(), keypoint.y(),
           keypoint.has_keypoint_label()
               ? std::make_optional(keypoint.keypoint_label())
               : std::nullopt,
           keypoint.has_score() ? std::make_optional(keypoint.score())
                                : std::nullopt});
    }
  }
  return detection;
}

DetectionResult ConvertToDetectionResult(
    std::vector<mediapipe::Detection> detections_proto) {
  DetectionResult detection_result;
  detection_result.detections.reserve(detections_proto.size());
  for (const auto& detection_proto : detections_proto) {
    detection_result.detections.push_back(ConvertToDetection(detection_proto));
  }
  return detection_result;
}
}  // namespace mediapipe::tasks::components::containers
