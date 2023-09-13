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

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"

namespace mediapipe {
namespace api2 {
namespace {

struct BoundingBoxHash {
  size_t operator()(const LocationData::BoundingBox& bbox) const {
    return std::hash<int>{}(bbox.xmin()) ^ std::hash<int>{}(bbox.ymin()) ^
           std::hash<int>{}(bbox.width()) ^ std::hash<int>{}(bbox.height());
  }
};

struct BoundingBoxEq {
  bool operator()(const LocationData::BoundingBox& lhs,
                  const LocationData::BoundingBox& rhs) const {
    return lhs.xmin() == rhs.xmin() && lhs.ymin() == rhs.ymin() &&
           lhs.width() == rhs.width() && lhs.height() == rhs.height();
  }
};

}  // namespace

// This Calculator deduplicates the bunding boxes with exactly the same
// coordinates, and folds the labels into a single Detection proto. Note
// non-maximum-suppression remove the overlapping bounding boxes within a class,
// while the deduplication operation merges bounding boxes from different
// classes.

// Example config:
// node {
//   calculator: "DetectionsDeduplicateCalculator"
//   input_stream: "detections"
//   output_stream: "deduplicated_detections"
// }
class DetectionsDeduplicateCalculator : public Node {
 public:
  static constexpr Input<std::vector<Detection>> kIn{""};
  static constexpr Output<std::vector<Detection>> kOut{""};

  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  absl::Status Open(mediapipe::CalculatorContext* cc) {
    cc->SetOffset(::mediapipe::TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Process(mediapipe::CalculatorContext* cc) {
    const std::vector<Detection>& raw_detections = kIn(cc).Get();
    absl::flat_hash_map<LocationData::BoundingBox, Detection*, BoundingBoxHash,
                        BoundingBoxEq>
        bbox_to_detections;
    std::vector<Detection> deduplicated_detections;
    for (const auto& detection : raw_detections) {
      if (!detection.has_location_data() ||
          !detection.location_data().has_bounding_box()) {
        return absl::InvalidArgumentError(
            "The location data of Detections must be BoundingBox.");
      }
      if (bbox_to_detections.contains(
              detection.location_data().bounding_box())) {
        // The bbox location already exists. Merge the detection labels into
        // the existing detection proto.
        Detection& deduplicated_detection =
            *bbox_to_detections[detection.location_data().bounding_box()];
        deduplicated_detection.mutable_score()->MergeFrom(detection.score());
        deduplicated_detection.mutable_label()->MergeFrom(detection.label());
        deduplicated_detection.mutable_label_id()->MergeFrom(
            detection.label_id());
        deduplicated_detection.mutable_display_name()->MergeFrom(
            detection.display_name());
      } else {
        // The bbox location appears first time. Add the detection to output
        // detection vector.
        deduplicated_detections.push_back(detection);
        bbox_to_detections[detection.location_data().bounding_box()] =
            &deduplicated_detections.back();
      }
    }
    kOut(cc).Send(std::move(deduplicated_detections));
    return absl::OkStatus();
  }
};

MEDIAPIPE_REGISTER_NODE(DetectionsDeduplicateCalculator);

}  // namespace api2
}  // namespace mediapipe
