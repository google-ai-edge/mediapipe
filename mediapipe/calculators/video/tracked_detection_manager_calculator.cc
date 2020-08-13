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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/node_hash_map.h"
#include "mediapipe/calculators/video/tracked_detection_manager_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/tracking/box_tracker.h"
#include "mediapipe/util/tracking/tracked_detection.h"
#include "mediapipe/util/tracking/tracked_detection_manager.h"
#include "mediapipe/util/tracking/tracking.h"

namespace mediapipe {
namespace {

constexpr int kDetectionUpdateTimeOutMS = 5000;
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kDetectionBoxesTag[] = "DETECTION_BOXES";
constexpr char kDetectionListTag[] = "DETECTION_LIST";
constexpr char kTrackingBoxesTag[] = "TRACKING_BOXES";
constexpr char kCancelObjectIdTag[] = "CANCEL_OBJECT_ID";

// Move |src| to the back of |dst|.
void MoveIds(std::vector<int>* dst, std::vector<int> src) {
  dst->insert(dst->end(), std::make_move_iterator(src.begin()),
              std::make_move_iterator(src.end()));
}

int64 GetInputTimestampMs(::mediapipe::CalculatorContext* cc) {
  return cc->InputTimestamp().Microseconds() / 1000;  // 1 ms = 1000 us.
}

// Converts a Mediapipe Detection Proto to a TrackedDetection class.
std::unique_ptr<TrackedDetection> GetTrackedDetectionFromDetection(
    const Detection& detection, int64 timestamp) {
  std::unique_ptr<TrackedDetection> tracked_detection =
      absl::make_unique<TrackedDetection>(detection.detection_id(), timestamp);
  const float top = detection.location_data().relative_bounding_box().ymin();
  const float bottom =
      detection.location_data().relative_bounding_box().ymin() +
      detection.location_data().relative_bounding_box().height();
  const float left = detection.location_data().relative_bounding_box().xmin();
  const float right = detection.location_data().relative_bounding_box().xmin() +
                      detection.location_data().relative_bounding_box().width();
  NormalizedRect bounding_box;
  bounding_box.set_x_center((left + right) / 2.f);
  bounding_box.set_y_center((top + bottom) / 2.f);
  bounding_box.set_height(bottom - top);
  bounding_box.set_width(right - left);
  tracked_detection->set_bounding_box(bounding_box);

  for (int i = 0; i < detection.label_size(); ++i) {
    tracked_detection->AddLabel(detection.label(i), detection.score(i));
  }
  return tracked_detection;
}

// Converts a TrackedDetection class to a Mediapipe Detection Proto.
Detection GetAxisAlignedDetectionFromTrackedDetection(
    const TrackedDetection& tracked_detection) {
  Detection detection;
  LocationData* location_data = detection.mutable_location_data();

  auto corners = tracked_detection.GetCorners();

  float x_min = std::numeric_limits<float>::max();
  float x_max = std::numeric_limits<float>::min();
  float y_min = std::numeric_limits<float>::max();
  float y_max = std::numeric_limits<float>::min();
  for (int i = 0; i < 4; ++i) {
    x_min = std::min(x_min, corners[i].x());
    x_max = std::max(x_max, corners[i].x());
    y_min = std::min(y_min, corners[i].y());
    y_max = std::max(y_max, corners[i].y());
  }
  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  LocationData::RelativeBoundingBox* relative_bbox =
      location_data->mutable_relative_bounding_box();
  relative_bbox->set_xmin(x_min);
  relative_bbox->set_ymin(y_min);
  relative_bbox->set_width(x_max - x_min);
  relative_bbox->set_height(y_max - y_min);

  // Use previous id which is the id the object when it's first detected.
  if (tracked_detection.previous_id() > 0) {
    detection.set_detection_id(tracked_detection.previous_id());
  } else {
    detection.set_detection_id(tracked_detection.unique_id());
  }
  for (const auto& label_and_score : tracked_detection.label_to_score_map()) {
    detection.add_label(label_and_score.first);
    detection.add_score(label_and_score.second);
  }
  return detection;
}

}  // namespace

// TrackedDetectionManagerCalculator accepts detections and tracking results at
// different frame rate for real time tracking of targets.
// Input:
//   DETECTIONS: A vector<Detection> of newly detected targets.
//   TRACKING_BOXES: A TimedBoxProtoList which contains a list of tracked boxes
//   from previous detections.
//
// Output:
//   CANCEL_OBJECT_ID: Ids of targets that are missing/lost such that it should
//   be removed from tracking.
//   DETECTIONS: List of detections that are being tracked.
//   DETECTION_BOXES: List of bounding boxes of detections that are being
//   tracked.
//
// Usage example:
// node {
//   calculator: "TrackedDetectionManagerCalculator"
//   input_stream: "DETECTIONS:detections"
//   input_stream: "TRACKING_BOXES:boxes"
//   output_stream: "CANCEL_OBJECT_ID:cancel_object_id"
//   output_stream: "DETECTIONS:output_detections"
// }
class TrackedDetectionManagerCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Open(CalculatorContext* cc) override;

  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  // Adds new list of detections to |waiting_for_update_detections_|.
  void AddDetectionList(const DetectionList& detection_list,
                        CalculatorContext* cc);
  void AddDetections(const std::vector<Detection>& detections,
                     CalculatorContext* cc);

  // Manages existing and new detections.
  TrackedDetectionManager tracked_detection_manager_;

  // Set of detections that are not up to date yet. These detections will be
  // added to the detection manager until they got updated from the box tracker.
  absl::node_hash_map<int, std::unique_ptr<TrackedDetection>>
      waiting_for_update_detections_;
};
REGISTER_CALCULATOR(TrackedDetectionManagerCalculator);

::mediapipe::Status TrackedDetectionManagerCalculator::GetContract(
    CalculatorContract* cc) {
  if (cc->Inputs().HasTag(kDetectionsTag)) {
    cc->Inputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();
  }
  if (cc->Inputs().HasTag(kDetectionListTag)) {
    cc->Inputs().Tag(kDetectionListTag).Set<DetectionList>();
  }
  if (cc->Inputs().HasTag(kTrackingBoxesTag)) {
    cc->Inputs().Tag(kTrackingBoxesTag).Set<TimedBoxProtoList>();
  }

  if (cc->Outputs().HasTag(kCancelObjectIdTag)) {
    cc->Outputs().Tag(kCancelObjectIdTag).Set<int>();
  }
  if (cc->Outputs().HasTag(kDetectionsTag)) {
    cc->Outputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();
  }
  if (cc->Outputs().HasTag(kDetectionBoxesTag)) {
    cc->Outputs().Tag(kDetectionBoxesTag).Set<std::vector<NormalizedRect>>();
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TrackedDetectionManagerCalculator::Open(
    CalculatorContext* cc) {
  mediapipe::TrackedDetectionManagerCalculatorOptions options =
      cc->Options<mediapipe::TrackedDetectionManagerCalculatorOptions>();
  tracked_detection_manager_.SetConfig(
      options.tracked_detection_manager_options());
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TrackedDetectionManagerCalculator::Process(
    CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kTrackingBoxesTag) &&
      !cc->Inputs().Tag(kTrackingBoxesTag).IsEmpty()) {
    const TimedBoxProtoList& tracked_boxes =
        cc->Inputs().Tag(kTrackingBoxesTag).Get<TimedBoxProtoList>();

    // Collect all detections that are removed.
    auto removed_detection_ids = absl::make_unique<std::vector<int>>();
    for (const TimedBoxProto& tracked_box : tracked_boxes.box()) {
      NormalizedRect bounding_box;
      bounding_box.set_x_center((tracked_box.left() + tracked_box.right()) /
                                2.f);
      bounding_box.set_y_center((tracked_box.bottom() + tracked_box.top()) /
                                2.f);
      bounding_box.set_height(tracked_box.bottom() - tracked_box.top());
      bounding_box.set_width(tracked_box.right() - tracked_box.left());
      bounding_box.set_rotation(tracked_box.rotation());
      // First check if this box updates a detection that's waiting for
      // update from the tracker.
      auto waiting_for_update_detectoin_ptr =
          waiting_for_update_detections_.find(tracked_box.id());
      if (waiting_for_update_detectoin_ptr !=
          waiting_for_update_detections_.end()) {
        // Add the detection and remove duplicated detections.
        auto removed_ids = tracked_detection_manager_.AddDetection(
            std::move(waiting_for_update_detectoin_ptr->second));
        MoveIds(removed_detection_ids.get(), std::move(removed_ids));

        waiting_for_update_detections_.erase(waiting_for_update_detectoin_ptr);
      }
      auto removed_ids = tracked_detection_manager_.UpdateDetectionLocation(
          tracked_box.id(), bounding_box, tracked_box.time_msec());
      MoveIds(removed_detection_ids.get(), std::move(removed_ids));
    }
    // TODO: Should be handled automatically in detection manager.
    auto removed_ids = tracked_detection_manager_.RemoveObsoleteDetections(
        GetInputTimestampMs(cc) - kDetectionUpdateTimeOutMS);
    MoveIds(removed_detection_ids.get(), std::move(removed_ids));

    // TODO: Should be handled automatically in detection manager.
    removed_ids = tracked_detection_manager_.RemoveOutOfViewDetections();
    MoveIds(removed_detection_ids.get(), std::move(removed_ids));

    if (!removed_detection_ids->empty() &&
        cc->Outputs().HasTag(kCancelObjectIdTag)) {
      auto timestamp = cc->InputTimestamp();
      for (int box_id : *removed_detection_ids) {
        // The timestamp is incremented (by 1 us) because currently the box
        // tracker calculator only accepts one cancel object ID for any given
        // timestamp.
        cc->Outputs()
            .Tag(kCancelObjectIdTag)
            .AddPacket(mediapipe::MakePacket<int>(box_id).At(timestamp++));
      }
    }

    // Output detections and corresponding bounding boxes.
    const auto& all_detections =
        tracked_detection_manager_.GetAllTrackedDetections();
    auto output_detections = absl::make_unique<std::vector<Detection>>();
    auto output_boxes = absl::make_unique<std::vector<NormalizedRect>>();

    for (const auto& detection_ptr : all_detections) {
      const auto& detection = *detection_ptr.second;
      // Only output detections that are synced.
      if (detection.last_updated_timestamp() <
          cc->InputTimestamp().Microseconds() / 1000) {
        continue;
      }
      output_detections->emplace_back(
          GetAxisAlignedDetectionFromTrackedDetection(detection));
      output_boxes->emplace_back(detection.bounding_box());
    }
    if (cc->Outputs().HasTag(kDetectionsTag)) {
      cc->Outputs()
          .Tag(kDetectionsTag)
          .Add(output_detections.release(), cc->InputTimestamp());
    }

    if (cc->Outputs().HasTag(kDetectionBoxesTag)) {
      cc->Outputs()
          .Tag(kDetectionBoxesTag)
          .Add(output_boxes.release(), cc->InputTimestamp());
    }
  }

  if (cc->Inputs().HasTag(kDetectionsTag) &&
      !cc->Inputs().Tag(kDetectionsTag).IsEmpty()) {
    const auto detections =
        cc->Inputs().Tag(kDetectionsTag).Get<std::vector<Detection>>();
    AddDetections(detections, cc);
  }

  if (cc->Inputs().HasTag(kDetectionListTag) &&
      !cc->Inputs().Tag(kDetectionListTag).IsEmpty()) {
    const auto detection_list =
        cc->Inputs().Tag(kDetectionListTag).Get<DetectionList>();
    AddDetectionList(detection_list, cc);
  }

  return ::mediapipe::OkStatus();
}

void TrackedDetectionManagerCalculator::AddDetectionList(
    const DetectionList& detection_list, CalculatorContext* cc) {
  for (const auto& detection : detection_list.detection()) {
    // Convert from microseconds to milliseconds.
    std::unique_ptr<TrackedDetection> new_detection =
        GetTrackedDetectionFromDetection(
            detection, cc->InputTimestamp().Microseconds() / 1000);

    const int id = new_detection->unique_id();
    waiting_for_update_detections_[id] = std::move(new_detection);
  }
}

void TrackedDetectionManagerCalculator::AddDetections(
    const std::vector<Detection>& detections, CalculatorContext* cc) {
  for (const auto& detection : detections) {
    // Convert from microseconds to milliseconds.
    std::unique_ptr<TrackedDetection> new_detection =
        GetTrackedDetectionFromDetection(
            detection, cc->InputTimestamp().Microseconds() / 1000);

    const int id = new_detection->unique_id();
    waiting_for_update_detections_[id] = std::move(new_detection);
  }
}

}  // namespace mediapipe
