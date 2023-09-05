// Copyright 2020 The MediaPipe Authors.
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

#include "mediapipe/modules/objectron/calculators/frame_annotation_tracker.h"

#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "mediapipe/modules/objectron/calculators/annotation_data.pb.h"
#include "mediapipe/modules/objectron/calculators/box_util.h"
#include "mediapipe/util/tracking/box_tracker.pb.h"

namespace mediapipe {

void FrameAnnotationTracker::AddDetectionResult(
    const FrameAnnotation& frame_annotation) {
  const int64_t time_us =
      static_cast<int64_t>(std::round(frame_annotation.timestamp()));
  for (const auto& object_annotation : frame_annotation.annotations()) {
    detected_objects_[time_us + object_annotation.object_id()] =
        object_annotation;
  }
}

FrameAnnotation FrameAnnotationTracker::ConsolidateTrackingResult(
    const TimedBoxProtoList& tracked_boxes,
    absl::flat_hash_set<int>* cancel_object_ids) {
  ABSL_CHECK(cancel_object_ids != nullptr);
  FrameAnnotation frame_annotation;
  std::vector<int64_t> keys_to_be_deleted;
  for (const auto& detected_obj : detected_objects_) {
    const int object_id = detected_obj.second.object_id();
    if (cancel_object_ids->contains(object_id)) {
      // Remember duplicated detections' keys.
      keys_to_be_deleted.push_back(detected_obj.first);
      continue;
    }
    TimedBoxProto ref_box;
    for (const auto& box : tracked_boxes.box()) {
      if (box.id() == object_id) {
        ref_box = box;
        break;
      }
    }
    if (!ref_box.has_id() || ref_box.id() < 0) {
      ABSL_LOG(ERROR) << "Can't find matching tracked box for object id: "
                      << object_id << ". Likely lost tracking of it.";
      keys_to_be_deleted.push_back(detected_obj.first);
      continue;
    }

    // Find duplicated boxes
    for (const auto& box : tracked_boxes.box()) {
      if (box.id() != object_id) {
        if (ComputeBoxIoU(ref_box, box) > iou_threshold_) {
          cancel_object_ids->insert(box.id());
        }
      }
    }

    // Map ObjectAnnotation from detection to tracked time.
    // First, gather all keypoints from source detection.
    std::vector<cv::Point2f> key_points;
    for (const auto& keypoint : detected_obj.second.keypoints()) {
      key_points.push_back(
          cv::Point2f(keypoint.point_2d().x(), keypoint.point_2d().y()));
    }
    // Second, find source box.
    TimedBoxProto src_box;
    ComputeBoundingRect(key_points, &src_box);
    ObjectAnnotation* tracked_obj = frame_annotation.add_annotations();
    tracked_obj->set_object_id(ref_box.id());
    // Finally, map all keypoints in the source detection to tracked location.
    for (const auto& keypoint : detected_obj.second.keypoints()) {
      cv::Point2f dst = MapPoint(
          src_box, ref_box,
          cv::Point2f(keypoint.point_2d().x(), keypoint.point_2d().y()),
          img_width_, img_height_);
      auto* dst_point = tracked_obj->add_keypoints()->mutable_point_2d();
      dst_point->set_x(dst.x);
      dst_point->set_y(dst.y);
    }
  }

  for (const auto& key : keys_to_be_deleted) {
    detected_objects_.erase(key);
  }

  return frame_annotation;
}

}  // namespace mediapipe
