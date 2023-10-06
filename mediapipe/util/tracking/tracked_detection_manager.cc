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

#include "mediapipe/util/tracking/tracked_detection_manager.h"

#include <vector>

#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/util/tracking/tracked_detection.h"

namespace {

using mediapipe::TrackedDetection;

// Checks if a point is out of view.
// x and y should both be in [0, 1] to be considered in view.
bool IsPointOutOfView(float x, float y) {
  return x < 0.0f || x > 1.0f || y < 0.0f || y > 1.0f;
}

// Checks if all corners of a bounding box of an object are out of view.
bool AreCornersOutOfView(const TrackedDetection& object) {
  std::array<Vector2_f, 4> corners = object.GetCorners();
  for (int k = 0; k < 4; ++k) {
    if (!IsPointOutOfView(corners[k].x(), corners[k].y())) {
      return false;
    }
  }
  return true;
}
}  // namespace

namespace mediapipe {

std::vector<int> TrackedDetectionManager::AddDetection(
    std::unique_ptr<TrackedDetection> detection) {
  std::vector<int> ids_to_remove;

  int64_t latest_duplicate_timestamp = 0;
  // TODO: All detections should be fastforwarded to the current
  // timestamp before adding the detection manager. E.g. only check they are the
  // same if the timestamp are the same.
  for (auto& existing_detection_ptr : detections_) {
    const auto& existing_detection = *existing_detection_ptr.second;
    if (detection->IsSameAs(existing_detection,
                            config_.is_same_detection_max_area_ratio(),
                            config_.is_same_detection_min_overlap_ratio())) {
      // Merge previous labels to the new detection. Because new detections
      // usually have better bounding box than the one from tracking.
      // TODO: This might cause unstable change of the bounding box.
      // TODO: Add a filter for the box using new detection as
      // observation.
      detection->MergeLabelScore(existing_detection);
      // Pick the duplicated detection that has the latest initial timestamp as
      // the previous detection of the new one.
      if (existing_detection.initial_timestamp() > latest_duplicate_timestamp) {
        latest_duplicate_timestamp = existing_detection.initial_timestamp();
        if (existing_detection.previous_id() == -1) {
          detection->set_previous_id(existing_detection.unique_id());
        } else {
          detection->set_previous_id(existing_detection.previous_id());
        }
      }
      ids_to_remove.push_back(existing_detection_ptr.first);
    }
  }
  // Erase old detections.
  for (auto id : ids_to_remove) {
    detections_.erase(id);
  }
  const int id = detection->unique_id();
  detections_[id] = std::move(detection);
  return ids_to_remove;
}

std::vector<int> TrackedDetectionManager::UpdateDetectionLocation(
    int id, const NormalizedRect& bounding_box, int64_t timestamp) {
  // TODO: Remove all boxes that are not updating.
  auto detection_ptr = detections_.find(id);
  if (detection_ptr == detections_.end()) {
    return std::vector<int>();
  }
  auto& detection = *detection_ptr->second;
  detection.set_bounding_box(bounding_box);
  detection.set_last_updated_timestamp(timestamp);

  // It's required to do this here in addition to in AddDetection because during
  // fast motion, two or more detections of the same object could coexist since
  // the locations could be quite different before they are propagated to the
  // same timestamp.
  return RemoveDuplicatedDetections(detection.unique_id());
}

std::vector<int> TrackedDetectionManager::RemoveObsoleteDetections(
    int64_t timestamp) {
  std::vector<int> ids_to_remove;
  for (auto& existing_detection : detections_) {
    if (existing_detection.second->last_updated_timestamp() < timestamp) {
      ids_to_remove.push_back(existing_detection.first);
    }
  }
  for (auto idx : ids_to_remove) {
    detections_.erase(idx);
  }
  return ids_to_remove;
}

std::vector<int> TrackedDetectionManager::RemoveOutOfViewDetections() {
  std::vector<int> ids_to_remove;
  for (auto& existing_detection : detections_) {
    if (AreCornersOutOfView(*existing_detection.second)) {
      ids_to_remove.push_back(existing_detection.first);
    }
  }
  for (auto idx : ids_to_remove) {
    detections_.erase(idx);
  }
  return ids_to_remove;
}

std::vector<int> TrackedDetectionManager::RemoveDuplicatedDetections(int id) {
  std::vector<int> ids_to_remove;
  auto detection_ptr = detections_.find(id);
  if (detection_ptr == detections_.end()) {
    return ids_to_remove;
  }
  auto& detection = *detection_ptr->second;

  // For duplciated detections, we keep the one that's added most recently.
  auto latest_detection = detection_ptr->second.get();
  // For setting up the |previous_id| of the latest detection. For now, if there
  // are multiple duplicated detections at the same timestamp, we will use the
  // one that has the second latest initial timestamp
  const TrackedDetection* previous_detection = nullptr;
  for (auto& existing_detection : detections_) {
    auto other = *(existing_detection.second);
    if (detection.unique_id() != other.unique_id()) {
      // Only check if they are updated at the same timestamp. Comparing
      // locations of detections at different timestamp is not correct.
      if (detection.last_updated_timestamp() ==
          other.last_updated_timestamp()) {
        if (detection.IsSameAs(other,
                               config_.is_same_detection_max_area_ratio(),
                               config_.is_same_detection_min_overlap_ratio())) {
          const TrackedDetection* detection_to_remove = nullptr;
          if (latest_detection->initial_timestamp() >=
              other.initial_timestamp()) {
            // Removes the earlier one.
            ids_to_remove.push_back(other.unique_id());
            detection_to_remove = existing_detection.second.get();
            latest_detection->MergeLabelScore(other);
          } else {
            ids_to_remove.push_back(latest_detection->unique_id());
            detection_to_remove = latest_detection;
            existing_detection.second->MergeLabelScore(*latest_detection);
            latest_detection = existing_detection.second.get();
          }
          if (!previous_detection ||
              previous_detection->initial_timestamp() <
                  detection_to_remove->initial_timestamp()) {
            previous_detection = detection_to_remove;
          }
        }
      }
    }
  }

  // If the latest detection is not the one passed into this function, it might
  // already has its previous detection. In that case, we don't override it.
  if (latest_detection->previous_id() == -1 && previous_detection) {
    if (previous_detection->previous_id() == -1) {
      latest_detection->set_previous_id(previous_detection->unique_id());
    } else {
      latest_detection->set_previous_id(previous_detection->previous_id());
    }
  }

  for (auto idx : ids_to_remove) {
    detections_.erase(idx);
  }
  return ids_to_remove;
}

}  // namespace mediapipe
