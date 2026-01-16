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

#include "mediapipe/util/tracking/tracked_detection.h"

#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe {

namespace {

using ::mediapipe::NormalizedRect;

// Struct for carrying boundary information.
struct NormalizedRectBounds {
  float left, right, top, bottom;
};

// Computes the area of a NormalizedRect.
float BoxArea(const NormalizedRect& box) { return box.width() * box.height(); }

// Computes the bounds of a NormalizedRect.
void GetNormalizedRectBounds(const NormalizedRect& box,
                             NormalizedRectBounds* bounds) {
  bounds->left = box.x_center() - box.width() / 2.f;
  bounds->right = box.x_center() + box.width() / 2.f;
  bounds->top = box.y_center() - box.height() / 2.f;
  bounds->bottom = box.y_center() + box.height() / 2.f;
}

// Computes the overlapping area of two boxes, ignoring rotation.
float OverlapArea(const NormalizedRect& box1, const NormalizedRect& box2) {
  NormalizedRectBounds bounds1, bounds2;
  GetNormalizedRectBounds(box1, &bounds1);
  GetNormalizedRectBounds(box2, &bounds2);
  const float x_overlap =
      std::max(0.f, std::min(bounds1.right, bounds2.right) -
                        std::max(bounds1.left, bounds2.left));
  const float y_overlap =
      std::max(0.f, std::min(bounds1.bottom, bounds2.bottom) -
                        std::max(bounds1.top, bounds2.top));
  return x_overlap * y_overlap;
}

std::array<Vector2_f, 4> ComputeCorners(const NormalizedRect& normalized_box,
                                        const Vector2_f& center,
                                        int image_width, int image_height) {
  NormalizedRectBounds bounds;
  GetNormalizedRectBounds(normalized_box, &bounds);
  // Rotate 4 corner w.r.t. center.
  std::array<Vector2_f, 4> corners{{
      Vector2_f(bounds.left * image_width, bounds.top * image_height),
      Vector2_f(bounds.left * image_width, bounds.bottom * image_height),
      Vector2_f(bounds.right * image_width, bounds.bottom * image_height),
      Vector2_f(bounds.right * image_width, bounds.top * image_height),
  }};
  if (std::abs(normalized_box.rotation()) <= 1e-5) {
    return corners;
  }

  const float cos_a = std::cos(normalized_box.rotation());
  const float sin_a = std::sin(normalized_box.rotation());
  for (int k = 0; k < 4; ++k) {
    // Scale and rotate w.r.t. center.
    const Vector2_f rad = corners[k] - center;
    const Vector2_f rot_rad(cos_a * rad.x() - sin_a * rad.y(),
                            sin_a * rad.x() + cos_a * rad.y());
    corners[k] = center + rot_rad;
  }
  return corners;
}

}  // namespace

void TrackedDetection::AddLabel(const std::string& label, float score) {
  auto label_ptr = label_to_score_map_.find(label);
  if (label_ptr == label_to_score_map_.end()) {
    label_to_score_map_[label] = score;
  } else {
    label_ptr->second = label_ptr->second > score ? label_ptr->second : score;
  }
}

bool TrackedDetection::IsSameAs(const TrackedDetection& other,
                                float max_area_ratio,
                                float min_overlap_ratio) const {
  const auto box0 = bounding_box_;
  const auto box1 = other.bounding_box_;
  const double box0_area = BoxArea(box0);
  const double box1_area = BoxArea(box1);
  const double overlap_area = OverlapArea(box0, box1);

  // For cases where a small object is in front of a big object.
  // TODO: This is hard threshold. Make the threshold smaller
  // (e.g. 2.0) will cause issues when two detections of the same object is
  // vertial to each other. For example, if we first get a detection (e.g. a
  // long water bottle) vertically and then change the camera to horizontal
  // quickly, then it will get another detection which will have a diamond shape
  // that is much larger than the previous rectangle one.
  if (box0_area / box1_area > max_area_ratio ||
      box1_area / box0_area > max_area_ratio)
    return false;

  if (overlap_area / box0_area > min_overlap_ratio ||
      overlap_area / box1_area > min_overlap_ratio) {
    return true;
  }

  return false;
}

void TrackedDetection::MergeLabelScore(const TrackedDetection& other) {
  for (const auto& label_score : other.label_to_score_map()) {
    const auto label_score_ptr = label_to_score_map_.find(label_score.first);
    if (label_score_ptr == label_to_score_map_.end()) {
      AddLabel(label_score.first, label_score.second);
    } else {
      // TODO: Consider other strategy of merging scores, e.g. mean.
      label_score_ptr->second =
          std::max(label_score_ptr->second, label_score.second);
    }
  }
}

std::array<Vector2_f, 4> TrackedDetection::GetCorners(
    float image_width, float image_height) const {
  NormalizedRectBounds bounds;
  GetNormalizedRectBounds(bounding_box_, &bounds);
  Vector2_f center((bounds.right + bounds.left) / 2.0f * image_width,
                   (bounds.bottom + bounds.top) / 2.0f * image_height);

  return ComputeCorners(bounding_box_, center, image_width, image_height);
}

}  // namespace mediapipe
