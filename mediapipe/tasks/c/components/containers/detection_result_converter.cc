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

#include "mediapipe/tasks/c/components/containers/detection_result_converter.h"

#include <cstdlib>

#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/components/containers/category_converter.h"
#include "mediapipe/tasks/c/components/containers/detection_result.h"
#include "mediapipe/tasks/c/components/containers/keypoint.h"
#include "mediapipe/tasks/c/components/containers/keypoint_converter.h"
#include "mediapipe/tasks/c/components/containers/rect_converter.h"
#include "mediapipe/tasks/cc/components/containers/detection_result.h"

namespace mediapipe::tasks::c::components::containers {

void CppConvertToDetection(
    const mediapipe::tasks::components::containers::Detection& in,
    ::Detection* out) {
  out->categories_count = in.categories.size();
  out->categories = new Category[out->categories_count];
  for (size_t i = 0; i < out->categories_count; ++i) {
    CppConvertToCategory(in.categories[i], &out->categories[i]);
  }

  CppConvertToRect(in.bounding_box, &out->bounding_box);

  if (in.keypoints.has_value()) {
    auto& keypoints = in.keypoints.value();
    out->keypoints_count = keypoints.size();
    out->keypoints = new NormalizedKeypoint[out->keypoints_count];
    for (size_t i = 0; i < out->keypoints_count; ++i) {
      CppConvertToNormalizedKeypoint(keypoints[i], &out->keypoints[i]);
    }
  } else {
    out->keypoints = nullptr;
    out->keypoints_count = 0;
  }
}

void CppConvertToDetectionResult(
    const mediapipe::tasks::components::containers::DetectionResult& in,
    ::DetectionResult* out) {
  out->detections_count = in.detections.size();
  out->detections = new ::Detection[out->detections_count];
  for (size_t i = 0; i < out->detections_count; ++i) {
    CppConvertToDetection(in.detections[i], &out->detections[i]);
  }
}

// Functions to free the memory of C structures.
void CppCloseDetection(::Detection* in) {
  for (size_t i = 0; i < in->categories_count; ++i) {
    CppCloseCategory(&in->categories[i]);
  }
  delete[] in->categories;
  in->categories = nullptr;
  for (size_t i = 0; i < in->keypoints_count; ++i) {
    CppCloseNormalizedKeypoint(&in->keypoints[i]);
  }
  delete[] in->keypoints;
  in->keypoints = nullptr;
}

void CppCloseDetectionResult(::DetectionResult* in) {
  for (size_t i = 0; i < in->detections_count; ++i) {
    CppCloseDetection(&in->detections[i]);
  }
  delete[] in->detections;
  in->detections = nullptr;
}

}  // namespace mediapipe::tasks::c::components::containers
