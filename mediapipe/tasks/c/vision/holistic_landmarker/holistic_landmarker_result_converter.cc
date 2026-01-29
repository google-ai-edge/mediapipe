/* Copyright 2026 The MediaPipe Authors.

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

#include "mediapipe/tasks/c/vision/holistic_landmarker/holistic_landmarker_result_converter.h"

#include <cstdint>
#include <vector>

#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/components/containers/category_converter.h"
#include "mediapipe/tasks/c/components/containers/landmark_converter.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "mediapipe/tasks/c/vision/holistic_landmarker/holistic_landmarker_result.h"
#include "mediapipe/tasks/cc/components/containers/category.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/holistic_landmarker_result.h"

namespace mediapipe::tasks::c::vision::holistic_landmarker {

using CppCategory = ::mediapipe::tasks::components::containers::Category;
using CppLandmark = ::mediapipe::tasks::components::containers::Landmark;
using CppNormalizedLandmark =
    ::mediapipe::tasks::components::containers::NormalizedLandmark;
using mediapipe::tasks::c::components::containers::CppCloseNormalizedLandmarks;
using mediapipe::tasks::c::components::containers::CppConvertToCategory;
using mediapipe::tasks::c::components::containers::CppConvertToLandmarks;
using mediapipe::tasks::c::components::containers::
    CppConvertToNormalizedLandmarks;

void CppConvertToHolisticLandmarkerResult(
    const ::mediapipe::tasks::vision::holistic_landmarker::
        HolisticLandmarkerResult& in,
    HolisticLandmarkerResult* out) {
  // Convert face_landmarks
  CppConvertToNormalizedLandmarks(in.face_landmarks.landmarks,
                                  &out->face_landmarks);

  // Convert face_blendshapes
  if (in.face_blendshapes.has_value()) {
    const auto& blendshapes = in.face_blendshapes.value();
    out->face_blendshapes.categories_count = blendshapes.size();
    out->face_blendshapes.categories = new Category[blendshapes.size()];
    for (uint32_t i = 0; i < blendshapes.size(); ++i) {
      CppConvertToCategory(blendshapes[i],
                           &out->face_blendshapes.categories[i]);
    }
  } else {
    out->face_blendshapes.categories_count = 0;
    out->face_blendshapes.categories = nullptr;
  }

  // Convert pose_landmarks
  CppConvertToNormalizedLandmarks(in.pose_landmarks.landmarks,
                                  &out->pose_landmarks);

  // Convert pose_world_landmarks
  CppConvertToLandmarks(in.pose_world_landmarks.landmarks,
                        &out->pose_world_landmarks);

  // Convert hand landmarks
  CppConvertToNormalizedLandmarks(in.left_hand_landmarks.landmarks,
                                  &out->left_hand_landmarks);
  CppConvertToNormalizedLandmarks(in.right_hand_landmarks.landmarks,
                                  &out->right_hand_landmarks);
  CppConvertToLandmarks(in.left_hand_world_landmarks.landmarks,
                        &out->left_hand_world_landmarks);
  CppConvertToLandmarks(in.right_hand_world_landmarks.landmarks,
                        &out->right_hand_world_landmarks);

  // Convert pose_segmentation_masks
  if (in.pose_segmentation_masks.has_value()) {
    out->pose_segmentation_mask =
        new MpImageInternal{.image = in.pose_segmentation_masks.value()};
  } else {
    out->pose_segmentation_mask = nullptr;
  }
}

void CppCloseHolisticLandmarkerResult(HolisticLandmarkerResult* result) {
  CppCloseNormalizedLandmarks(&result->face_landmarks);
  if (result->face_blendshapes.categories) {
    for (uint32_t i = 0; i < result->face_blendshapes.categories_count; ++i) {
      components::containers::CppCloseCategory(
          &result->face_blendshapes.categories[i]);
    }
    delete[] result->face_blendshapes.categories;
    result->face_blendshapes.categories = nullptr;
    result->face_blendshapes.categories_count = 0;
  }

  CppCloseNormalizedLandmarks(&result->pose_landmarks);
  components::containers::CppCloseLandmarks(&result->pose_world_landmarks);
  CppCloseNormalizedLandmarks(&result->left_hand_landmarks);
  CppCloseNormalizedLandmarks(&result->right_hand_landmarks);
  components::containers::CppCloseLandmarks(&result->left_hand_world_landmarks);
  components::containers::CppCloseLandmarks(
      &result->right_hand_world_landmarks);

  if (result->pose_segmentation_mask) {
    delete result->pose_segmentation_mask;
    result->pose_segmentation_mask = nullptr;
  }
}

}  // namespace mediapipe::tasks::c::vision::holistic_landmarker
