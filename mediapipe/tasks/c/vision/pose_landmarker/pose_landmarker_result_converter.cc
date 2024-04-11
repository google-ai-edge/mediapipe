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

#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker_result_converter.h"

#include <cstdint>
#include <vector>

#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/c/components/containers/landmark_converter.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker_result.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker_result.h"

namespace mediapipe::tasks::c::components::containers {

using CppLandmark = ::mediapipe::tasks::components::containers::Landmark;
using CppNormalizedLandmark =
    ::mediapipe::tasks::components::containers::NormalizedLandmark;

void CppConvertToPoseLandmarkerResult(
    const mediapipe::tasks::vision::pose_landmarker::PoseLandmarkerResult& in,
    PoseLandmarkerResult* out) {
  if (in.segmentation_masks.has_value()) {
    out->segmentation_masks_count = in.segmentation_masks.value().size();
    out->segmentation_masks = new MpMask[out->segmentation_masks_count];
    for (uint32_t i = 0; i < out->segmentation_masks_count; ++i) {
      const auto& image_frame =
          in.segmentation_masks.value()[i].GetImageFrameSharedPtr();

      MpMask mp_mask = {
          .type = MpMask::IMAGE_FRAME,
          .image_frame = {.mask_format = MaskFormat::FLOAT,
                          .image_buffer = image_frame->PixelData(),
                          .width = image_frame->Width(),
                          .height = image_frame->Height()}};
      out->segmentation_masks[i] = mp_mask;
    }
  } else {
    out->segmentation_masks_count = 0;
    out->segmentation_masks = nullptr;
  }

  out->pose_landmarks_count = in.pose_landmarks.size();
  out->pose_landmarks = new NormalizedLandmarks[out->pose_landmarks_count];
  for (uint32_t i = 0; i < out->pose_landmarks_count; ++i) {
    std::vector<CppNormalizedLandmark> cpp_normalized_landmarks;
    for (uint32_t j = 0; j < in.pose_landmarks[i].landmarks.size(); ++j) {
      const auto& cpp_landmark = in.pose_landmarks[i].landmarks[j];
      cpp_normalized_landmarks.push_back(cpp_landmark);
    }
    CppConvertToNormalizedLandmarks(cpp_normalized_landmarks,
                                    &out->pose_landmarks[i]);
  }

  out->pose_world_landmarks_count = in.pose_world_landmarks.size();
  out->pose_world_landmarks = new Landmarks[out->pose_world_landmarks_count];
  for (uint32_t i = 0; i < out->pose_world_landmarks_count; ++i) {
    std::vector<CppLandmark> cpp_landmarks;
    for (uint32_t j = 0; j < in.pose_world_landmarks[i].landmarks.size(); ++j) {
      const auto& cpp_landmark = in.pose_world_landmarks[i].landmarks[j];
      cpp_landmarks.push_back(cpp_landmark);
    }
    CppConvertToLandmarks(cpp_landmarks, &out->pose_world_landmarks[i]);
  }
}

void CppClosePoseLandmarkerResult(PoseLandmarkerResult* result) {
  if (result->segmentation_masks) {
    delete[] result->segmentation_masks;
    result->segmentation_masks = nullptr;
    result->segmentation_masks_count = 0;
  }

  for (uint32_t i = 0; i < result->pose_landmarks_count; ++i) {
    CppCloseNormalizedLandmarks(&result->pose_landmarks[i]);
  }
  delete[] result->pose_landmarks;

  for (uint32_t i = 0; i < result->pose_world_landmarks_count; ++i) {
    CppCloseLandmarks(&result->pose_world_landmarks[i]);
  }
  delete[] result->pose_world_landmarks;

  result->pose_landmarks = nullptr;
  result->pose_world_landmarks = nullptr;

  result->pose_landmarks_count = 0;
  result->pose_world_landmarks_count = 0;
}

}  // namespace mediapipe::tasks::c::components::containers
