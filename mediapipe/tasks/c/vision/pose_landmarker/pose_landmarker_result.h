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

#ifndef MEDIAPIPE_TASKS_C_VISION_POSE_LANDMARKER_RESULT_POSE_LANDMARKER_RESULT_H_
#define MEDIAPIPE_TASKS_C_VISION_POSE_LANDMARKER_RESULT_POSE_LANDMARKER_RESULT_H_

#include <cstdint>

#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/c/vision/core/common.h"

#ifndef MP_EXPORT
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

// The hand landmarker result from PoseLandmarker, where each vector
// element represents a single hand detected in the image.
struct PoseLandmarkerResult {
  // Segmentation masks for pose.
  struct MpMask* segmentation_masks;

  // The number of elements in the segmentation_masks array.
  uint32_t segmentation_masks_count;

  // Detected hand landmarks in normalized image coordinates.
  struct NormalizedLandmarks* pose_landmarks;

  // The number of elements in the pose_landmarks array.
  uint32_t pose_landmarks_count;

  // Detected hand landmarks in world coordinates.
  struct Landmarks* pose_world_landmarks;

  // The number of elements in the pose_world_landmarks array.
  uint32_t pose_world_landmarks_count;
};

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_POSE_LANDMARKER_RESULT_POSE_LANDMARKER_RESULT_H_
