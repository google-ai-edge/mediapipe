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

#ifndef MEDIAPIPE_TASKS_C_VISION_HOLISTIC_LANDMARKER_HOLISTIC_LANDMARKER_RESULT_H_
#define MEDIAPIPE_TASKS_C_VISION_HOLISTIC_LANDMARKER_HOLISTIC_LANDMARKER_RESULT_H_

#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/c/vision/core/image.h"

#ifdef __cplusplus
extern "C" {
#endif

// The holistic landmarks detection result from HolisticLandmarker.
struct HolisticLandmarkerResult {
  // Detected face landmarks in normalized image coordinates.
  struct NormalizedLandmarks face_landmarks;

  // Detected pose landmarks in normalized image coordinates.
  struct NormalizedLandmarks pose_landmarks;

  // Detected pose landmarks in world coordinates.
  struct Landmarks pose_world_landmarks;

  // Left hand landmarks in normalized image coordinates.
  struct NormalizedLandmarks left_hand_landmarks;

  // Right hand landmarks in normalized image coordinates.
  struct NormalizedLandmarks right_hand_landmarks;

  // Left hand landmarks in world coordinates.
  struct Landmarks left_hand_world_landmarks;

  // Right hand landmarks in world coordinates.
  struct Landmarks right_hand_world_landmarks;

  // Optional face blendshapes.
  struct Categories face_blendshapes;

  // Optional pose segmentation mask.
  MpImagePtr pose_segmentation_mask;
};

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_HOLISTIC_LANDMARKER_HOLISTIC_LANDMARKER_RESULT_H_
