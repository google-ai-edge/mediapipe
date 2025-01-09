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

#ifndef MEDIAPIPE_TASKS_C_VISION_FACE_LANDMARKER_RESULT_FACE_LANDMARKER_RESULT_H_
#define MEDIAPIPE_TASKS_C_VISION_FACE_LANDMARKER_RESULT_FACE_LANDMARKER_RESULT_H_

#include <cstdint>

#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/c/components/containers/matrix.h"

#ifndef MP_EXPORT
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

// The hand landmarker result from HandLandmarker, where each vector
// element represents a single hand detected in the image.
struct FaceLandmarkerResult {
  // Detected face landmarks in normalized image coordinates.
  struct NormalizedLandmarks* face_landmarks;

  // The number of elements in the face_landmarks array.
  uint32_t face_landmarks_count;

  // Optional face blendshapes results.
  struct Categories* face_blendshapes;

  // The number of elements in the face_blendshapes array.
  uint32_t face_blendshapes_count;

  // Optional facial transformation matrixes.
  struct Matrix* facial_transformation_matrixes;

  // The number of elements in the facial_transformation_matrixes array.
  uint32_t facial_transformation_matrixes_count;
};

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_VISION_FACE_LANDMARKER_RESULT_FACE_LANDMARKER_RESULT_H_
