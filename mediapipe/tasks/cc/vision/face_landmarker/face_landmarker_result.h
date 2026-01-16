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

#ifndef MEDIAPIPE_TASKS_CC_VISION_FACE_LANDMARKER_FACE_LANDMARKER_RESULT_H_
#define MEDIAPIPE_TASKS_CC_VISION_FACE_LANDMARKER_FACE_LANDMARKER_RESULT_H_

#include <optional>
#include <vector>

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/matrix_data.pb.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace face_landmarker {

// The face landmarks detection result from FaceLandmarker, where each vector
// element represents a single face detected in the image.
struct FaceLandmarkerResult {
  // Detected face landmarks in normalized image coordinates.
  std::vector<components::containers::NormalizedLandmarks> face_landmarks;
  // Optional face blendshapes results.
  std::optional<std::vector<components::containers::Classifications>>
      face_blendshapes;
  // Optional facial transformation matrix.
  std::optional<std::vector<Matrix>> facial_transformation_matrixes;
};

// Convert face landmarks result from proto format to FaceLandmarkerResult.
FaceLandmarkerResult ConvertToFaceLandmarkerResult(
    std::vector<mediapipe::NormalizedLandmarkList> face_landmarks_proto,
    std::optional<std::vector<mediapipe::ClassificationList>>
        face_blendshapes_proto = std::nullopt,
    std::optional<std::vector<mediapipe::MatrixData>>
        facial_transformation_matrixes_proto = std::nullopt);

}  // namespace face_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_FACE_LANDMARKER_FACE_LANDMARKER_RESULT_H_
