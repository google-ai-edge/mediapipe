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

#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker_result_converter.h"

#include <cstdint>
#include <vector>

#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/components/containers/category_converter.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/c/components/containers/landmark_converter.h"
#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker_result.h"
#include "mediapipe/tasks/cc/components/containers/category.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_result.h"

namespace mediapipe::tasks::c::components::containers {

using CppCategory = ::mediapipe::tasks::components::containers::Category;
using CppLandmark = ::mediapipe::tasks::components::containers::Landmark;
using CppNormalizedLandmark =
    ::mediapipe::tasks::components::containers::NormalizedLandmark;

void CppConvertToHandLandmarkerResult(
    const mediapipe::tasks::vision::hand_landmarker::HandLandmarkerResult& in,
    HandLandmarkerResult* out) {
  out->handedness_count = in.handedness.size();
  out->handedness = new Categories[out->handedness_count];

  for (uint32_t i = 0; i < out->handedness_count; ++i) {
    uint32_t categories_count = in.handedness[i].categories.size();
    out->handedness[i].categories_count = categories_count;
    out->handedness[i].categories = new Category[categories_count];

    for (uint32_t j = 0; j < categories_count; ++j) {
      const auto& cpp_category = in.handedness[i].categories[j];
      CppConvertToCategory(cpp_category, &out->handedness[i].categories[j]);
    }
  }

  out->hand_landmarks_count = in.hand_landmarks.size();
  out->hand_landmarks = new NormalizedLandmarks[out->hand_landmarks_count];
  for (uint32_t i = 0; i < out->hand_landmarks_count; ++i) {
    std::vector<CppNormalizedLandmark> cpp_normalized_landmarks;
    for (uint32_t j = 0; j < in.hand_landmarks[i].landmarks.size(); ++j) {
      const auto& cpp_landmark = in.hand_landmarks[i].landmarks[j];
      cpp_normalized_landmarks.push_back(cpp_landmark);
    }
    CppConvertToNormalizedLandmarks(cpp_normalized_landmarks,
                                    &out->hand_landmarks[i]);
  }

  out->hand_world_landmarks_count = in.hand_world_landmarks.size();
  out->hand_world_landmarks = new Landmarks[out->hand_world_landmarks_count];
  for (uint32_t i = 0; i < out->hand_world_landmarks_count; ++i) {
    std::vector<CppLandmark> cpp_landmarks;
    for (uint32_t j = 0; j < in.hand_world_landmarks[i].landmarks.size(); ++j) {
      const auto& cpp_landmark = in.hand_world_landmarks[i].landmarks[j];
      cpp_landmarks.push_back(cpp_landmark);
    }
    CppConvertToLandmarks(cpp_landmarks, &out->hand_world_landmarks[i]);
  }
}

void CppCloseHandLandmarkerResult(HandLandmarkerResult* result) {
  for (uint32_t i = 0; i < result->handedness_count; ++i) {
    CppCloseCategories(&result->handedness[i]);
  }
  delete[] result->handedness;

  for (uint32_t i = 0; i < result->hand_landmarks_count; ++i) {
    CppCloseNormalizedLandmarks(&result->hand_landmarks[i]);
  }
  delete[] result->hand_landmarks;

  for (uint32_t i = 0; i < result->hand_world_landmarks_count; ++i) {
    CppCloseLandmarks(&result->hand_world_landmarks[i]);
  }
  delete[] result->hand_world_landmarks;

  result->handedness = nullptr;
  result->hand_landmarks = nullptr;
  result->hand_world_landmarks = nullptr;

  result->handedness_count = 0;
  result->hand_landmarks_count = 0;
  result->hand_world_landmarks_count = 0;
}

}  // namespace mediapipe::tasks::c::components::containers
