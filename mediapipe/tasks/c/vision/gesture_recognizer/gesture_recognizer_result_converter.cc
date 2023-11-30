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

#include "mediapipe/tasks/c/vision/gesture_recognizer/gesture_recognizer_result_converter.h"

#include <cstdint>
#include <vector>

#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/components/containers/category_converter.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/c/components/containers/landmark_converter.h"
#include "mediapipe/tasks/c/vision/gesture_recognizer/gesture_recognizer_result.h"
#include "mediapipe/tasks/cc/components/containers/category.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/gesture_recognizer_result.h"

namespace mediapipe::tasks::c::components::containers {

using CppCategory = ::mediapipe::tasks::components::containers::Category;
using CppLandmark = ::mediapipe::tasks::components::containers::Landmark;
using CppNormalizedLandmark =
    ::mediapipe::tasks::components::containers::NormalizedLandmark;

void CppConvertToGestureRecognizerResult(
    const mediapipe::tasks::vision::gesture_recognizer::GestureRecognizerResult&
        in,
    GestureRecognizerResult* out) {
  out->gestures_count = in.gestures.size();
  out->gestures = new Categories[out->gestures_count];

  for (uint32_t i = 0; i < out->gestures_count; ++i) {
    uint32_t categories_count = in.gestures[i].classification_size();
    out->gestures[i].categories_count = categories_count;
    out->gestures[i].categories = new Category[categories_count];

    for (uint32_t j = 0; j < categories_count; ++j) {
      const auto& classification = in.gestures[i].classification(j);

      CppCategory cpp_category;
      // Set fields from the Classification protobuf
      cpp_category.index = classification.index();
      cpp_category.score = classification.score();
      if (classification.has_label()) {
        cpp_category.category_name = classification.label();
      }
      if (classification.has_display_name()) {
        cpp_category.display_name = classification.display_name();
      }

      CppConvertToCategory(cpp_category, &out->gestures[i].categories[j]);
    }
  }

  out->handedness_count = in.handedness.size();
  out->handedness = new Categories[out->handedness_count];

  for (uint32_t i = 0; i < out->handedness_count; ++i) {
    uint32_t categories_count = in.handedness[i].classification_size();
    out->handedness[i].categories_count = categories_count;
    out->handedness[i].categories = new Category[categories_count];

    for (uint32_t j = 0; j < categories_count; ++j) {
      const auto& classification = in.handedness[i].classification(j);

      CppCategory cpp_category;
      // Set fields from the Classification protobuf
      cpp_category.index = classification.index();
      cpp_category.score = classification.score();
      if (classification.has_label()) {
        cpp_category.category_name = classification.label();
      }
      if (classification.has_display_name()) {
        cpp_category.display_name = classification.display_name();
      }

      CppConvertToCategory(cpp_category, &out->handedness[i].categories[j]);
    }
  }

  out->hand_landmarks_count = in.hand_landmarks.size();
  out->hand_landmarks = new NormalizedLandmarks[out->hand_landmarks_count];
  for (uint32_t i = 0; i < out->hand_landmarks_count; ++i) {
    std::vector<CppNormalizedLandmark> cpp_normalized_landmarks;
    for (uint32_t j = 0; j < in.hand_landmarks[i].landmark_size(); ++j) {
      const auto& landmark = in.hand_landmarks[i].landmark(j);
      CppNormalizedLandmark cpp_landmark;
      cpp_landmark.x = landmark.x();
      cpp_landmark.y = landmark.y();
      cpp_landmark.z = landmark.z();
      if (landmark.has_presence()) {
        cpp_landmark.presence = landmark.presence();
      }
      if (landmark.has_visibility()) {
        cpp_landmark.visibility = landmark.visibility();
      }
      cpp_normalized_landmarks.push_back(cpp_landmark);
    }
    CppConvertToNormalizedLandmarks(cpp_normalized_landmarks,
                                    &out->hand_landmarks[i]);
  }

  out->hand_world_landmarks_count = in.hand_world_landmarks.size();
  out->hand_world_landmarks = new Landmarks[out->hand_world_landmarks_count];
  for (uint32_t i = 0; i < out->hand_world_landmarks_count; ++i) {
    std::vector<CppLandmark> cpp_landmarks;
    for (uint32_t j = 0; j < in.hand_world_landmarks[i].landmark_size(); ++j) {
      const auto& landmark = in.hand_world_landmarks[i].landmark(j);
      CppLandmark cpp_landmark;
      cpp_landmark.x = landmark.x();
      cpp_landmark.y = landmark.y();
      cpp_landmark.z = landmark.z();
      if (landmark.has_presence()) {
        cpp_landmark.presence = landmark.presence();
      }
      if (landmark.has_visibility()) {
        cpp_landmark.visibility = landmark.visibility();
      }
      cpp_landmarks.push_back(cpp_landmark);
    }
    CppConvertToLandmarks(cpp_landmarks, &out->hand_world_landmarks[i]);
  }
}

void CppCloseGestureRecognizerResult(GestureRecognizerResult* result) {
  for (uint32_t i = 0; i < result->gestures_count; ++i) {
    CppCloseCategories(&result->gestures[i]);
  }
  delete[] result->gestures;

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

  result->gestures = nullptr;
  result->handedness = nullptr;
  result->hand_landmarks = nullptr;
  result->hand_world_landmarks = nullptr;

  result->gestures_count = 0;
  result->handedness_count = 0;
  result->hand_landmarks_count = 0;
  result->hand_world_landmarks_count = 0;
}

}  // namespace mediapipe::tasks::c::components::containers
