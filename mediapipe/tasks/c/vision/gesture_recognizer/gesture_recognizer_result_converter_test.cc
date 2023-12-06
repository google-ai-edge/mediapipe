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
#include <string>

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/c/vision/gesture_recognizer/gesture_recognizer_result.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/gesture_recognizer_result.h"

namespace mediapipe::tasks::c::components::containers {

using mediapipe::ClassificationList;
using mediapipe::LandmarkList;
using mediapipe::NormalizedLandmarkList;

void InitGestureRecognizerResult(
    ::mediapipe::tasks::vision::gesture_recognizer::GestureRecognizerResult*
        cpp_result) {
  // Initialize gestures
  mediapipe::Classification classification_for_gestures;
  classification_for_gestures.set_index(0);
  classification_for_gestures.set_score(0.9f);
  classification_for_gestures.set_label("gesture_label_1");
  classification_for_gestures.set_display_name("gesture_display_name_1");
  ClassificationList gestures_list;
  *gestures_list.add_classification() = classification_for_gestures;
  cpp_result->gestures.push_back(gestures_list);

  // Initialize handedness
  mediapipe::Classification classification_for_handedness;
  classification_for_handedness.set_index(1);
  classification_for_handedness.set_score(0.8f);
  classification_for_handedness.set_label("handeness_label_1");
  classification_for_handedness.set_display_name("handeness_display_name_1");
  ClassificationList handedness_list;
  *handedness_list.add_classification() = classification_for_handedness;
  cpp_result->handedness.push_back(handedness_list);

  // Initialize hand_landmarks
  NormalizedLandmarkList normalized_landmark_list;
  auto& normalized_landmark = *normalized_landmark_list.add_landmark();
  normalized_landmark.set_x(0.1f);
  normalized_landmark.set_y(0.2f);
  normalized_landmark.set_z(0.3f);

  cpp_result->hand_landmarks.push_back(normalized_landmark_list);

  // Initialize hand_world_landmarks
  LandmarkList landmark_list;
  auto& landmark = *landmark_list.add_landmark();
  landmark.set_x(1.0f);
  landmark.set_y(1.1f);
  landmark.set_z(1.2f);

  cpp_result->hand_world_landmarks.push_back(landmark_list);
}

TEST(GestureRecognizerResultConverterTest, ConvertsCustomResult) {
  ::mediapipe::tasks::vision::gesture_recognizer::GestureRecognizerResult
      cpp_result;
  InitGestureRecognizerResult(&cpp_result);

  GestureRecognizerResult c_result;
  CppConvertToGestureRecognizerResult(cpp_result, &c_result);

  // Verify conversion of gestures
  EXPECT_NE(c_result.gestures, nullptr);
  EXPECT_EQ(c_result.gestures_count, cpp_result.gestures.size());

  for (uint32_t i = 0; i < c_result.gestures_count; ++i) {
    EXPECT_EQ(c_result.gestures[i].categories_count,
              cpp_result.gestures[i].classification_size());
    for (uint32_t j = 0; j < c_result.gestures[i].categories_count; ++j) {
      auto gesture = cpp_result.gestures[i].classification(j);
      EXPECT_EQ(std::string(c_result.gestures[i].categories[j].category_name),
                gesture.label());
      EXPECT_FLOAT_EQ(c_result.gestures[i].categories[j].score,
                      gesture.score());
    }
  }

  // Verify conversion of handedness
  EXPECT_NE(c_result.handedness, nullptr);
  EXPECT_EQ(c_result.handedness_count, cpp_result.handedness.size());

  for (uint32_t i = 0; i < c_result.handedness_count; ++i) {
    EXPECT_EQ(c_result.handedness[i].categories_count,
              cpp_result.handedness[i].classification_size());
    for (uint32_t j = 0; j < c_result.handedness[i].categories_count; ++j) {
      auto handedness = cpp_result.handedness[i].classification(j);
      EXPECT_EQ(std::string(c_result.handedness[i].categories[j].category_name),
                handedness.label());
      EXPECT_FLOAT_EQ(c_result.handedness[i].categories[j].score,
                      handedness.score());
    }
  }

  // Verify conversion of hand_landmarks
  EXPECT_NE(c_result.hand_landmarks, nullptr);
  EXPECT_EQ(c_result.hand_landmarks_count, cpp_result.hand_landmarks.size());

  for (uint32_t i = 0; i < c_result.hand_landmarks_count; ++i) {
    EXPECT_EQ(c_result.hand_landmarks[i].landmarks_count,
              cpp_result.hand_landmarks[i].landmark_size());
    for (uint32_t j = 0; j < c_result.hand_landmarks[i].landmarks_count; ++j) {
      const auto& landmark = cpp_result.hand_landmarks[i].landmark(j);
      EXPECT_FLOAT_EQ(c_result.hand_landmarks[i].landmarks[j].x, landmark.x());
      EXPECT_FLOAT_EQ(c_result.hand_landmarks[i].landmarks[j].y, landmark.y());
      EXPECT_FLOAT_EQ(c_result.hand_landmarks[i].landmarks[j].z, landmark.z());
    }
  }

  // Verify conversion of hand_world_landmarks
  EXPECT_NE(c_result.hand_world_landmarks, nullptr);
  EXPECT_EQ(c_result.hand_world_landmarks_count,
            cpp_result.hand_world_landmarks.size());
  for (uint32_t i = 0; i < c_result.hand_world_landmarks_count; ++i) {
    EXPECT_EQ(c_result.hand_world_landmarks[i].landmarks_count,
              cpp_result.hand_world_landmarks[i].landmark_size());
    for (uint32_t j = 0; j < c_result.hand_world_landmarks[i].landmarks_count;
         ++j) {
      const auto& landmark = cpp_result.hand_world_landmarks[i].landmark(j);
      EXPECT_FLOAT_EQ(c_result.hand_world_landmarks[i].landmarks[j].x,
                      landmark.x());
      EXPECT_FLOAT_EQ(c_result.hand_world_landmarks[i].landmarks[j].y,
                      landmark.y());
      EXPECT_FLOAT_EQ(c_result.hand_world_landmarks[i].landmarks[j].z,
                      landmark.z());
    }
  }

  CppCloseGestureRecognizerResult(&c_result);
}

TEST(GestureRecognizerResultConverterTest, FreesMemory) {
  ::mediapipe::tasks::vision::gesture_recognizer::GestureRecognizerResult
      cpp_result;
  InitGestureRecognizerResult(&cpp_result);

  GestureRecognizerResult c_result;
  CppConvertToGestureRecognizerResult(cpp_result, &c_result);

  EXPECT_NE(c_result.gestures, nullptr);
  EXPECT_NE(c_result.handedness, nullptr);
  EXPECT_NE(c_result.hand_landmarks, nullptr);
  EXPECT_NE(c_result.hand_world_landmarks, nullptr);

  CppCloseGestureRecognizerResult(&c_result);

  EXPECT_EQ(c_result.gestures, nullptr);
  EXPECT_EQ(c_result.handedness, nullptr);
  EXPECT_EQ(c_result.hand_landmarks, nullptr);
  EXPECT_EQ(c_result.hand_world_landmarks, nullptr);
}

}  // namespace mediapipe::tasks::c::components::containers
