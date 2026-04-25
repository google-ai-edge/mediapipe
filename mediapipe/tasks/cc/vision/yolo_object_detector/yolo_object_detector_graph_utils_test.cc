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

// Tests for yolo_object_detector_graph_utils.h

#include "mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph_utils.h"

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::tasks::vision::yolo_object_detector {
namespace {

// Tests InferDecodeMode() which is pure logic with no file I/O.

TEST(YoloObjectDetectorGraphUtilsTest, InferDecodeMode_EndToEnd_BoxesFirst) {
  // [300, 6] → one dim is 6 → END_TO_END
  EXPECT_EQ(InferDecodeMode({300, 6}), YoloDecodeMode::kEndToEnd);
}

TEST(YoloObjectDetectorGraphUtilsTest, InferDecodeMode_EndToEnd_FeaturesFirst) {
  // [6, 300] → one dim is 6 → END_TO_END
  EXPECT_EQ(InferDecodeMode({6, 300}), YoloDecodeMode::kEndToEnd);
}

TEST(YoloObjectDetectorGraphUtilsTest, InferDecodeMode_Ultralytics_FeaturesFirst) {
  // [84, 8400] — typical COCO YOLO output → ULTRALYTICS
  EXPECT_EQ(InferDecodeMode({84, 8400}), YoloDecodeMode::kUltralytics);
}

TEST(YoloObjectDetectorGraphUtilsTest, InferDecodeMode_Ultralytics_BoxesFirst) {
  // [8400, 84] transposed → ULTRALYTICS
  EXPECT_EQ(InferDecodeMode({8400, 84}), YoloDecodeMode::kUltralytics);
}

TEST(YoloObjectDetectorGraphUtilsTest, InferDecodeMode_WithLeadingSingleton) {
  // [1, 300, 6] → squeeze → [300, 6] → END_TO_END
  EXPECT_EQ(InferDecodeMode({1, 300, 6}), YoloDecodeMode::kEndToEnd);
}

TEST(YoloObjectDetectorGraphUtilsTest, InferDecodeMode_WithMultipleLeadingSingletons) {
  // [1, 1, 84, 8400] → squeeze → [84, 8400] → ULTRALYTICS
  EXPECT_EQ(InferDecodeMode({1, 1, 84, 8400}), YoloDecodeMode::kUltralytics);
}

}  // namespace
}  // namespace mediapipe::tasks::vision::yolo_object_detector
