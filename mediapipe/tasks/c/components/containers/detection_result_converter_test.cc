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

#include "mediapipe/tasks/c/components/containers/detection_result_converter.h"

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/detection_result.h"
#include "mediapipe/tasks/cc/components/containers/detection_result.h"

namespace mediapipe::tasks::c::components::containers {

TEST(DetectionResultConverterTest, ConvertsDetectionResultCustomCategory) {
  mediapipe::tasks::components::containers::DetectionResult
      cpp_detection_result = {/* detections= */ {
          {/* categories= */ {{/* index= */ 1, /* score= */ 0.1,
                               /* category_name= */ "cat",
                               /* display_name= */ "cat"}},
           /* bounding_box= */ {10, 11, 12, 13},
           {/* keypoints */ {{0.1, 0.1, "foo", 0.5}}}}}};

  DetectionResult c_detection_result;
  CppConvertToDetectionResult(cpp_detection_result, &c_detection_result);
  EXPECT_NE(c_detection_result.detections, nullptr);
  EXPECT_EQ(c_detection_result.detections_count, 1);
  EXPECT_NE(c_detection_result.detections[0].categories, nullptr);
  EXPECT_EQ(c_detection_result.detections[0].categories_count, 1);
  EXPECT_EQ(c_detection_result.detections[0].bounding_box.left, 10);
  EXPECT_EQ(c_detection_result.detections[0].bounding_box.top, 11);
  EXPECT_EQ(c_detection_result.detections[0].bounding_box.right, 12);
  EXPECT_EQ(c_detection_result.detections[0].bounding_box.bottom, 13);
  EXPECT_NE(c_detection_result.detections[0].keypoints, nullptr);

  CppCloseDetectionResult(&c_detection_result);
}

TEST(DetectionResultConverterTest, ConvertsDetectionResultNoCategory) {
  mediapipe::tasks::components::containers::DetectionResult
      cpp_detection_result = {/* detections= */ {/* categories= */ {}}};

  DetectionResult c_detection_result;
  CppConvertToDetectionResult(cpp_detection_result, &c_detection_result);
  EXPECT_NE(c_detection_result.detections, nullptr);
  EXPECT_EQ(c_detection_result.detections_count, 1);
  EXPECT_NE(c_detection_result.detections[0].categories, nullptr);
  EXPECT_EQ(c_detection_result.detections[0].categories_count, 0);

  CppCloseDetectionResult(&c_detection_result);
}

TEST(DetectionResultConverterTest, FreesMemory) {
  mediapipe::tasks::components::containers::DetectionResult
      cpp_detection_result = {/* detections= */ {{/* categories= */ {}}}};

  DetectionResult c_detection_result;
  CppConvertToDetectionResult(cpp_detection_result, &c_detection_result);
  EXPECT_NE(c_detection_result.detections, nullptr);

  CppCloseDetectionResult(&c_detection_result);
  EXPECT_EQ(c_detection_result.detections, nullptr);
}

}  // namespace mediapipe::tasks::c::components::containers
