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

#include "mediapipe/tasks/c/components/containers/rect_converter.h"

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/rect.h"
#include "mediapipe/tasks/cc/components/containers/rect.h"

namespace mediapipe::tasks::c::components::containers {

TEST(RectConverterTest, ConvertsRectCustomValues) {
  mediapipe::tasks::components::containers::Rect cpp_rect = {0, 1, 2, 3};

  MPRect c_rect;
  CppConvertToRect(cpp_rect, &c_rect);
  EXPECT_EQ(c_rect.left, 0);
  EXPECT_EQ(c_rect.top, 1);
  EXPECT_EQ(c_rect.right, 2);
  EXPECT_EQ(c_rect.bottom, 3);
}

TEST(RectFConverterTest, ConvertsRectFCustomValues) {
  mediapipe::tasks::components::containers::RectF cpp_rect = {0.1, 0.2, 0.3,
                                                              0.4};

  MPRectF c_rect;
  CppConvertToRectF(cpp_rect, &c_rect);
  EXPECT_FLOAT_EQ(c_rect.left, 0.1);
  EXPECT_FLOAT_EQ(c_rect.top, 0.2);
  EXPECT_FLOAT_EQ(c_rect.right, 0.3);
  EXPECT_FLOAT_EQ(c_rect.bottom, 0.4);
}

}  // namespace mediapipe::tasks::c::components::containers
