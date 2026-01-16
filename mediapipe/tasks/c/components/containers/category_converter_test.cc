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

#include "mediapipe/tasks/c/components/containers/category_converter.h"

#include <cstdlib>
#include <optional>
#include <string>

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/cc/components/containers/category.h"

namespace mediapipe::tasks::c::components::containers {

TEST(CategoryConverterTest, ConvertsCategoryCustomValues) {
  mediapipe::tasks::components::containers::Category cpp_category = {
      /* index= */ 1,
      /* score= */ 0.1,
      /* category_name= */ "category_name",
      /* display_name= */ "display_name",
  };

  Category c_category;
  CppConvertToCategory(cpp_category, &c_category);
  EXPECT_EQ(c_category.index, 1);
  EXPECT_FLOAT_EQ(c_category.score, 0.1);
  EXPECT_EQ(std::string{c_category.category_name}, "category_name");
  EXPECT_EQ(std::string{c_category.display_name}, "display_name");

  CppCloseCategory(&c_category);
}

TEST(CategoryConverterTest, ConvertsCategoryDefaultValues) {
  mediapipe::tasks::components::containers::Category cpp_category = {
      /* index= */ 1,
      /* score= */ 0.1,
      /* category_name= */ std::nullopt,
      /* display_name= */ std::nullopt,
  };

  Category c_category;
  CppConvertToCategory(cpp_category, &c_category);
  EXPECT_EQ(c_category.index, 1);
  EXPECT_FLOAT_EQ(c_category.score, 0.1);
  EXPECT_EQ(c_category.category_name, nullptr);
  EXPECT_EQ(c_category.display_name, nullptr);

  CppCloseCategory(&c_category);
}

}  // namespace mediapipe::tasks::c::components::containers
