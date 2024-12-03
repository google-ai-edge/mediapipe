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

#include "mediapipe/tasks/c/core/base_options_converter.h"

#include <cstring>
#include <string>

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/cc/core/base_options.h"

namespace mediapipe::tasks::c::core {

constexpr char kAssetBuffer[] = "abc";
constexpr char kModelAssetPath[] = "abc.tflite";

TEST(BaseOptionsConverterTest, ConvertsBaseOptionsAssetBuffer) {
  BaseOptions c_base_options = {/* model_asset_buffer= */ kAssetBuffer,
                                /* model_asset_buffer_count= */
                                static_cast<unsigned int>(strlen(kAssetBuffer)),
                                /* model_asset_path= */ nullptr};

  mediapipe::tasks::core::BaseOptions cpp_base_options = {};

  CppConvertToBaseOptions(c_base_options, &cpp_base_options);
  EXPECT_EQ(*cpp_base_options.model_asset_buffer, std::string{kAssetBuffer});
  EXPECT_EQ(cpp_base_options.model_asset_path, "");
}

TEST(BaseOptionsConverterTest, ConvertsBaseOptionsAssetPath) {
  BaseOptions c_base_options = {/* model_asset_buffer= */ nullptr,
                                /* model_asset_buffer_count= */ 0,
                                /* model_asset_path= */ kModelAssetPath};

  mediapipe::tasks::core::BaseOptions cpp_base_options = {};

  CppConvertToBaseOptions(c_base_options, &cpp_base_options);
  EXPECT_EQ(cpp_base_options.model_asset_buffer.get(), nullptr);
  EXPECT_EQ(cpp_base_options.model_asset_path, std::string{kModelAssetPath});
}

}  // namespace mediapipe::tasks::c::core
