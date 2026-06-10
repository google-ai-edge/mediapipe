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
  MpBaseOptions c_base_options = {
      /* model_asset_buffer= */ kAssetBuffer,
      /* model_asset_buffer_count= */
      static_cast<unsigned int>(strlen(kAssetBuffer)),
      /* model_asset_path= */ nullptr,
      /* delegate= */ MP_DELEGATE_CPU};

  mediapipe::tasks::core::BaseOptions cpp_base_options = {};

  CppConvertToBaseOptions(c_base_options, &cpp_base_options);
  EXPECT_EQ(*cpp_base_options.model_asset_buffer, std::string{kAssetBuffer});
  EXPECT_EQ(cpp_base_options.model_asset_path, "");
}

TEST(BaseOptionsConverterTest, ConvertsBaseOptionsAssetPath) {
  MpBaseOptions c_base_options = {/* model_asset_buffer= */ nullptr,
                                  /* model_asset_buffer_count= */ 0,
                                  /* model_asset_path= */ kModelAssetPath,
                                  /* delegate= */ MP_DELEGATE_CPU};

  mediapipe::tasks::core::BaseOptions cpp_base_options = {};

  CppConvertToBaseOptions(c_base_options, &cpp_base_options);
  EXPECT_EQ(cpp_base_options.model_asset_buffer.get(), nullptr);
  EXPECT_EQ(cpp_base_options.model_asset_path, std::string{kModelAssetPath});
}

TEST(BaseOptionsConverterTest, ConvertsBaseOptionsDelegate) {
  MpBaseOptions c_base_options = {/* model_asset_buffer= */ nullptr,
                                  /* model_asset_buffer_count= */ 0,
                                  /* model_asset_path= */ kModelAssetPath,
                                  /* delegate= */ MP_DELEGATE_GPU};

  mediapipe::tasks::core::BaseOptions cpp_base_options = {};

  CppConvertToBaseOptions(c_base_options, &cpp_base_options);
  EXPECT_EQ(cpp_base_options.delegate,
            mediapipe::tasks::core::BaseOptions::GPU);
}

TEST(BaseOptionsConverterTest, ConvertsBaseOptionsAppId) {
  MpBaseOptions c_base_options = {};
  c_base_options.app_id = "test_app_id";

  mediapipe::tasks::core::BaseOptions cpp_base_options = {};

  CppConvertToBaseOptions(c_base_options, &cpp_base_options);
  EXPECT_EQ(cpp_base_options.app_id, "test_app_id");
}

TEST(BaseOptionsConverterTest, ConvertsBaseOptionsAppVersion) {
  MpBaseOptions c_base_options = {};
  c_base_options.app_version = "test_app_version";

  mediapipe::tasks::core::BaseOptions cpp_base_options = {};

  CppConvertToBaseOptions(c_base_options, &cpp_base_options);
  EXPECT_EQ(cpp_base_options.app_version, "test_app_version");
}

TEST(BaseOptionsConverterTest, ToMpHostEnvironmentConvertsValues) {
  EXPECT_EQ(ToMpHostEnvironment(0), MP_HOST_ENVIRONMENT_UNKNOWN);
  EXPECT_EQ(ToMpHostEnvironment(1), MP_HOST_ENVIRONMENT_ANDROID);
  EXPECT_EQ(ToMpHostEnvironment(2), MP_HOST_ENVIRONMENT_IOS);
  EXPECT_EQ(ToMpHostEnvironment(3), MP_HOST_ENVIRONMENT_PYTHON);
  EXPECT_EQ(ToMpHostEnvironment(4), MP_HOST_ENVIRONMENT_WEB);
}

TEST(BaseOptionsConverterTest, ToMpHostSystemConvertsValues) {
  EXPECT_EQ(ToMpHostSystem(0), MP_HOST_SYSTEM_UNKNOWN);
  EXPECT_EQ(ToMpHostSystem(1), MP_HOST_SYSTEM_LINUX);
  EXPECT_EQ(ToMpHostSystem(2), MP_HOST_SYSTEM_MAC);
  EXPECT_EQ(ToMpHostSystem(3), MP_HOST_SYSTEM_WINDOWS);
  EXPECT_EQ(ToMpHostSystem(4), MP_HOST_SYSTEM_IOS);
  EXPECT_EQ(ToMpHostSystem(5), MP_HOST_SYSTEM_ANDROID);
}

}  // namespace mediapipe::tasks::c::core
