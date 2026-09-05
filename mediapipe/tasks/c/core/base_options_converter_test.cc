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
#include <variant>

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
      /* file_descriptor= */ -1,
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
                                  /* file_descriptor= */ -1,
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
                                  /* file_descriptor= */ -1,
                                  /* delegate= */ MP_DELEGATE_GPU};

  mediapipe::tasks::core::BaseOptions cpp_base_options = {};

  CppConvertToBaseOptions(c_base_options, &cpp_base_options);
  EXPECT_EQ(cpp_base_options.delegate,
            mediapipe::tasks::core::BaseOptions::GPU);
  EXPECT_FALSE(cpp_base_options.disable_default_service);
}

TEST(BaseOptionsConverterTest, ConvertsLiteRtOptionsDefault) {
  MpLiteRtOptions c_litert_options = {};

  auto cpp_litert_options = CppConvertToLiteRtOptions(c_litert_options);
  EXPECT_EQ(cpp_litert_options.hardware_accelerator,
            mediapipe::tasks::core::BaseOptions::LiteRtOptions::
                HardwareAccelerator::CPU);
  EXPECT_TRUE(std::holds_alternative<
              mediapipe::tasks::core::BaseOptions::LiteRtOptions::CpuOptions>(
      cpp_litert_options.accelerator_options));
}

TEST(BaseOptionsConverterTest, ConvertsLiteRtOptionsCpu) {
  MpLiteRtOptions c_litert_options = {.hardware_accelerator =
                                          MP_LITERT_HARDWARE_ACCELERATOR_CPU};

  auto cpp_litert_options = CppConvertToLiteRtOptions(c_litert_options);
  EXPECT_EQ(cpp_litert_options.hardware_accelerator,
            mediapipe::tasks::core::BaseOptions::LiteRtOptions::
                HardwareAccelerator::CPU);
  EXPECT_TRUE(std::holds_alternative<
              mediapipe::tasks::core::BaseOptions::LiteRtOptions::CpuOptions>(
      cpp_litert_options.accelerator_options));
}

TEST(BaseOptionsConverterTest, ConvertsLiteRtOptionsGpu) {
  MpLiteRtOptions c_litert_options = {.hardware_accelerator =
                                          MP_LITERT_HARDWARE_ACCELERATOR_GPU};

  auto cpp_litert_options = CppConvertToLiteRtOptions(c_litert_options);
  EXPECT_EQ(cpp_litert_options.hardware_accelerator,
            mediapipe::tasks::core::BaseOptions::LiteRtOptions::
                HardwareAccelerator::GPU);
  EXPECT_TRUE(std::holds_alternative<
              mediapipe::tasks::core::BaseOptions::LiteRtOptions::GpuOptions>(
      cpp_litert_options.accelerator_options));
}

TEST(BaseOptionsConverterTest, ConvertsLiteRtOptionsNpu) {
  MpLiteRtOptions c_litert_options = {
      .hardware_accelerator = MP_LITERT_HARDWARE_ACCELERATOR_NPU,
      .accelerator_options = {
          .npu_options = {.dispatch_library_directory = "/tmp/dispatch"}}};

  auto cpp_litert_options = CppConvertToLiteRtOptions(c_litert_options);
  EXPECT_EQ(cpp_litert_options.hardware_accelerator,
            mediapipe::tasks::core::BaseOptions::LiteRtOptions::
                HardwareAccelerator::NPU);
  ASSERT_TRUE(std::holds_alternative<
              mediapipe::tasks::core::BaseOptions::LiteRtOptions::NpuOptions>(
      cpp_litert_options.accelerator_options));
  const auto& npu_opts =
      std::get<mediapipe::tasks::core::BaseOptions::LiteRtOptions::NpuOptions>(
          cpp_litert_options.accelerator_options);
  EXPECT_EQ(npu_opts.dispatch_library_directory, "/tmp/dispatch");
}

TEST(BaseOptionsConverterTest, ConvertsLiteRtCpuOptionsDirectly) {
  MpLiteRtCpuOptions c_opts = {};
  auto cpp_opts = CppConvertToLiteRtCpuOptions(c_opts);
  (void)cpp_opts;
}

TEST(BaseOptionsConverterTest, ConvertsLiteRtGpuOptionsDirectly) {
  MpLiteRtGpuOptions c_opts = {};
  auto cpp_opts = CppConvertToLiteRtGpuOptions(c_opts);
  (void)cpp_opts;
}

TEST(BaseOptionsConverterTest, ConvertsLiteRtNpuOptionsDirectly) {
  MpLiteRtNpuOptions c_opts = {.dispatch_library_directory = "/tmp/dispatch"};
  auto cpp_opts = CppConvertToLiteRtNpuOptions(c_opts);
  EXPECT_EQ(cpp_opts.dispatch_library_directory, "/tmp/dispatch");
}

TEST(BaseOptionsConverterTest, ConvertsBaseOptionsLiteRtCpuDelegate) {
  MpLiteRtOptions litert_options = {.hardware_accelerator =
                                        MP_LITERT_HARDWARE_ACCELERATOR_CPU};
  MpBaseOptions c_base_options = {
      .model_asset_path = kModelAssetPath,
      .file_descriptor = -1,
      .delegate = MP_DELEGATE_LITERT,
      .litert_options = &litert_options,
  };

  mediapipe::tasks::core::BaseOptions cpp_base_options = {};

  CppConvertToBaseOptions(c_base_options, &cpp_base_options);
  EXPECT_EQ(cpp_base_options.delegate,
            mediapipe::tasks::core::BaseOptions::LITERT);
  ASSERT_TRUE(cpp_base_options.delegate_options.has_value());
  ASSERT_TRUE(std::holds_alternative<
              mediapipe::tasks::core::BaseOptions::LiteRtOptions>(
      *cpp_base_options.delegate_options));
  const auto& litert_opts =
      std::get<mediapipe::tasks::core::BaseOptions::LiteRtOptions>(
          *cpp_base_options.delegate_options);
  EXPECT_EQ(litert_opts.hardware_accelerator,
            mediapipe::tasks::core::BaseOptions::LiteRtOptions::
                HardwareAccelerator::CPU);
  EXPECT_TRUE(cpp_base_options.disable_default_service);
}

TEST(BaseOptionsConverterTest, ConvertsBaseOptionsLiteRtGpuDelegate) {
  MpLiteRtOptions litert_options = {.hardware_accelerator =
                                        MP_LITERT_HARDWARE_ACCELERATOR_GPU};
  MpBaseOptions c_base_options = {
      .model_asset_path = kModelAssetPath,
      .file_descriptor = -1,
      .delegate = MP_DELEGATE_LITERT,
      .litert_options = &litert_options,
  };

  mediapipe::tasks::core::BaseOptions cpp_base_options = {};

  CppConvertToBaseOptions(c_base_options, &cpp_base_options);
  EXPECT_EQ(cpp_base_options.delegate,
            mediapipe::tasks::core::BaseOptions::LITERT);
  ASSERT_TRUE(cpp_base_options.delegate_options.has_value());
  ASSERT_TRUE(std::holds_alternative<
              mediapipe::tasks::core::BaseOptions::LiteRtOptions>(
      *cpp_base_options.delegate_options));
  const auto& litert_opts =
      std::get<mediapipe::tasks::core::BaseOptions::LiteRtOptions>(
          *cpp_base_options.delegate_options);
  EXPECT_EQ(litert_opts.hardware_accelerator,
            mediapipe::tasks::core::BaseOptions::LiteRtOptions::
                HardwareAccelerator::GPU);
  EXPECT_FALSE(cpp_base_options.disable_default_service);
}

TEST(BaseOptionsConverterTest, ConvertsBaseOptionsLiteRtNpuDelegate) {
  MpLiteRtOptions litert_options = {
      .hardware_accelerator = MP_LITERT_HARDWARE_ACCELERATOR_NPU,
      .accelerator_options = {
          .npu_options = {.dispatch_library_directory = "/tmp/dispatch"}}};
  MpBaseOptions c_base_options = {
      .model_asset_path = kModelAssetPath,
      .file_descriptor = -1,
      .delegate = MP_DELEGATE_LITERT,
      .litert_options = &litert_options,
  };

  mediapipe::tasks::core::BaseOptions cpp_base_options = {};

  CppConvertToBaseOptions(c_base_options, &cpp_base_options);
  EXPECT_EQ(cpp_base_options.delegate,
            mediapipe::tasks::core::BaseOptions::LITERT);
  ASSERT_TRUE(cpp_base_options.delegate_options.has_value());
  ASSERT_TRUE(std::holds_alternative<
              mediapipe::tasks::core::BaseOptions::LiteRtOptions>(
      *cpp_base_options.delegate_options));
  const auto& litert_opts =
      std::get<mediapipe::tasks::core::BaseOptions::LiteRtOptions>(
          *cpp_base_options.delegate_options);
  EXPECT_EQ(litert_opts.hardware_accelerator,
            mediapipe::tasks::core::BaseOptions::LiteRtOptions::
                HardwareAccelerator::NPU);
  ASSERT_TRUE(std::holds_alternative<
              mediapipe::tasks::core::BaseOptions::LiteRtOptions::NpuOptions>(
      litert_opts.accelerator_options));
  const auto& npu_opts =
      std::get<mediapipe::tasks::core::BaseOptions::LiteRtOptions::NpuOptions>(
          litert_opts.accelerator_options);
  EXPECT_EQ(npu_opts.dispatch_library_directory, "/tmp/dispatch");
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

TEST(BaseOptionsConverterTest, ConvertsBaseOptionsFileDescriptor) {
  MpBaseOptions c_base_options = {};
  c_base_options.file_descriptor = 123;

  mediapipe::tasks::core::BaseOptions cpp_base_options = {};

  CppConvertToBaseOptions(c_base_options, &cpp_base_options);
  EXPECT_EQ(cpp_base_options.model_asset_descriptor_meta.fd, 123);
}

TEST(BaseOptionsConverterTest, ConvertsBaseOptionsFileDescriptorZero) {
  MpBaseOptions c_base_options = {};
  c_base_options.file_descriptor = 0;

  mediapipe::tasks::core::BaseOptions cpp_base_options = {};

  CppConvertToBaseOptions(c_base_options, &cpp_base_options);
  EXPECT_EQ(cpp_base_options.model_asset_descriptor_meta.fd, -1);
}

}  // namespace mediapipe::tasks::c::core
