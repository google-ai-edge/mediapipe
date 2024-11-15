/* Copyright 2024 The MediaPipe Authors.

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

#include "mediapipe/tasks/c/vision/face_stylizer/face_stylizer.h"

#include <cstdlib>
#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::vision::DecodeImageFromFile;
using testing::HasSubstr;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kModelName[] = "face_stylizer_color_ink.task";
constexpr char kImageFile[] = "portrait.jpg";
constexpr int kModelImageSize = 256;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

TEST(FaceStylizerTest, ImageModeTest) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);
  FaceStylizerOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
  };

  void* stylizer = face_stylizer_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(stylizer, nullptr);

  const auto& image_frame = image->GetImageFrameSharedPtr();
  const MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {.format = static_cast<ImageFormat>(image_frame->Format()),
                      .image_buffer = image_frame->PixelData(),
                      .width = image_frame->Width(),
                      .height = image_frame->Height()}};

  MpImage result = {.type = MpImage::IMAGE_FRAME};
  face_stylizer_stylize_image(stylizer, &mp_image, &result,
                              /* error_msg */ nullptr);
  EXPECT_EQ(result.image_frame.width, kModelImageSize);
  EXPECT_EQ(result.image_frame.height, kModelImageSize);
  face_stylizer_close_result(&result);
  face_stylizer_close(stylizer, /* error_msg */ nullptr);
}

TEST(FaceStylizerTest, InvalidArgumentHandling) {
  // It is an error to set neither the asset buffer nor the path.
  FaceStylizerOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ nullptr},
  };

  char* error_msg;
  void* stylizer = face_stylizer_create(&options, &error_msg);
  EXPECT_EQ(stylizer, nullptr);

  EXPECT_THAT(error_msg, HasSubstr("ExternalFile must specify"));

  free(error_msg);
}

TEST(FaceStylizerTest, FailedStylizationHandling) {
  const std::string model_path = GetFullPath(kModelName);
  FaceStylizerOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
  };

  void* stylizer = face_stylizer_create(&options, /* error_msg */
                                        nullptr);
  EXPECT_NE(stylizer, nullptr);

  const MpImage mp_image = {.type = MpImage::GPU_BUFFER, .gpu_buffer = {}};
  MpImage result = {.type = MpImage::IMAGE_FRAME};
  char* error_msg;
  face_stylizer_stylize_image(stylizer, &mp_image, &result, &error_msg);
  EXPECT_THAT(error_msg, HasSubstr("GPU Buffer not supported yet"));
  free(error_msg);
  face_stylizer_close(stylizer, /* error_msg */ nullptr);
}

}  // namespace
