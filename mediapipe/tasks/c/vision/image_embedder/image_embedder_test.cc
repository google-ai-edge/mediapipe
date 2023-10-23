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

#include "mediapipe/tasks/c/vision/image_embedder/image_embedder.h"

#include <cstdlib>
#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::vision::DecodeImageFromFile;
using testing::HasSubstr;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kModelName[] = "mobilenet_v3_small_100_224_embedder.tflite";
constexpr char kImageFile[] = "burger.jpg";
constexpr float kPrecision = 1e-6;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

TEST(ImageEmbedderTest, SmokeTest) {
  const auto image = DecodeImageFromFile(GetFullPath("burger.jpg"));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);
  ImageEmbedderOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* embedder_options= */
      {/* l2_normalize= */ true,
       /* quantize= */ false},
  };

  void* embedder = image_embedder_create(&options);
  EXPECT_NE(embedder, nullptr);

  const MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {
          .format = static_cast<ImageFormat>(
              image->GetImageFrameSharedPtr()->Format()),
          .image_buffer = image->GetImageFrameSharedPtr()->PixelData(),
          .width = image->GetImageFrameSharedPtr()->Width(),
          .height = image->GetImageFrameSharedPtr()->Height()}};

  ImageEmbedderResult result;
  image_embedder_embed_image(embedder, &mp_image, &result);
  EXPECT_EQ(result.embeddings_count, 1);
  EXPECT_NEAR(result.embeddings[0].float_embedding[0], -0.0142344, kPrecision);
  image_embedder_close_result(&result);
  image_embedder_close(embedder);
}

TEST(ImageEmbedderTest, InvalidArgumentHandling) {
  // It is an error to set neither the asset buffer nor the path.
  ImageEmbedderOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_path= */ nullptr},
      /* embedder_options= */ {},
  };

  char* error_msg;
  void* embedder = image_embedder_create(&options, &error_msg);
  EXPECT_EQ(embedder, nullptr);

  EXPECT_THAT(error_msg, HasSubstr("ExternalFile must specify"));

  free(error_msg);
}

TEST(ImageEmbedderTest, FailedEmbeddingHandling) {
  const std::string model_path = GetFullPath(kModelName);
  ImageEmbedderOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* embedder_options= */
      {/* l2_normalize= */ false,
       /* quantize= */ false},
  };

  void* embedder = image_embedder_create(&options);
  EXPECT_NE(embedder, nullptr);

  const MpImage mp_image = {.type = MpImage::GPU_BUFFER, .gpu_buffer = {}};
  ImageEmbedderResult result;
  char* error_msg;
  image_embedder_embed_image(embedder, &mp_image, &result, &error_msg);
  EXPECT_THAT(error_msg, HasSubstr("gpu buffer not supported yet"));
  free(error_msg);
  image_embedder_close(embedder);
}

}  // namespace
