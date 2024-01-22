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

#include <stdbool.h>

#include <cstdint>
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
constexpr char kModelName[] = "mobilenet_v3_small_100_224_embedder.tflite";
constexpr char kImageFile[] = "burger.jpg";
constexpr float kPrecision = 1e-6;
constexpr int kIterations = 100;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

// Utility function to check the sizes, head_index and head_names of a result
// produced by kMobileNetV3Embedder.
void CheckMobileNetV3Result(const ImageEmbedderResult& result, bool quantized) {
  EXPECT_EQ(result.embeddings_count, 1);
  EXPECT_EQ(result.embeddings[0].head_index, 0);
  EXPECT_EQ(std::string{result.embeddings[0].head_name}, "feature");
  if (quantized) {
    EXPECT_EQ(result.embeddings[0].values_count, 1024);
  } else {
    EXPECT_EQ(result.embeddings[0].values_count, 1024);
  }
}

TEST(ImageEmbedderTest, ImageModeTest) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);
  ImageEmbedderOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* embedder_options= */
      {/* l2_normalize= */ true,
       /* quantize= */ false}};

  void* embedder = image_embedder_create(&options,
                                         /* error_msg */ nullptr);
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
  image_embedder_embed_image(embedder, &mp_image, &result,
                             /* error_msg */ nullptr);
  CheckMobileNetV3Result(result, false);
  EXPECT_NEAR(result.embeddings[0].float_embedding[0], -0.0142344, kPrecision);
  image_embedder_close_result(&result);
  image_embedder_close(embedder, /* error_msg */ nullptr);
}

TEST(ImageEmbedderTest, SucceedsWithCosineSimilarity) {
  const auto image = DecodeImageFromFile(GetFullPath("burger.jpg"));
  ASSERT_TRUE(image.ok());
  const auto crop = DecodeImageFromFile(GetFullPath("burger_crop.jpg"));
  ASSERT_TRUE(crop.ok());

  const std::string model_path = GetFullPath(kModelName);
  ImageEmbedderOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* embedder_options= */
      {/* l2_normalize= */ true,
       /* quantize= */ false}};

  void* embedder = image_embedder_create(&options,
                                         /* error_msg */ nullptr);
  EXPECT_NE(embedder, nullptr);

  const MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {
          .format = static_cast<ImageFormat>(
              image->GetImageFrameSharedPtr()->Format()),
          .image_buffer = image->GetImageFrameSharedPtr()->PixelData(),
          .width = image->GetImageFrameSharedPtr()->Width(),
          .height = image->GetImageFrameSharedPtr()->Height()}};

  const MpImage mp_crop = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {
          .format = static_cast<ImageFormat>(
              crop->GetImageFrameSharedPtr()->Format()),
          .image_buffer = crop->GetImageFrameSharedPtr()->PixelData(),
          .width = crop->GetImageFrameSharedPtr()->Width(),
          .height = crop->GetImageFrameSharedPtr()->Height()}};

  // Extract both embeddings.
  ImageEmbedderResult image_result;
  image_embedder_embed_image(embedder, &mp_image, &image_result,
                             /* error_msg */ nullptr);
  ImageEmbedderResult crop_result;
  image_embedder_embed_image(embedder, &mp_crop, &crop_result,
                             /* error_msg */ nullptr);

  // Check results.
  CheckMobileNetV3Result(image_result, false);
  CheckMobileNetV3Result(crop_result, false);
  // Check cosine similarity.
  double similarity;
  image_embedder_cosine_similarity(image_result.embeddings[0],
                                   crop_result.embeddings[0], &similarity,
                                   /* error_msg */ nullptr);
  double expected_similarity = 0.925519;
  EXPECT_LE(abs(similarity - expected_similarity), kPrecision);
  image_embedder_close_result(&image_result);
  image_embedder_close_result(&crop_result);
  image_embedder_close(embedder, /* error_msg */ nullptr);
}

TEST(ImageEmbedderTest, VideoModeTest) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);
  ImageEmbedderOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::VIDEO,
      /* embedder_options= */
      {/* l2_normalize= */ true,
       /* quantize= */ false}};

  void* embedder = image_embedder_create(&options,
                                         /* error_msg */ nullptr);
  EXPECT_NE(embedder, nullptr);

  const auto& image_frame = image->GetImageFrameSharedPtr();
  const MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {.format = static_cast<ImageFormat>(image_frame->Format()),
                      .image_buffer = image_frame->PixelData(),
                      .width = image_frame->Width(),
                      .height = image_frame->Height()}};

  for (int i = 0; i < kIterations; ++i) {
    ImageEmbedderResult result;
    image_embedder_embed_for_video(embedder, &mp_image, i, &result,
                                   /* error_msg */ nullptr);
    CheckMobileNetV3Result(result, false);
    EXPECT_NEAR(result.embeddings[0].float_embedding[0], -0.0142344,
                kPrecision);
    image_embedder_close_result(&result);
  }
  image_embedder_close(embedder, /* error_msg */ nullptr);
}

// A structure to support LiveStreamModeTest below. This structure holds a
// static method `Fn` for a callback function of C API. A `static` qualifier
// allows to take an address of the method to follow API style. Another static
// struct member is `last_timestamp` that is used to verify that current
// timestamp is greater than the previous one.
struct LiveStreamModeCallback {
  static int64_t last_timestamp;
  static void Fn(const ImageEmbedderResult* embedder_result,
                 const MpImage& image, int64_t timestamp, char* error_msg) {
    ASSERT_NE(embedder_result, nullptr);
    ASSERT_EQ(error_msg, nullptr);
    CheckMobileNetV3Result(*embedder_result, false);
    EXPECT_NEAR(embedder_result->embeddings[0].float_embedding[0], -0.0142344,
                kPrecision);
    EXPECT_GT(image.image_frame.width, 0);
    EXPECT_GT(image.image_frame.height, 0);
    EXPECT_GT(timestamp, last_timestamp);
    last_timestamp++;
  }
};
int64_t LiveStreamModeCallback::last_timestamp = -1;

TEST(ImageEmbedderTest, LiveStreamModeTest) {
  const auto image = DecodeImageFromFile(GetFullPath(kImageFile));
  ASSERT_TRUE(image.ok());

  const std::string model_path = GetFullPath(kModelName);

  ImageEmbedderOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::LIVE_STREAM,
      /* embedder_options= */
      {/* l2_normalize= */ true,
       /* quantize= */ false},
      /* result_callback= */ LiveStreamModeCallback::Fn,
  };

  void* embedder = image_embedder_create(&options,
                                         /* error_msg */ nullptr);
  EXPECT_NE(embedder, nullptr);

  const auto& image_frame = image->GetImageFrameSharedPtr();
  const MpImage mp_image = {
      .type = MpImage::IMAGE_FRAME,
      .image_frame = {.format = static_cast<ImageFormat>(image_frame->Format()),
                      .image_buffer = image_frame->PixelData(),
                      .width = image_frame->Width(),
                      .height = image_frame->Height()}};

  for (int i = 0; i < kIterations; ++i) {
    EXPECT_GE(image_embedder_embed_async(embedder, &mp_image, i,
                                         /* error_msg */ nullptr),
              0);
  }
  image_embedder_close(embedder, /* error_msg */ nullptr);

  // Due to the flow limiter, the total of outputs might be smaller than the
  // number of iterations.
  EXPECT_LE(LiveStreamModeCallback::last_timestamp, kIterations);
  EXPECT_GT(LiveStreamModeCallback::last_timestamp, 0);
}

TEST(ImageEmbedderTest, InvalidArgumentHandling) {
  // It is an error to set neither the asset buffer nor the path.
  ImageEmbedderOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
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
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* running_mode= */ RunningMode::IMAGE,
      /* embedder_options= */
      {/* l2_normalize= */ false,
       /* quantize= */ false},
  };

  void* embedder = image_embedder_create(&options,
                                         /* error_msg */ nullptr);
  EXPECT_NE(embedder, nullptr);

  const MpImage mp_image = {.type = MpImage::GPU_BUFFER, .gpu_buffer = {}};
  ImageEmbedderResult result;
  char* error_msg;
  image_embedder_embed_image(embedder, &mp_image, &result, &error_msg);
  EXPECT_THAT(error_msg, HasSubstr("GPU Buffer not supported yet."));
  free(error_msg);
  image_embedder_close(embedder, /* error_msg */ nullptr);
}

}  // namespace
