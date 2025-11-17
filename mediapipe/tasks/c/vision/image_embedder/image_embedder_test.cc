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
#include "absl/synchronization/blocking_counter.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/embedding_result.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/core/image_test_util.h"

namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::vision::core::CreateEmptyGpuMpImage;
using ::mediapipe::tasks::vision::core::GetImage;
using ::mediapipe::tasks::vision::core::ScopedMpImage;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kModelName[] = "mobilenet_v3_small_100_224_embedder.tflite";
constexpr char kImageFile[] = "burger.jpg";
constexpr float kPrecision = 1e-6;
constexpr int kIterations = 5;
constexpr int kSleepBetweenFramesMilliseconds = 100;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

// Utility function to check the sizes, head_index and head_names of a result
// produced by kMobileNetV3Embedder.
void CheckMobileNetV3Result(const ImageEmbedderResult& result) {
  EXPECT_EQ(result.embeddings_count, 1);
  EXPECT_EQ(result.embeddings[0].head_index, 0);
  EXPECT_EQ(std::string{result.embeddings[0].head_name}, "feature");
  EXPECT_EQ(result.embeddings[0].values_count, 1024);
}

TEST(ImageEmbedderTest, ImageModeTest) {
  const ScopedMpImage image = GetImage(GetFullPath(kImageFile));

  const std::string model_path = GetFullPath(kModelName);
  ImageEmbedderOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::IMAGE,
      .embedder_options = {.l2_normalize = true, .quantize = false}};

  MpImageEmbedderPtr embedder;
  ASSERT_EQ(MpImageEmbedderCreate(&options, &embedder), kMpOk);
  ASSERT_NE(embedder, nullptr);

  ImageEmbedderResult result;
  ASSERT_EQ(MpImageEmbedderEmbedImage(embedder, image.get(),
                                      /* image_processing_options= */ nullptr,
                                      &result),
            kMpOk);
  CheckMobileNetV3Result(result);
  EXPECT_NEAR(result.embeddings[0].float_embedding[0], -0.0142344, kPrecision);
  MpImageEmbedderCloseResult(&result);
  ASSERT_EQ(MpImageEmbedderClose(embedder), kMpOk);
}

TEST(ImageEmbedderTest, ImageModeTestWithQuantization) {
  const ScopedMpImage image = GetImage(GetFullPath(kImageFile));

  const std::string model_path = GetFullPath(kModelName);
  ImageEmbedderOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::IMAGE,
      .embedder_options = {.l2_normalize = false, .quantize = true}};

  MpImageEmbedderPtr embedder;
  ASSERT_EQ(MpImageEmbedderCreate(&options, &embedder), kMpOk);
  ASSERT_NE(embedder, nullptr);

  ImageEmbedderResult result;
  ASSERT_EQ(MpImageEmbedderEmbedImage(embedder, image.get(),
                                      /* image_processing_options= */ nullptr,
                                      &result),
            kMpOk);
  CheckMobileNetV3Result(result);
  EXPECT_EQ(result.embeddings[0].quantized_embedding[0], '\xE5');
  MpImageEmbedderCloseResult(&result);
  ASSERT_EQ(MpImageEmbedderClose(embedder), kMpOk);
}

TEST(ImageEmbedderTest, ImageModeTestWithRotation) {
  const ScopedMpImage image = GetImage(GetFullPath("burger_rotated.jpg"));
  ASSERT_NE(image, nullptr);

  const std::string model_path = GetFullPath(kModelName);
  ImageEmbedderOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::IMAGE,
      .embedder_options = {.l2_normalize = true, .quantize = false}};

  MpImageEmbedderPtr embedder;
  ASSERT_EQ(MpImageEmbedderCreate(&options, &embedder), kMpOk);
  ASSERT_NE(embedder, nullptr);

  ImageProcessingOptions image_processing_options;
  image_processing_options.has_region_of_interest = 0;
  image_processing_options.rotation_degrees = -90;

  ImageEmbedderResult result;
  ASSERT_EQ(MpImageEmbedderEmbedImage(embedder, image.get(),
                                      &image_processing_options, &result),
            kMpOk);
  CheckMobileNetV3Result(result);
  EXPECT_NEAR(result.embeddings[0].float_embedding[0], -0.0149445, kPrecision);
  MpImageEmbedderCloseResult(&result);
  ASSERT_EQ(MpImageEmbedderClose(embedder), kMpOk);
}

TEST(ImageEmbedderTest, SucceedsWithCosineSimilarity) {
  const ScopedMpImage image = GetImage(GetFullPath("burger.jpg"));
  const ScopedMpImage crop = GetImage(GetFullPath("burger_crop.jpg"));

  const std::string model_path = GetFullPath(kModelName);
  ImageEmbedderOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::IMAGE,
      .embedder_options = {.l2_normalize = true, .quantize = false}};

  MpImageEmbedderPtr embedder;
  ASSERT_EQ(MpImageEmbedderCreate(&options, &embedder), kMpOk);
  EXPECT_NE(embedder, nullptr);

  // Extract both embeddings.
  ImageEmbedderResult image_result;
  ASSERT_EQ(MpImageEmbedderEmbedImage(embedder, image.get(),
                                      /* image_processing_options= */ nullptr,
                                      &image_result),
            kMpOk);
  ImageEmbedderResult crop_result;
  ASSERT_EQ(MpImageEmbedderEmbedImage(embedder, crop.get(),
                                      /* image_processing_options= */ nullptr,
                                      &crop_result),
            kMpOk);

  // Check results.
  CheckMobileNetV3Result(image_result);
  CheckMobileNetV3Result(crop_result);
  // Check cosine similarity.
  double similarity;
  ASSERT_EQ(
      MpImageEmbedderCosineSimilarity(image_result.embeddings[0],
                                      crop_result.embeddings[0], &similarity),
      kMpOk);
  double expected_similarity = 0.925519;
  EXPECT_LE(abs(similarity - expected_similarity), kPrecision);
  MpImageEmbedderCloseResult(&image_result);
  MpImageEmbedderCloseResult(&crop_result);
  ASSERT_EQ(MpImageEmbedderClose(embedder), kMpOk);
}

TEST(ImageEmbedderTest, VideoModeTest) {
  const ScopedMpImage image = GetImage(GetFullPath(kImageFile));

  const std::string model_path = GetFullPath(kModelName);
  ImageEmbedderOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::VIDEO,
      .embedder_options = {.l2_normalize = true, .quantize = false}};

  MpImageEmbedderPtr embedder;
  ASSERT_EQ(MpImageEmbedderCreate(&options, &embedder), kMpOk);
  ASSERT_NE(embedder, nullptr);

  for (int i = 0; i < kIterations; ++i) {
    ImageEmbedderResult result;
    ASSERT_EQ(MpImageEmbedderEmbedForVideo(
                  embedder, image.get(),
                  /* image_processing_options= */ nullptr, i, &result),
              kMpOk);
    CheckMobileNetV3Result(result);
    EXPECT_NEAR(result.embeddings[0].float_embedding[0], -0.0142344,
                kPrecision);
    MpImageEmbedderCloseResult(&result);
  }
  ASSERT_EQ(MpImageEmbedderClose(embedder), kMpOk);
}

// A structure to support LiveStreamModeTest below. This structure holds a
// static method `Fn` for a callback function of C API. A `static` qualifier
// allows to take an address of the method to follow API style. Another static
// struct member is `last_timestamp` that is used to verify that current
// timestamp is greater than the previous one.
struct LiveStreamModeCallback {
  static int64_t last_timestamp;
  static absl::BlockingCounter* blocking_counter;
  static void Fn(MpStatus status, const EmbeddingResult* embedder_result,
                 MpImagePtr image, int64_t timestamp) {
    ASSERT_EQ(status, kMpOk);
    ASSERT_NE(embedder_result, nullptr);
    CheckMobileNetV3Result(*embedder_result);
    EXPECT_NEAR(embedder_result->embeddings[0].float_embedding[0], -0.0142344,
                kPrecision);
    EXPECT_GT(MpImageGetWidth(image), 0);
    EXPECT_GT(MpImageGetHeight(image), 0);
    EXPECT_GT(timestamp, last_timestamp);
    last_timestamp++;

    if (blocking_counter) {
      blocking_counter->DecrementCount();
    }
  }
};
int64_t LiveStreamModeCallback::last_timestamp = -1;
absl::BlockingCounter* LiveStreamModeCallback::blocking_counter = nullptr;

TEST(ImageEmbedderTest, LiveStreamModeTest) {
  const ScopedMpImage image = GetImage(GetFullPath(kImageFile));

  const std::string model_path = GetFullPath(kModelName);

  ImageEmbedderOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::LIVE_STREAM,
      .embedder_options = {.l2_normalize = true, .quantize = false},
      .result_callback = LiveStreamModeCallback::Fn,
  };

  MpImageEmbedderPtr embedder;
  ASSERT_EQ(MpImageEmbedderCreate(&options, &embedder), kMpOk);
  ASSERT_NE(embedder, nullptr);

  absl::BlockingCounter counter(kIterations);
  LiveStreamModeCallback::blocking_counter = &counter;

  for (int i = 0; i < kIterations; ++i) {
    ASSERT_EQ(
        MpImageEmbedderEmbedAsync(embedder, image.get(),
                                  /* image_processing_options= */ nullptr, i),
        kMpOk);
    // Short sleep so that MediaPipe does not drop frames.
    absl::SleepFor(absl::Milliseconds(kSleepBetweenFramesMilliseconds));
  }

  // Wait for all callbacks to be invoked.
  counter.Wait();
  LiveStreamModeCallback::blocking_counter = nullptr;

  ASSERT_EQ(MpImageEmbedderClose(embedder), kMpOk);

  // Due to the flow limiter, the total of outputs might be smaller than the
  // number of iterations.
  EXPECT_LE(LiveStreamModeCallback::last_timestamp, kIterations);
  EXPECT_GT(LiveStreamModeCallback::last_timestamp, 0);
}

TEST(ImageEmbedderTest, InvalidArgumentHandling) {
  // It is an error to set neither the asset buffer nor the path.
  ImageEmbedderOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = nullptr},
      .embedder_options = {},
  };

  MpImageEmbedderPtr embedder = nullptr;
  EXPECT_EQ(MpImageEmbedderCreate(&options, &embedder), kMpInvalidArgument);
  EXPECT_EQ(embedder, nullptr);
}

TEST(ImageEmbedderTest, FailedEmbeddingHandling) {
  const std::string model_path = GetFullPath(kModelName);
  ImageEmbedderOptions options = {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path.c_str()},
      .running_mode = RunningMode::IMAGE,
      .embedder_options = {.l2_normalize = false, .quantize = false}};

  MpImageEmbedderPtr embedder;
  ASSERT_EQ(MpImageEmbedderCreate(&options, &embedder), kMpOk);
  EXPECT_NE(embedder, nullptr);

  const ScopedMpImage image = CreateEmptyGpuMpImage();
  ImageEmbedderResult result;
  EXPECT_EQ(MpImageEmbedderEmbedImage(embedder, image.get(),
                                      /* image_processing_options= */ nullptr,
                                      &result),
            kMpInvalidArgument);
  ASSERT_EQ(MpImageEmbedderClose(embedder), kMpOk);
}

}  // namespace
