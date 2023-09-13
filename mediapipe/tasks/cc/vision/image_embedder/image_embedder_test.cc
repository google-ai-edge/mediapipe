/* Copyright 2022 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/vision/image_embedder/image_embedder.h"

#include <memory>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/test_util.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_embedder {
namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::components::containers::RectF;
using ::mediapipe::tasks::vision::core::ImageProcessingOptions;
using ::testing::HasSubstr;
using ::testing::Optional;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kMobileNetV3Embedder[] =
    "mobilenet_v3_small_100_224_embedder.tflite";
constexpr double kSimilarityTolerancy = 1e-6;

// Utility function to check the sizes, head_index and head_names of a result
// produced by kMobileNetV3Embedder.
void CheckMobileNetV3Result(const ImageEmbedderResult& result, bool quantized) {
  EXPECT_EQ(result.embeddings.size(), 1);
  EXPECT_EQ(result.embeddings[0].head_index, 0);
  EXPECT_EQ(result.embeddings[0].head_name, "feature");
  if (quantized) {
    EXPECT_EQ(result.embeddings[0].quantized_embedding.size(), 1024);
  } else {
    EXPECT_EQ(result.embeddings[0].float_embedding.size(), 1024);
  }
}

// A custom OpResolver only containing the Ops required by the test model.
class MobileNetV3OpResolver : public ::tflite::MutableOpResolver {
 public:
  MobileNetV3OpResolver() {
    AddBuiltin(::tflite::BuiltinOperator_MUL,
               ::tflite::ops::builtin::Register_MUL());
    AddBuiltin(::tflite::BuiltinOperator_SUB,
               ::tflite::ops::builtin::Register_SUB());
    AddBuiltin(::tflite::BuiltinOperator_CONV_2D,
               ::tflite::ops::builtin::Register_CONV_2D());
    AddBuiltin(::tflite::BuiltinOperator_HARD_SWISH,
               ::tflite::ops::builtin::Register_HARD_SWISH());
    AddBuiltin(::tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
               ::tflite::ops::builtin::Register_DEPTHWISE_CONV_2D());
    AddBuiltin(::tflite::BuiltinOperator_MEAN,
               ::tflite::ops::builtin::Register_MEAN());
    AddBuiltin(::tflite::BuiltinOperator_ADD,
               ::tflite::ops::builtin::Register_ADD());
    AddBuiltin(::tflite::BuiltinOperator_AVERAGE_POOL_2D,
               ::tflite::ops::builtin::Register_AVERAGE_POOL_2D());
    AddBuiltin(::tflite::BuiltinOperator_RESHAPE,
               ::tflite::ops::builtin::Register_RESHAPE());
  }

  MobileNetV3OpResolver(const MobileNetV3OpResolver& r) = delete;
};

// A custom OpResolver missing Ops required by the test model.
class MobileNetV3OpResolverMissingOps : public ::tflite::MutableOpResolver {
 public:
  MobileNetV3OpResolverMissingOps() {
    AddBuiltin(::tflite::BuiltinOperator_SOFTMAX,
               ::tflite::ops::builtin::Register_SOFTMAX());
  }

  MobileNetV3OpResolverMissingOps(const MobileNetV3OpResolverMissingOps& r) =
      delete;
};

class CreateTest : public tflite::testing::Test {};

TEST_F(CreateTest, SucceedsWithSelectiveOpResolver) {
  auto options = std::make_unique<ImageEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetV3Embedder);
  options->base_options.op_resolver = std::make_unique<MobileNetV3OpResolver>();

  MP_ASSERT_OK(ImageEmbedder::Create(std::move(options)));
}

TEST_F(CreateTest, FailsWithSelectiveOpResolverMissingOps) {
  auto options = std::make_unique<ImageEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetV3Embedder);
  options->base_options.op_resolver =
      std::make_unique<MobileNetV3OpResolverMissingOps>();

  auto image_embedder = ImageEmbedder::Create(std::move(options));

  EXPECT_EQ(image_embedder.status().code(), absl::StatusCode::kInternal);
  EXPECT_THAT(image_embedder.status().message(),
              HasSubstr("interpreter_builder(&interpreter) == kTfLiteOk"));
}

TEST_F(CreateTest, FailsWithMissingModel) {
  auto image_embedder =
      ImageEmbedder::Create(std::make_unique<ImageEmbedderOptions>());

  EXPECT_EQ(image_embedder.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      image_embedder.status().message(),
      HasSubstr("ExternalFile must specify at least one of 'file_content', "
                "'file_name', 'file_pointer_meta' or 'file_descriptor_meta'."));
  EXPECT_THAT(image_embedder.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(CreateTest, FailsWithIllegalCallbackInImageOrVideoMode) {
  for (auto running_mode :
       {core::RunningMode::IMAGE, core::RunningMode::VIDEO}) {
    auto options = std::make_unique<ImageEmbedderOptions>();
    options->base_options.model_asset_path =
        JoinPath("./", kTestDataDirectory, kMobileNetV3Embedder);
    options->running_mode = running_mode;
    options->result_callback = [](absl::StatusOr<ImageEmbedderResult>,
                                  const Image& image, int64_t timestamp_ms) {};

    auto image_embedder = ImageEmbedder::Create(std::move(options));

    EXPECT_EQ(image_embedder.status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_THAT(
        image_embedder.status().message(),
        HasSubstr("a user-defined result callback shouldn't be provided"));
    EXPECT_THAT(image_embedder.status().GetPayload(kMediaPipeTasksPayload),
                Optional(absl::Cord(absl::StrCat(
                    MediaPipeTasksStatus::kInvalidTaskGraphConfigError))));
  }
}

TEST_F(CreateTest, FailsWithMissingCallbackInLiveStreamMode) {
  auto options = std::make_unique<ImageEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetV3Embedder);
  options->running_mode = core::RunningMode::LIVE_STREAM;

  auto image_embedder = ImageEmbedder::Create(std::move(options));

  EXPECT_EQ(image_embedder.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(image_embedder.status().message(),
              HasSubstr("a user-defined result callback must be provided"));
  EXPECT_THAT(image_embedder.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kInvalidTaskGraphConfigError))));
}

class ImageModeTest : public tflite::testing::Test {};

TEST_F(ImageModeTest, FailsWithCallingWrongMethod) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetV3Embedder);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageEmbedder> image_embedder,
                          ImageEmbedder::Create(std::move(options)));

  auto results = image_embedder->EmbedForVideo(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the video mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));

  results = image_embedder->EmbedAsync(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the live stream mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));
  MP_ASSERT_OK(image_embedder->Close());
}

TEST_F(ImageModeTest, SucceedsWithoutL2Normalization) {
  auto options = std::make_unique<ImageEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetV3Embedder);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageEmbedder> image_embedder,
                          ImageEmbedder::Create(std::move(options)));
  // Load images: one is a crop of the other.
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  MP_ASSERT_OK_AND_ASSIGN(
      Image crop, DecodeImageFromFile(
                      JoinPath("./", kTestDataDirectory, "burger_crop.jpg")));

  // Extract both embeddings.
  MP_ASSERT_OK_AND_ASSIGN(const ImageEmbedderResult& image_result,
                          image_embedder->Embed(image));
  MP_ASSERT_OK_AND_ASSIGN(const ImageEmbedderResult& crop_result,
                          image_embedder->Embed(crop));

  // Check results.
  CheckMobileNetV3Result(image_result, false);
  CheckMobileNetV3Result(crop_result, false);
  // CheckCosineSimilarity.
  MP_ASSERT_OK_AND_ASSIGN(double similarity, ImageEmbedder::CosineSimilarity(
                                                 image_result.embeddings[0],
                                                 crop_result.embeddings[0]));
  double expected_similarity = 0.925519;
  EXPECT_LE(abs(similarity - expected_similarity), kSimilarityTolerancy);
}

TEST_F(ImageModeTest, SucceedsWithL2Normalization) {
  auto options = std::make_unique<ImageEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetV3Embedder);
  options->embedder_options.l2_normalize = true;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageEmbedder> image_embedder,
                          ImageEmbedder::Create(std::move(options)));
  // Load images: one is a crop of the other.
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  MP_ASSERT_OK_AND_ASSIGN(
      Image crop, DecodeImageFromFile(
                      JoinPath("./", kTestDataDirectory, "burger_crop.jpg")));

  // Extract both embeddings.
  MP_ASSERT_OK_AND_ASSIGN(const ImageEmbedderResult& image_result,
                          image_embedder->Embed(image));
  MP_ASSERT_OK_AND_ASSIGN(const ImageEmbedderResult& crop_result,
                          image_embedder->Embed(crop));

  // Check results.
  CheckMobileNetV3Result(image_result, false);
  CheckMobileNetV3Result(crop_result, false);
  // CheckCosineSimilarity.
  MP_ASSERT_OK_AND_ASSIGN(double similarity, ImageEmbedder::CosineSimilarity(
                                                 image_result.embeddings[0],
                                                 crop_result.embeddings[0]));
  double expected_similarity = 0.925519;
  EXPECT_LE(abs(similarity - expected_similarity), kSimilarityTolerancy);
}

TEST_F(ImageModeTest, SucceedsWithQuantization) {
  auto options = std::make_unique<ImageEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetV3Embedder);
  options->embedder_options.quantize = true;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageEmbedder> image_embedder,
                          ImageEmbedder::Create(std::move(options)));
  // Load images: one is a crop of the other.
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  MP_ASSERT_OK_AND_ASSIGN(
      Image crop, DecodeImageFromFile(
                      JoinPath("./", kTestDataDirectory, "burger_crop.jpg")));

  // Extract both embeddings.
  MP_ASSERT_OK_AND_ASSIGN(const ImageEmbedderResult& image_result,
                          image_embedder->Embed(image));
  MP_ASSERT_OK_AND_ASSIGN(const ImageEmbedderResult& crop_result,
                          image_embedder->Embed(crop));

  // Check results.
  CheckMobileNetV3Result(image_result, true);
  CheckMobileNetV3Result(crop_result, true);
  // CheckCosineSimilarity.
  MP_ASSERT_OK_AND_ASSIGN(double similarity, ImageEmbedder::CosineSimilarity(
                                                 image_result.embeddings[0],
                                                 crop_result.embeddings[0]));
  double expected_similarity = 0.926791;
  EXPECT_LE(abs(similarity - expected_similarity), kSimilarityTolerancy);
}

TEST_F(ImageModeTest, SucceedsWithRegionOfInterest) {
  auto options = std::make_unique<ImageEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetV3Embedder);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageEmbedder> image_embedder,
                          ImageEmbedder::Create(std::move(options)));
  // Load images: one is a crop of the other.
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  MP_ASSERT_OK_AND_ASSIGN(
      Image crop, DecodeImageFromFile(
                      JoinPath("./", kTestDataDirectory, "burger_crop.jpg")));
  // Region-of-interest in "burger.jpg" corresponding to "burger_crop.jpg".
  RectF roi{/*left=*/0, /*top=*/0, /*right=*/0.833333, /*bottom=*/1};
  ImageProcessingOptions image_processing_options{roi, /*rotation_degrees=*/0};

  // Extract both embeddings.
  MP_ASSERT_OK_AND_ASSIGN(
      const ImageEmbedderResult& image_result,
      image_embedder->Embed(image, image_processing_options));
  MP_ASSERT_OK_AND_ASSIGN(const ImageEmbedderResult& crop_result,
                          image_embedder->Embed(crop));

  // Check results.
  CheckMobileNetV3Result(image_result, false);
  CheckMobileNetV3Result(crop_result, false);
  // CheckCosineSimilarity.
  MP_ASSERT_OK_AND_ASSIGN(double similarity, ImageEmbedder::CosineSimilarity(
                                                 image_result.embeddings[0],
                                                 crop_result.embeddings[0]));
  double expected_similarity = 0.999931;
  EXPECT_LE(abs(similarity - expected_similarity), kSimilarityTolerancy);
}

TEST_F(ImageModeTest, SucceedsWithRotation) {
  auto options = std::make_unique<ImageEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetV3Embedder);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageEmbedder> image_embedder,
                          ImageEmbedder::Create(std::move(options)));
  // Load images: one is a rotated version of the other.
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  MP_ASSERT_OK_AND_ASSIGN(Image rotated,
                          DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                       "burger_rotated.jpg")));
  ImageProcessingOptions image_processing_options;
  image_processing_options.rotation_degrees = -90;

  // Extract both embeddings.
  MP_ASSERT_OK_AND_ASSIGN(const ImageEmbedderResult& image_result,
                          image_embedder->Embed(image));
  MP_ASSERT_OK_AND_ASSIGN(
      const ImageEmbedderResult& rotated_result,
      image_embedder->Embed(rotated, image_processing_options));

  // Check results.
  CheckMobileNetV3Result(image_result, false);
  CheckMobileNetV3Result(rotated_result, false);
  // CheckCosineSimilarity.
  MP_ASSERT_OK_AND_ASSIGN(double similarity, ImageEmbedder::CosineSimilarity(
                                                 image_result.embeddings[0],
                                                 rotated_result.embeddings[0]));
  double expected_similarity = 0.98223;
  EXPECT_LE(abs(similarity - expected_similarity), kSimilarityTolerancy);
}

TEST_F(ImageModeTest, SucceedsWithRegionOfInterestAndRotation) {
  auto options = std::make_unique<ImageEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetV3Embedder);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageEmbedder> image_embedder,
                          ImageEmbedder::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      Image crop, DecodeImageFromFile(
                      JoinPath("./", kTestDataDirectory, "burger_crop.jpg")));
  MP_ASSERT_OK_AND_ASSIGN(Image rotated,
                          DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                       "burger_rotated.jpg")));
  // Region-of-interest corresponding to burger_crop.jpg.
  RectF roi{/*left=*/0, /*top=*/0, /*right=*/1, /*bottom=*/0.8333333};
  ImageProcessingOptions image_processing_options{roi,
                                                  /*rotation_degrees=*/-90};

  // Extract both embeddings.
  MP_ASSERT_OK_AND_ASSIGN(const ImageEmbedderResult& crop_result,
                          image_embedder->Embed(crop));
  MP_ASSERT_OK_AND_ASSIGN(
      const ImageEmbedderResult& rotated_result,
      image_embedder->Embed(rotated, image_processing_options));

  // Check results.
  CheckMobileNetV3Result(crop_result, false);
  CheckMobileNetV3Result(rotated_result, false);
  // CheckCosineSimilarity.
  MP_ASSERT_OK_AND_ASSIGN(double similarity, ImageEmbedder::CosineSimilarity(
                                                 crop_result.embeddings[0],
                                                 rotated_result.embeddings[0]));
  double expected_similarity = 0.974683;
  EXPECT_LE(abs(similarity - expected_similarity), kSimilarityTolerancy);
}

class VideoModeTest : public tflite::testing::Test {};

TEST_F(VideoModeTest, FailsWithCallingWrongMethod) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetV3Embedder);
  options->running_mode = core::RunningMode::VIDEO;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageEmbedder> image_embedder,
                          ImageEmbedder::Create(std::move(options)));

  auto results = image_embedder->Embed(image);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the image mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));

  results = image_embedder->EmbedAsync(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the live stream mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));
  MP_ASSERT_OK(image_embedder->Close());
}

TEST_F(VideoModeTest, FailsWithOutOfOrderInputTimestamps) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetV3Embedder);
  options->running_mode = core::RunningMode::VIDEO;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageEmbedder> image_embedder,
                          ImageEmbedder::Create(std::move(options)));

  MP_ASSERT_OK(image_embedder->EmbedForVideo(image, 1));
  auto results = image_embedder->EmbedForVideo(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("timestamp must be monotonically increasing"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInvalidTimestampError))));
  MP_ASSERT_OK(image_embedder->EmbedForVideo(image, 2));
  MP_ASSERT_OK(image_embedder->Close());
}

TEST_F(VideoModeTest, Succeeds) {
  int iterations = 100;
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetV3Embedder);
  options->running_mode = core::RunningMode::VIDEO;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageEmbedder> image_embedder,
                          ImageEmbedder::Create(std::move(options)));

  ImageEmbedderResult previous_results;
  for (int i = 0; i < iterations; ++i) {
    MP_ASSERT_OK_AND_ASSIGN(auto results,
                            image_embedder->EmbedForVideo(image, i));
    CheckMobileNetV3Result(results, false);
    if (i > 0) {
      MP_ASSERT_OK_AND_ASSIGN(
          double similarity,
          ImageEmbedder::CosineSimilarity(results.embeddings[0],
                                          previous_results.embeddings[0]));
      double expected_similarity = 1.000000;
      EXPECT_LE(abs(similarity - expected_similarity), kSimilarityTolerancy);
    }
    previous_results = results;
  }
  MP_ASSERT_OK(image_embedder->Close());
}

class LiveStreamModeTest : public tflite::testing::Test {};

TEST_F(LiveStreamModeTest, FailsWithCallingWrongMethod) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetV3Embedder);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->result_callback = [](absl::StatusOr<ImageEmbedderResult>,
                                const Image& image, int64_t timestamp_ms) {};
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageEmbedder> image_embedder,
                          ImageEmbedder::Create(std::move(options)));

  auto results = image_embedder->Embed(image);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the image mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));

  results = image_embedder->EmbedForVideo(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the video mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));
  MP_ASSERT_OK(image_embedder->Close());
}

TEST_F(LiveStreamModeTest, FailsWithOutOfOrderInputTimestamps) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetV3Embedder);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->result_callback = [](absl::StatusOr<ImageEmbedderResult>,
                                const Image& image, int64_t timestamp_ms) {};
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageEmbedder> image_embedder,
                          ImageEmbedder::Create(std::move(options)));

  MP_ASSERT_OK(image_embedder->EmbedAsync(image, 1));
  auto status = image_embedder->EmbedAsync(image, 0);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("timestamp must be monotonically increasing"));
  EXPECT_THAT(status.GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInvalidTimestampError))));
  MP_ASSERT_OK(image_embedder->EmbedAsync(image, 2));
  MP_ASSERT_OK(image_embedder->Close());
}

struct LiveStreamModeResults {
  ImageEmbedderResult embedding_result;
  std::pair<int, int> image_size;
  int64_t timestamp_ms;
};

TEST_F(LiveStreamModeTest, Succeeds) {
  int iterations = 100;
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  std::vector<LiveStreamModeResults> results;
  auto options = std::make_unique<ImageEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetV3Embedder);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->result_callback =
      [&results](absl::StatusOr<ImageEmbedderResult> embedding_result,
                 const Image& image, int64_t timestamp_ms) {
        MP_ASSERT_OK(embedding_result.status());
        results.push_back(
            {.embedding_result = std::move(embedding_result).value(),
             .image_size = {image.width(), image.height()},
             .timestamp_ms = timestamp_ms});
      };
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageEmbedder> image_embedder,
                          ImageEmbedder::Create(std::move(options)));

  for (int i = 0; i < iterations; ++i) {
    MP_ASSERT_OK(image_embedder->EmbedAsync(image, i));
  }
  MP_ASSERT_OK(image_embedder->Close());

  // Due to the flow limiter, the total of outputs will be smaller than the
  // number of iterations.
  ASSERT_LE(results.size(), iterations);
  ASSERT_GT(results.size(), 0);
  int64_t timestamp_ms = -1;
  for (int i = 0; i < results.size(); ++i) {
    const auto& result = results[i];
    EXPECT_GT(result.timestamp_ms, timestamp_ms);
    timestamp_ms = result.timestamp_ms;
    EXPECT_EQ(result.image_size.first, image.width());
    EXPECT_EQ(result.image_size.second, image.height());
    CheckMobileNetV3Result(result.embedding_result, false);
    if (i > 0) {
      MP_ASSERT_OK_AND_ASSIGN(
          double similarity,
          ImageEmbedder::CosineSimilarity(
              result.embedding_result.embeddings[0],
              results[i - 1].embedding_result.embeddings[0]));
      double expected_similarity = 1.000000;
      EXPECT_LE(abs(similarity - expected_similarity), kSimilarityTolerancy);
    }
  }
}

}  // namespace
}  // namespace image_embedder
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
