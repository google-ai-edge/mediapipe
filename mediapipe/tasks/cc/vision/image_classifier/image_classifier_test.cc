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

#include "mediapipe/tasks/cc/vision/image_classifier/image_classifier.h"

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/containers/category.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"
#include "mediapipe/tasks/cc/components/containers/rect.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/test_util.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_classifier {
namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::components::containers::Category;
using ::mediapipe::tasks::components::containers::Classifications;
using ::mediapipe::tasks::components::containers::RectF;
using ::mediapipe::tasks::vision::core::ImageProcessingOptions;
using ::testing::HasSubstr;
using ::testing::Optional;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kMobileNetFloatWithMetadata[] = "mobilenet_v2_1.0_224.tflite";
constexpr char kMobileNetQuantizedWithMetadata[] =
    "mobilenet_v1_0.25_224_quant.tflite";
constexpr char kMobileNetQuantizedWithDummyScoreCalibration[] =
    "mobilenet_v1_0.25_224_quant_with_dummy_score_calibration.tflite";

// Checks that the two provided `ImageClassifierResult` are equal, with a
// tolerancy on floating-point score to account for numerical instabilities.
void ExpectApproximatelyEqual(const ImageClassifierResult& actual,
                              const ImageClassifierResult& expected) {
  const float kPrecision = 1e-6;
  ASSERT_EQ(actual.classifications.size(), expected.classifications.size());
  for (int i = 0; i < actual.classifications.size(); ++i) {
    const Classifications& a = actual.classifications[i];
    const Classifications& b = expected.classifications[i];
    EXPECT_EQ(a.head_index, b.head_index);
    EXPECT_EQ(a.head_name, b.head_name);
    EXPECT_EQ(a.categories.size(), b.categories.size());
    for (int j = 0; j < a.categories.size(); ++j) {
      const Category& x = a.categories[j];
      const Category& y = b.categories[j];
      EXPECT_EQ(x.index, y.index);
      EXPECT_NEAR(x.score, y.score, kPrecision);
      EXPECT_EQ(x.category_name, y.category_name);
      EXPECT_EQ(x.display_name, y.display_name);
    }
  }
}

// Generates expected results for "burger.jpg" using kMobileNetFloatWithMetadata
// with max_results set to 3.
ImageClassifierResult GenerateBurgerResults() {
  ImageClassifierResult result;
  result.classifications.emplace_back(Classifications{
      /*categories=*/{
          {/*index=*/934, /*score=*/0.793959200,
           /*category_name=*/"cheeseburger"},
          {/*index=*/932, /*score=*/0.027392805, /*category_name=*/"bagel"},
          {/*index=*/925, /*score=*/0.019340655,
           /*category_name=*/"guacamole"}},
      /*head_index=*/0,
      /*head_name=*/"probability"});
  return result;
}

// Generates expected results for "multi_objects.jpg" using
// kMobileNetFloatWithMetadata with max_results set to 1 and the right bounding
// box set around the soccer ball.
ImageClassifierResult GenerateSoccerBallResults() {
  ImageClassifierResult result;
  result.classifications.emplace_back(
      Classifications{/*categories=*/{{/*index=*/806, /*score=*/0.996527493,
                                       /*category_name=*/"soccer ball"}},
                      /*head_index=*/0,
                      /*head_name=*/"probability"});
  return result;
}

// A custom OpResolver only containing the Ops required by the test model.
class MobileNetQuantizedOpResolver : public ::tflite::MutableOpResolver {
 public:
  MobileNetQuantizedOpResolver() {
    AddBuiltin(::tflite::BuiltinOperator_AVERAGE_POOL_2D,
               ::tflite::ops::builtin::Register_AVERAGE_POOL_2D());
    AddBuiltin(::tflite::BuiltinOperator_CONV_2D,
               ::tflite::ops::builtin::Register_CONV_2D());
    AddBuiltin(::tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
               ::tflite::ops::builtin::Register_DEPTHWISE_CONV_2D());
    AddBuiltin(::tflite::BuiltinOperator_RESHAPE,
               ::tflite::ops::builtin::Register_RESHAPE());
    AddBuiltin(::tflite::BuiltinOperator_SOFTMAX,
               ::tflite::ops::builtin::Register_SOFTMAX());
  }

  MobileNetQuantizedOpResolver(const MobileNetQuantizedOpResolver& r) = delete;
};

// A custom OpResolver missing Ops required by the test model.
class MobileNetQuantizedOpResolverMissingOps
    : public ::tflite::MutableOpResolver {
 public:
  MobileNetQuantizedOpResolverMissingOps() {
    AddBuiltin(::tflite::BuiltinOperator_SOFTMAX,
               ::tflite::ops::builtin::Register_SOFTMAX());
  }

  MobileNetQuantizedOpResolverMissingOps(
      const MobileNetQuantizedOpResolverMissingOps& r) = delete;
};

class CreateTest : public tflite::testing::Test {};

TEST_F(CreateTest, SucceedsWithSelectiveOpResolver) {
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetQuantizedWithMetadata);
  options->base_options.op_resolver =
      std::make_unique<MobileNetQuantizedOpResolver>();

  MP_ASSERT_OK(ImageClassifier::Create(std::move(options)));
}

TEST_F(CreateTest, FailsWithSelectiveOpResolverMissingOps) {
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetQuantizedWithMetadata);
  options->base_options.op_resolver =
      std::make_unique<MobileNetQuantizedOpResolverMissingOps>();

  auto image_classifier = ImageClassifier::Create(std::move(options));

  // TODO: Make MediaPipe InferenceCalculator report the detailed
  // interpreter errors (e.g., "Encountered unresolved custom op").
  EXPECT_EQ(image_classifier.status().code(), absl::StatusCode::kInternal);
  EXPECT_THAT(image_classifier.status().message(),
              HasSubstr("interpreter_builder(&interpreter) == kTfLiteOk"));
}
TEST_F(CreateTest, FailsWithMissingModel) {
  auto image_classifier =
      ImageClassifier::Create(std::make_unique<ImageClassifierOptions>());

  EXPECT_EQ(image_classifier.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      image_classifier.status().message(),
      HasSubstr("ExternalFile must specify at least one of 'file_content', "
                "'file_name', 'file_pointer_meta' or 'file_descriptor_meta'."));
  EXPECT_THAT(image_classifier.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(CreateTest, FailsWithInvalidMaxResults) {
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetQuantizedWithMetadata);
  options->classifier_options.max_results = 0;

  auto image_classifier = ImageClassifier::Create(std::move(options));

  EXPECT_EQ(image_classifier.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(image_classifier.status().message(),
              HasSubstr("Invalid `max_results` option"));
  EXPECT_THAT(image_classifier.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(CreateTest, FailsWithCombinedAllowlistAndDenylist) {
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetQuantizedWithMetadata);
  options->classifier_options.category_allowlist = {"foo"};
  options->classifier_options.category_denylist = {"bar"};

  auto image_classifier = ImageClassifier::Create(std::move(options));

  EXPECT_EQ(image_classifier.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(image_classifier.status().message(),
              HasSubstr("mutually exclusive options"));
  EXPECT_THAT(image_classifier.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(CreateTest, FailsWithIllegalCallbackInImageOrVideoMode) {
  for (auto running_mode :
       {core::RunningMode::IMAGE, core::RunningMode::VIDEO}) {
    auto options = std::make_unique<ImageClassifierOptions>();
    options->base_options.model_asset_path =
        JoinPath("./", kTestDataDirectory, kMobileNetQuantizedWithMetadata);
    options->running_mode = running_mode;
    options->result_callback = [](absl::StatusOr<ImageClassifierResult>,
                                  const Image& image, int64_t timestamp_ms) {};

    auto image_classifier = ImageClassifier::Create(std::move(options));

    EXPECT_EQ(image_classifier.status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_THAT(
        image_classifier.status().message(),
        HasSubstr("a user-defined result callback shouldn't be provided"));
    EXPECT_THAT(image_classifier.status().GetPayload(kMediaPipeTasksPayload),
                Optional(absl::Cord(absl::StrCat(
                    MediaPipeTasksStatus::kInvalidTaskGraphConfigError))));
  }
}

TEST_F(CreateTest, FailsWithMissingCallbackInLiveStreamMode) {
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetQuantizedWithMetadata);
  options->running_mode = core::RunningMode::LIVE_STREAM;

  auto image_classifier = ImageClassifier::Create(std::move(options));

  EXPECT_EQ(image_classifier.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(image_classifier.status().message(),
              HasSubstr("a user-defined result callback must be provided"));
  EXPECT_THAT(image_classifier.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kInvalidTaskGraphConfigError))));
}

class ImageModeTest : public tflite::testing::Test {};

TEST_F(ImageModeTest, FailsWithCallingWrongMethod) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  auto results = image_classifier->ClassifyForVideo(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the video mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));

  results = image_classifier->ClassifyAsync(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the live stream mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));
  MP_ASSERT_OK(image_classifier->Close());
}

TEST_F(ImageModeTest, SucceedsWithFloatModel) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata);
  options->classifier_options.max_results = 3;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  MP_ASSERT_OK_AND_ASSIGN(auto results, image_classifier->Classify(image));

  ExpectApproximatelyEqual(results, GenerateBurgerResults());
}

TEST_F(ImageModeTest, SucceedsWithQuantizedModel) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetQuantizedWithMetadata);
  // Due to quantization, multiple results beyond top-1 have the exact same
  // score. This leads to unstability in results ordering, so we only ask for
  // top-1 here.
  options->classifier_options.max_results = 1;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  MP_ASSERT_OK_AND_ASSIGN(auto results, image_classifier->Classify(image));

  ImageClassifierResult expected;
  expected.classifications.emplace_back(
      Classifications{/*categories=*/{{/*index=*/934, /*score=*/0.97265625,
                                       /*category_name=*/"cheeseburger"}},
                      /*head_index=*/0,
                      /*head_name=*/"probability"});
  ExpectApproximatelyEqual(results, expected);
}

TEST_F(ImageModeTest, SucceedsWithMaxResultsOption) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata);
  options->classifier_options.max_results = 1;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  MP_ASSERT_OK_AND_ASSIGN(auto results, image_classifier->Classify(image));

  ImageClassifierResult expected;
  expected.classifications.emplace_back(
      Classifications{/*categories=*/{{/*index=*/934, /*score=*/0.7939592,
                                       /*category_name=*/"cheeseburger"}},
                      /*head_index=*/0,
                      /*head_name=*/"probability"});
  ExpectApproximatelyEqual(results, expected);
}

TEST_F(ImageModeTest, SucceedsWithScoreThresholdOption) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata);
  options->classifier_options.score_threshold = 0.02;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  MP_ASSERT_OK_AND_ASSIGN(auto results, image_classifier->Classify(image));

  ImageClassifierResult expected;
  expected.classifications.emplace_back(Classifications{
      /*categories=*/{
          {/*index=*/934, /*score=*/0.7939592,
           /*category_name=*/"cheeseburger"},
          {/*index=*/932, /*score=*/0.027392805, /*category_name=*/"bagel"}},
      /*head_index=*/0,
      /*head_name=*/"probability"});
  ExpectApproximatelyEqual(results, expected);
}

TEST_F(ImageModeTest, SucceedsWithAllowlistOption) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata);
  options->classifier_options.category_allowlist = {"cheeseburger", "guacamole",
                                                    "meat loaf"};
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  MP_ASSERT_OK_AND_ASSIGN(auto results, image_classifier->Classify(image));

  ImageClassifierResult expected;
  expected.classifications.emplace_back(Classifications{
      /*categories=*/{
          {/*index=*/934, /*score=*/0.7939592,
           /*category_name=*/"cheeseburger"},
          {/*index=*/925, /*score=*/0.019340655, /*category_name=*/"guacamole"},
          {/*index=*/963, /*score=*/0.0063278517,
           /*category_name=*/"meat loaf"}},
      /*head_index=*/0,
      /*head_name=*/"probability"});
  ExpectApproximatelyEqual(results, expected);
}

TEST_F(ImageModeTest, SucceedsWithDenylistOption) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata);
  options->classifier_options.max_results = 3;
  options->classifier_options.category_denylist = {"bagel"};
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  MP_ASSERT_OK_AND_ASSIGN(auto results, image_classifier->Classify(image));

  ImageClassifierResult expected;
  expected.classifications.emplace_back(Classifications{
      /*categories=*/{
          {/*index=*/934, /*score=*/0.7939592,
           /*category_name=*/"cheeseburger"},
          {/*index=*/925, /*score=*/0.019340655, /*category_name=*/"guacamole"},
          {/*index=*/963, /*score=*/0.0063278517,
           /*category_name=*/"meat loaf"}},
      /*head_index=*/0,
      /*head_name=*/"probability"});
  ExpectApproximatelyEqual(results, expected);
}

TEST_F(ImageModeTest, SucceedsWithScoreCalibration) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path = JoinPath(
      "./", kTestDataDirectory, kMobileNetQuantizedWithDummyScoreCalibration);
  // Due to quantization, multiple results beyond top-1 have the exact same
  // score. This leads to unstability in results ordering, so we only ask for
  // top-1 here.
  options->classifier_options.max_results = 1;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  MP_ASSERT_OK_AND_ASSIGN(auto results, image_classifier->Classify(image));

  ImageClassifierResult expected;
  expected.classifications.emplace_back(
      Classifications{/*categories=*/{{/*index=*/934, /*score=*/0.725648628,
                                       /*category_name=*/"cheeseburger"}},
                      /*head_index=*/0,
                      /*head_name=*/"probability"});
  ExpectApproximatelyEqual(results, expected);
}

TEST_F(ImageModeTest, SucceedsWithRegionOfInterest) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                       "multi_objects.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata);
  options->classifier_options.max_results = 1;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));
  // Region-of-interest around the soccer ball.
  RectF roi{/*left=*/0.45, /*top=*/0.3075, /*right=*/0.614, /*bottom=*/0.7345};
  ImageProcessingOptions image_processing_options{roi, /*rotation_degrees=*/0};

  MP_ASSERT_OK_AND_ASSIGN(auto results, image_classifier->Classify(
                                            image, image_processing_options));

  ExpectApproximatelyEqual(results, GenerateSoccerBallResults());
}

TEST_F(ImageModeTest, SucceedsWithRotation) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                       "burger_rotated.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata);
  options->classifier_options.max_results = 3;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  // Specify a 90° anti-clockwise rotation.
  ImageProcessingOptions image_processing_options;
  image_processing_options.rotation_degrees = -90;

  MP_ASSERT_OK_AND_ASSIGN(auto results, image_classifier->Classify(
                                            image, image_processing_options));

  // Results differ slightly from the non-rotated image, but that's expected
  // as models are very sensitive to the slightest numerical differences
  // introduced by the rotation and JPG encoding.
  ImageClassifierResult expected;
  expected.classifications.emplace_back(Classifications{
      /*categories=*/{
          {/*index=*/934, /*score=*/0.754467, /*category_name=*/"cheeseburger"},
          {/*index=*/925, /*score=*/0.0288028, /*category_name=*/"guacamole"},
          {/*index=*/932, /*score=*/0.0286119, /*category_name=*/"bagel"}},
      /*head_index=*/0,
      /*head_name=*/"probability"});
  ExpectApproximatelyEqual(results, expected);
}

TEST_F(ImageModeTest, SucceedsWithRegionOfInterestAndRotation) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                "multi_objects_rotated.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata);
  options->classifier_options.max_results = 1;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));
  // Region-of-interest around the soccer ball, with 90° anti-clockwise
  // rotation.
  RectF roi{/*left=*/0.2655, /*top=*/0.45, /*right=*/0.6925,
            /*bottom=*/0.614};
  ImageProcessingOptions image_processing_options{roi,
                                                  /*rotation_degrees=*/-90};

  MP_ASSERT_OK_AND_ASSIGN(auto results, image_classifier->Classify(
                                            image, image_processing_options));

  ImageClassifierResult expected;
  expected.classifications.emplace_back(
      Classifications{/*categories=*/{{/*index=*/806, /*score=*/0.997684,
                                       /*category_name=*/"soccer ball"}},
                      /*head_index=*/0,
                      /*head_name=*/"probability"});
  ExpectApproximatelyEqual(results, expected);
}

// Testing all these once with ImageClassifier.
TEST_F(ImageModeTest, FailsWithInvalidImageProcessingOptions) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                       "multi_objects.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  // Invalid: left > right.
  RectF roi{/*left=*/0.9, /*top=*/0, /*right=*/0.1, /*bottom=*/1};
  ImageProcessingOptions image_processing_options{roi,
                                                  /*rotation_degrees=*/0};
  auto results = image_classifier->Classify(image, image_processing_options);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("Expected RectF with left < right and top < bottom"));
  EXPECT_THAT(
      results.status().GetPayload(kMediaPipeTasksPayload),
      Optional(absl::Cord(absl::StrCat(
          MediaPipeTasksStatus::kImageProcessingInvalidArgumentError))));

  // Invalid: top > bottom.
  roi = {/*left=*/0, /*top=*/0.9, /*right=*/1, /*bottom=*/0.1};
  image_processing_options = {roi,
                              /*rotation_degrees=*/0};
  results = image_classifier->Classify(image, image_processing_options);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("Expected RectF with left < right and top < bottom"));
  EXPECT_THAT(
      results.status().GetPayload(kMediaPipeTasksPayload),
      Optional(absl::Cord(absl::StrCat(
          MediaPipeTasksStatus::kImageProcessingInvalidArgumentError))));

  // Invalid: coordinates out of [0,1] range.
  roi = {/*left=*/-0.1, /*top=*/0, /*right=*/1, /*bottom=*/1};
  image_processing_options = {roi,
                              /*rotation_degrees=*/0};
  results = image_classifier->Classify(image, image_processing_options);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("Expected RectF values to be in [0,1]"));
  EXPECT_THAT(
      results.status().GetPayload(kMediaPipeTasksPayload),
      Optional(absl::Cord(absl::StrCat(
          MediaPipeTasksStatus::kImageProcessingInvalidArgumentError))));

  // Invalid: rotation not a multiple of 90°.
  image_processing_options = {/*region_of_interest=*/std::nullopt,
                              /*rotation_degrees=*/1};
  results = image_classifier->Classify(image, image_processing_options);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("Expected rotation to be a multiple of 90°"));
  EXPECT_THAT(
      results.status().GetPayload(kMediaPipeTasksPayload),
      Optional(absl::Cord(absl::StrCat(
          MediaPipeTasksStatus::kImageProcessingInvalidArgumentError))));
}

class VideoModeTest : public tflite::testing::Test {};

TEST_F(VideoModeTest, FailsWithCallingWrongMethod) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata);
  options->running_mode = core::RunningMode::VIDEO;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  auto results = image_classifier->Classify(image);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the image mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));

  results = image_classifier->ClassifyAsync(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the live stream mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));
  MP_ASSERT_OK(image_classifier->Close());
}

TEST_F(VideoModeTest, FailsWithOutOfOrderInputTimestamps) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata);
  options->running_mode = core::RunningMode::VIDEO;
  options->classifier_options.max_results = 3;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  MP_ASSERT_OK(image_classifier->ClassifyForVideo(image, 1));
  auto results = image_classifier->ClassifyForVideo(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("timestamp must be monotonically increasing"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInvalidTimestampError))));
  MP_ASSERT_OK(image_classifier->ClassifyForVideo(image, 2));
  MP_ASSERT_OK(image_classifier->Close());
}

TEST_F(VideoModeTest, Succeeds) {
  int iterations = 100;
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata);
  options->running_mode = core::RunningMode::VIDEO;
  options->classifier_options.max_results = 3;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  for (int i = 0; i < iterations; ++i) {
    MP_ASSERT_OK_AND_ASSIGN(auto results,
                            image_classifier->ClassifyForVideo(image, i));
    ExpectApproximatelyEqual(results, GenerateBurgerResults());
  }
  MP_ASSERT_OK(image_classifier->Close());
}

TEST_F(VideoModeTest, SucceedsWithRegionOfInterest) {
  int iterations = 100;
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                       "multi_objects.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata);
  options->running_mode = core::RunningMode::VIDEO;
  options->classifier_options.max_results = 1;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));
  // Crop around the soccer ball.
  // Region-of-interest around the soccer ball.
  RectF roi{/*left=*/0.45, /*top=*/0.3075, /*right=*/0.614, /*bottom=*/0.7345};
  ImageProcessingOptions image_processing_options{roi, /*rotation_degrees=*/0};

  for (int i = 0; i < iterations; ++i) {
    MP_ASSERT_OK_AND_ASSIGN(
        auto results,
        image_classifier->ClassifyForVideo(image, i, image_processing_options));
    ExpectApproximatelyEqual(results, GenerateSoccerBallResults());
  }
  MP_ASSERT_OK(image_classifier->Close());
}

class LiveStreamModeTest : public tflite::testing::Test {};

TEST_F(LiveStreamModeTest, FailsWithCallingWrongMethod) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->result_callback = [](absl::StatusOr<ImageClassifierResult>,
                                const Image& image, int64_t timestamp_ms) {};
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  auto results = image_classifier->Classify(image);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the image mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));

  results = image_classifier->ClassifyForVideo(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the video mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));
  MP_ASSERT_OK(image_classifier->Close());
}

TEST_F(LiveStreamModeTest, FailsWithOutOfOrderInputTimestamps) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->result_callback = [](absl::StatusOr<ImageClassifierResult>,
                                const Image& image, int64_t timestamp_ms) {};
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  MP_ASSERT_OK(image_classifier->ClassifyAsync(image, 1));
  auto status = image_classifier->ClassifyAsync(image, 0);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("timestamp must be monotonically increasing"));
  EXPECT_THAT(status.GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInvalidTimestampError))));
  MP_ASSERT_OK(image_classifier->ClassifyAsync(image, 2));
  MP_ASSERT_OK(image_classifier->Close());
}

struct LiveStreamModeResults {
  ImageClassifierResult classification_result;
  std::pair<int, int> image_size;
  int64_t timestamp_ms;
};

TEST_F(LiveStreamModeTest, Succeeds) {
  int iterations = 100;
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  std::vector<LiveStreamModeResults> results;
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->classifier_options.max_results = 3;
  options->result_callback =
      [&results](absl::StatusOr<ImageClassifierResult> classification_result,
                 const Image& image, int64_t timestamp_ms) {
        MP_ASSERT_OK(classification_result.status());
        results.push_back(
            {.classification_result = std::move(classification_result).value(),
             .image_size = {image.width(), image.height()},
             .timestamp_ms = timestamp_ms});
      };
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  for (int i = 0; i < iterations; ++i) {
    MP_ASSERT_OK(image_classifier->ClassifyAsync(image, i));
  }
  MP_ASSERT_OK(image_classifier->Close());

  // Due to the flow limiter, the total of outputs will be smaller than the
  // number of iterations.
  ASSERT_LE(results.size(), iterations);
  ASSERT_GT(results.size(), 0);
  int64_t timestamp_ms = -1;
  for (const auto& result : results) {
    EXPECT_GT(result.timestamp_ms, timestamp_ms);
    timestamp_ms = result.timestamp_ms;
    EXPECT_EQ(result.image_size.first, image.width());
    EXPECT_EQ(result.image_size.second, image.height());
    ExpectApproximatelyEqual(result.classification_result,
                             GenerateBurgerResults());
  }
}

TEST_F(LiveStreamModeTest, SucceedsWithRegionOfInterest) {
  int iterations = 100;
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                       "multi_objects.jpg")));
  std::vector<LiveStreamModeResults> results;
  auto options = std::make_unique<ImageClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->classifier_options.max_results = 1;
  options->result_callback =
      [&results](absl::StatusOr<ImageClassifierResult> classification_result,
                 const Image& image, int64_t timestamp_ms) {
        MP_ASSERT_OK(classification_result.status());
        results.push_back(
            {.classification_result = std::move(classification_result).value(),
             .image_size = {image.width(), image.height()},
             .timestamp_ms = timestamp_ms});
      };
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));
  // Crop around the soccer ball.
  RectF roi{/*left=*/0.45, /*top=*/0.3075, /*right=*/0.614, /*bottom=*/0.7345};
  ImageProcessingOptions image_processing_options{roi, /*rotation_degrees=*/0};

  for (int i = 0; i < iterations; ++i) {
    MP_ASSERT_OK(
        image_classifier->ClassifyAsync(image, i, image_processing_options));
  }
  MP_ASSERT_OK(image_classifier->Close());

  // Due to the flow limiter, the total of outputs will be smaller than the
  // number of iterations.
  ASSERT_LE(results.size(), iterations);
  ASSERT_GT(results.size(), 0);
  int64_t timestamp_ms = -1;
  for (const auto& result : results) {
    EXPECT_GT(result.timestamp_ms, timestamp_ms);
    timestamp_ms = result.timestamp_ms;
    EXPECT_EQ(result.image_size.first, image.width());
    EXPECT_EQ(result.image_size.second, image.height());
    ExpectApproximatelyEqual(result.classification_result,
                             GenerateSoccerBallResults());
  }
}

}  // namespace
}  // namespace image_classifier
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
