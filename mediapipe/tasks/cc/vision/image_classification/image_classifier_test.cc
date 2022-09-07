/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

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

#include "mediapipe/tasks/cc/vision/image_classification/image_classifier.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/classifier_options.pb.h"
#include "mediapipe/tasks/cc/components/containers/category.pb.h"
#include "mediapipe/tasks/cc/components/containers/classifications.pb.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/vision/image_classification/image_classifier_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/shims/cc/shims_test_util.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/mutable_op_resolver.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace {

using ::mediapipe::file::JoinPath;
using ::testing::HasSubstr;
using ::testing::Optional;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kMobileNetFloatWithMetadata[] = "mobilenet_v2_1.0_224.tflite";
constexpr char kMobileNetQuantizedWithMetadata[] =
    "mobilenet_v1_0.25_224_quant.tflite";

// Checks that the two provided `ClassificationResult` are equal, with a
// tolerancy on floating-point score to account for numerical instabilities.
void ExpectApproximatelyEqual(const ClassificationResult& actual,
                              const ClassificationResult& expected) {
  const float kPrecision = 1e-6;
  ASSERT_EQ(actual.classifications_size(), expected.classifications_size());
  for (int i = 0; i < actual.classifications_size(); ++i) {
    const Classifications& a = actual.classifications(i);
    const Classifications& b = expected.classifications(i);
    EXPECT_EQ(a.head_index(), b.head_index());
    EXPECT_EQ(a.head_name(), b.head_name());
    EXPECT_EQ(a.entries_size(), b.entries_size());
    for (int j = 0; j < a.entries_size(); ++j) {
      const ClassificationEntry& x = a.entries(j);
      const ClassificationEntry& y = b.entries(j);
      EXPECT_EQ(x.timestamp_ms(), y.timestamp_ms());
      EXPECT_EQ(x.categories_size(), y.categories_size());
      for (int k = 0; k < x.categories_size(); ++k) {
        EXPECT_EQ(x.categories(k).index(), y.categories(k).index());
        EXPECT_EQ(x.categories(k).category_name(),
                  y.categories(k).category_name());
        EXPECT_EQ(x.categories(k).display_name(),
                  y.categories(k).display_name());
        EXPECT_NEAR(x.categories(k).score(), y.categories(k).score(),
                    kPrecision);
      }
    }
  }
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

class CreateTest : public tflite_shims::testing::Test {};

TEST_F(CreateTest, SucceedsWithSelectiveOpResolver) {
  auto options = std::make_unique<ImageClassifierOptions>();
  options->mutable_base_options()->mutable_model_file()->set_file_name(
      JoinPath("./", kTestDataDirectory, kMobileNetQuantizedWithMetadata));

  MP_ASSERT_OK(ImageClassifier::Create(
      std::move(options), absl::make_unique<MobileNetQuantizedOpResolver>()));
}

TEST_F(CreateTest, FailsWithSelectiveOpResolverMissingOps) {
  auto options = std::make_unique<ImageClassifierOptions>();
  options->mutable_base_options()->mutable_model_file()->set_file_name(
      JoinPath("./", kTestDataDirectory, kMobileNetQuantizedWithMetadata));

  auto image_classifier_or = ImageClassifier::Create(
      std::move(options),
      absl::make_unique<MobileNetQuantizedOpResolverMissingOps>());

  // TODO: Make MediaPipe InferenceCalculator report the detailed
  // interpreter errors (e.g., "Encountered unresolved custom op").
  EXPECT_EQ(image_classifier_or.status().code(), absl::StatusCode::kInternal);
  EXPECT_THAT(image_classifier_or.status().message(),
              HasSubstr("interpreter_builder(&interpreter) == kTfLiteOk"));
}
TEST_F(CreateTest, FailsWithMissingModel) {
  auto image_classifier_or =
      ImageClassifier::Create(std::make_unique<ImageClassifierOptions>());

  EXPECT_EQ(image_classifier_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      image_classifier_or.status().message(),
      HasSubstr("ExternalFile must specify at least one of 'file_content', "
                "'file_name' or 'file_descriptor_meta'."));
  EXPECT_THAT(image_classifier_or.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(CreateTest, FailsWithInvalidMaxResults) {
  auto options = std::make_unique<ImageClassifierOptions>();
  options->mutable_base_options()->mutable_model_file()->set_file_name(
      JoinPath("./", kTestDataDirectory, kMobileNetQuantizedWithMetadata));
  options->mutable_classifier_options()->set_max_results(0);

  auto image_classifier_or = ImageClassifier::Create(std::move(options));

  EXPECT_EQ(image_classifier_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(image_classifier_or.status().message(),
              HasSubstr("Invalid `max_results` option"));
  EXPECT_THAT(image_classifier_or.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(CreateTest, FailsWithCombinedAllowlistAndDenylist) {
  auto options = std::make_unique<ImageClassifierOptions>();
  options->mutable_base_options()->mutable_model_file()->set_file_name(
      JoinPath("./", kTestDataDirectory, kMobileNetQuantizedWithMetadata));
  options->mutable_classifier_options()->add_category_allowlist("foo");
  options->mutable_classifier_options()->add_category_denylist("bar");

  auto image_classifier_or = ImageClassifier::Create(std::move(options));

  EXPECT_EQ(image_classifier_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(image_classifier_or.status().message(),
              HasSubstr("mutually exclusive options"));
  EXPECT_THAT(image_classifier_or.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

class ClassifyTest : public tflite_shims::testing::Test {};

TEST_F(ClassifyTest, SucceedsWithFloatModel) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->mutable_base_options()->mutable_model_file()->set_file_name(
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata));
  options->mutable_classifier_options()->set_max_results(3);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  MP_ASSERT_OK_AND_ASSIGN(auto results, image_classifier->Classify(image));

  ExpectApproximatelyEqual(results, ParseTextProtoOrDie<ClassificationResult>(
                                        R"pb(classifications {
                                               entries {
                                                 categories {
                                                   index: 934
                                                   score: 0.7939592
                                                   category_name: "cheeseburger"
                                                 }
                                                 categories {
                                                   index: 932
                                                   score: 0.027392805
                                                   category_name: "bagel"
                                                 }
                                                 categories {
                                                   index: 925
                                                   score: 0.019340655
                                                   category_name: "guacamole"
                                                 }
                                                 timestamp_ms: 0
                                               }
                                               head_index: 0
                                               head_name: "probability"
                                             })pb"));
}

TEST_F(ClassifyTest, SucceedsWithQuantizedModel) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->mutable_base_options()->mutable_model_file()->set_file_name(
      JoinPath("./", kTestDataDirectory, kMobileNetQuantizedWithMetadata));
  // Due to quantization, multiple results beyond top-1 have the exact same
  // score. This leads to unstability in results ordering, so we only ask for
  // top-1 here.
  options->mutable_classifier_options()->set_max_results(1);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  MP_ASSERT_OK_AND_ASSIGN(auto results, image_classifier->Classify(image));

  ExpectApproximatelyEqual(results, ParseTextProtoOrDie<ClassificationResult>(
                                        R"pb(classifications {
                                               entries {
                                                 categories {
                                                   index: 934
                                                   score: 0.97265625
                                                   category_name: "cheeseburger"
                                                 }
                                                 timestamp_ms: 0
                                               }
                                               head_index: 0
                                               head_name: "probability"
                                             })pb"));
}

TEST_F(ClassifyTest, SucceedsWithMaxResultsOption) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->mutable_base_options()->mutable_model_file()->set_file_name(
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata));
  options->mutable_classifier_options()->set_max_results(1);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  MP_ASSERT_OK_AND_ASSIGN(auto results, image_classifier->Classify(image));

  ExpectApproximatelyEqual(results, ParseTextProtoOrDie<ClassificationResult>(
                                        R"pb(classifications {
                                               entries {
                                                 categories {
                                                   index: 934
                                                   score: 0.7939592
                                                   category_name: "cheeseburger"
                                                 }
                                                 timestamp_ms: 0
                                               }
                                               head_index: 0
                                               head_name: "probability"
                                             })pb"));
}

TEST_F(ClassifyTest, SucceedsWithScoreThresholdOption) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->mutable_base_options()->mutable_model_file()->set_file_name(
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata));
  options->mutable_classifier_options()->set_score_threshold(0.02);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  MP_ASSERT_OK_AND_ASSIGN(auto results, image_classifier->Classify(image));

  ExpectApproximatelyEqual(results, ParseTextProtoOrDie<ClassificationResult>(
                                        R"pb(classifications {
                                               entries {
                                                 categories {
                                                   index: 934
                                                   score: 0.7939592
                                                   category_name: "cheeseburger"
                                                 }
                                                 categories {
                                                   index: 932
                                                   score: 0.027392805
                                                   category_name: "bagel"
                                                 }
                                                 timestamp_ms: 0
                                               }
                                               head_index: 0
                                               head_name: "probability"
                                             })pb"));
}

TEST_F(ClassifyTest, SucceedsWithAllowlistOption) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->mutable_base_options()->mutable_model_file()->set_file_name(
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata));
  options->mutable_classifier_options()->add_category_allowlist("cheeseburger");
  options->mutable_classifier_options()->add_category_allowlist("guacamole");
  options->mutable_classifier_options()->add_category_allowlist("meat loaf");
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  MP_ASSERT_OK_AND_ASSIGN(auto results, image_classifier->Classify(image));

  ExpectApproximatelyEqual(results, ParseTextProtoOrDie<ClassificationResult>(
                                        R"pb(classifications {
                                               entries {
                                                 categories {
                                                   index: 934
                                                   score: 0.7939592
                                                   category_name: "cheeseburger"
                                                 }
                                                 categories {
                                                   index: 925
                                                   score: 0.019340655
                                                   category_name: "guacamole"
                                                 }
                                                 categories {
                                                   index: 963
                                                   score: 0.0063278517
                                                   category_name: "meat loaf"
                                                 }
                                                 timestamp_ms: 0
                                               }
                                               head_index: 0
                                               head_name: "probability"
                                             })pb"));
}

TEST_F(ClassifyTest, SucceedsWithDenylistOption) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "burger.jpg")));
  auto options = std::make_unique<ImageClassifierOptions>();
  options->mutable_base_options()->mutable_model_file()->set_file_name(
      JoinPath("./", kTestDataDirectory, kMobileNetFloatWithMetadata));
  options->mutable_classifier_options()->set_max_results(3);
  options->mutable_classifier_options()->add_category_denylist("bagel");
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageClassifier> image_classifier,
                          ImageClassifier::Create(std::move(options)));

  MP_ASSERT_OK_AND_ASSIGN(auto results, image_classifier->Classify(image));

  ExpectApproximatelyEqual(results, ParseTextProtoOrDie<ClassificationResult>(
                                        R"pb(classifications {
                                               entries {
                                                 categories {
                                                   index: 934
                                                   score: 0.7939592
                                                   category_name: "cheeseburger"
                                                 }
                                                 categories {
                                                   index: 925
                                                   score: 0.019340655
                                                   category_name: "guacamole"
                                                 }
                                                 categories {
                                                   index: 963
                                                   score: 0.0063278517
                                                   category_name: "meat loaf"
                                                 }
                                                 timestamp_ms: 0
                                               }
                                               head_index: 0
                                               head_name: "probability"
                                             })pb"));
}

}  // namespace
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
