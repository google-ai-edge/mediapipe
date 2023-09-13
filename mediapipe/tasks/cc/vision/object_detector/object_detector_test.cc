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

#include "mediapipe/tasks/cc/vision/object_detector/object_detector.h"

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/components/containers/detection_result.h"
#include "mediapipe/tasks/cc/components/containers/rect.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/utils/image_tensor_specs.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/test_util.h"

namespace tflite {
namespace ops {
namespace custom {

// Forward declaration for the custom Detection_PostProcess op.
//
// See:
// https://medium.com/@bsramasubramanian/running-a-tensorflow-lite-model-in-python-with-custom-ops-9b2b46efd355
TfLiteRegistration* Register_DETECTION_POSTPROCESS();

}  // namespace custom
}  // namespace ops
}  // namespace tflite

namespace mediapipe {
namespace tasks {
namespace vision {
namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::components::containers::ConvertToDetectionResult;
using ::mediapipe::tasks::components::containers::Detection;
using ::mediapipe::tasks::components::containers::DetectionResult;
using ::mediapipe::tasks::components::containers::RectF;
using ::mediapipe::tasks::vision::core::ImageProcessingOptions;
using ::testing::HasSubstr;
using ::testing::Optional;
using DetectionProto = mediapipe::Detection;

constexpr absl::string_view kTestDataDirectory{
    "/mediapipe/tasks/testdata/vision/"};
constexpr absl::string_view kMobileSsdWithMetadata{
    "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite"};
constexpr absl::string_view kMobileSsdWithDummyScoreCalibration{
    "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29_with_dummy_score_calibration."
    "tflite"};
// The model has different output tensor order.
constexpr absl::string_view kEfficientDetWithMetadata{
    "coco_efficientdet_lite0_v1_1.0_quant_2021_09_06.tflite"};
constexpr absl::string_view kEfficientDetWithoutNms{
    "efficientdet_lite0_fp16_no_nms.tflite"};

// Checks that the two provided `Detection` proto vectors are equal, with a
// tolerancy on floating-point scores to account for numerical instabilities.
// If the proto definition changes, please also change this function.
void ExpectApproximatelyEqual(const ObjectDetectorResult& actual,
                              const ObjectDetectorResult& expected) {
  const float kPrecision = 1e-6;
  EXPECT_EQ(actual.detections.size(), expected.detections.size());
  for (int i = 0; i < actual.detections.size(); ++i) {
    const Detection& a = actual.detections[i];
    const Detection& b = expected.detections[i];
    EXPECT_EQ(a.bounding_box, b.bounding_box);
    EXPECT_EQ(a.categories.size(), 1);
    EXPECT_EQ(b.categories.size(), 1);
    EXPECT_EQ(a.categories[0].category_name, b.categories[0].category_name);
    EXPECT_NEAR(a.categories[0].score, b.categories[0].score, kPrecision);
  }
}

std::vector<DetectionProto>
GenerateMobileSsdNoImageResizingFullExpectedResults() {
  return {ParseTextProtoOrDie<DetectionProto>(R"pb(
            label: "cat"
            score: 0.6210937
            location_data {
              format: BOUNDING_BOX
              bounding_box { xmin: 15 ymin: 197 width: 98 height: 99 }
            })pb"),
          ParseTextProtoOrDie<DetectionProto>(R"pb(
            label: "cat"
            score: 0.609375
            location_data {
              format: BOUNDING_BOX
              bounding_box { xmin: 150 ymin: 78 width: 104 height: 223 }
            })pb"),
          ParseTextProtoOrDie<DetectionProto>(R"pb(
            label: "cat"
            score: 0.5
            location_data {
              format: BOUNDING_BOX
              bounding_box { xmin: 64 ymin: 199 width: 42 height: 101 }
            })pb"),
          ParseTextProtoOrDie<DetectionProto>(R"pb(
            label: "dog"
            score: 0.5
            location_data {
              format: BOUNDING_BOX
              bounding_box { xmin: 14 ymin: 110 width: 153 height: 193 }
            })pb")};
}

// OpResolver including the custom Detection_PostProcess op.
class MobileSsdQuantizedOpResolver : public ::tflite::MutableOpResolver {
 public:
  MobileSsdQuantizedOpResolver() {
    AddBuiltin(::tflite::BuiltinOperator_CONCATENATION,
               ::tflite::ops::builtin::Register_CONCATENATION());
    AddBuiltin(::tflite::BuiltinOperator_CONV_2D,
               ::tflite::ops::builtin::Register_CONV_2D());
    AddBuiltin(::tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
               ::tflite::ops::builtin::Register_DEPTHWISE_CONV_2D());
    AddBuiltin(::tflite::BuiltinOperator_RESHAPE,
               ::tflite::ops::builtin::Register_RESHAPE());
    AddBuiltin(::tflite::BuiltinOperator_LOGISTIC,
               ::tflite::ops::builtin::Register_LOGISTIC());
    AddBuiltin(::tflite::BuiltinOperator_ADD,
               ::tflite::ops::builtin::Register_ADD());
    AddCustom("TFLite_Detection_PostProcess",
              tflite::ops::custom::Register_DETECTION_POSTPROCESS());
  }

  MobileSsdQuantizedOpResolver(const MobileSsdQuantizedOpResolver& r) = delete;
};

class CreateFromOptionsTest : public tflite::testing::Test {};

TEST_F(CreateFromOptionsTest, SucceedsWithSelectiveOpResolver) {
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
  options->base_options.op_resolver =
      absl::make_unique<MobileSsdQuantizedOpResolver>();
  MP_ASSERT_OK(ObjectDetector::Create(std::move(options)));
}

// OpResolver missing the Detection_PostProcess op.
class MobileSsdQuantizedOpResolverMissingOps
    : public ::tflite::MutableOpResolver {
 public:
  MobileSsdQuantizedOpResolverMissingOps() {
    AddBuiltin(::tflite::BuiltinOperator_CONCATENATION,
               ::tflite::ops::builtin::Register_CONCATENATION());
    AddBuiltin(::tflite::BuiltinOperator_CONV_2D,
               ::tflite::ops::builtin::Register_CONV_2D());
    AddBuiltin(::tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
               ::tflite::ops::builtin::Register_DEPTHWISE_CONV_2D());
    AddBuiltin(::tflite::BuiltinOperator_RESHAPE,
               ::tflite::ops::builtin::Register_RESHAPE());
    AddBuiltin(::tflite::BuiltinOperator_LOGISTIC,
               ::tflite::ops::builtin::Register_LOGISTIC());
    AddBuiltin(::tflite::BuiltinOperator_ADD,
               ::tflite::ops::builtin::Register_ADD());
  }

  MobileSsdQuantizedOpResolverMissingOps(
      const MobileSsdQuantizedOpResolverMissingOps& r) = delete;
};

TEST_F(CreateFromOptionsTest, FailsWithSelectiveOpResolverMissingOps) {
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
  options->base_options.op_resolver =
      absl::make_unique<MobileSsdQuantizedOpResolverMissingOps>();
  auto object_detector = ObjectDetector::Create(std::move(options));
  // TODO: Make MediaPipe InferenceCalculator report the detailed.
  // interpreter errors (e.g., "Encountered unresolved custom op").
  EXPECT_EQ(object_detector.status().code(), absl::StatusCode::kInternal);
  EXPECT_THAT(object_detector.status().message(),
              HasSubstr("interpreter->AllocateTensors() == kTfLiteOk"));
}

TEST_F(CreateFromOptionsTest, FailsWithMissingModel) {
  auto options = std::make_unique<ObjectDetectorOptions>();
  absl::StatusOr<std::unique_ptr<ObjectDetector>> object_detector =
      ObjectDetector::Create(std::move(options));

  EXPECT_EQ(object_detector.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      object_detector.status().message(),
      HasSubstr("ExternalFile must specify at least one of 'file_content', "
                "'file_name', 'file_pointer_meta' or 'file_descriptor_meta'."));
  EXPECT_THAT(object_detector.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(CreateFromOptionsTest, FailsWithInvalidMaxResults) {
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
  options->max_results = 0;

  absl::StatusOr<std::unique_ptr<ObjectDetector>> object_detector =
      ObjectDetector::Create(std::move(options));

  EXPECT_EQ(object_detector.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(object_detector.status().message(),
              HasSubstr("Invalid `max_results` option"));
  EXPECT_THAT(object_detector.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(CreateFromOptionsTest, FailsWithCombinedAllowlistAndDenylist) {
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
  options->category_allowlist.push_back("foo");
  options->category_denylist.push_back("bar");
  absl::StatusOr<std::unique_ptr<ObjectDetector>> object_detector =
      ObjectDetector::Create(std::move(options));

  EXPECT_EQ(object_detector.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(object_detector.status().message(),
              HasSubstr("mutually exclusive options"));
  EXPECT_THAT(object_detector.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(CreateFromOptionsTest, FailsWithIllegalCallbackInImageOrVideoMode) {
  for (auto running_mode :
       {core::RunningMode::IMAGE, core::RunningMode::VIDEO}) {
    auto options = std::make_unique<ObjectDetectorOptions>();
    options->base_options.model_asset_path =
        JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
    options->running_mode = running_mode;
    options->result_callback =
        [](absl::StatusOr<ObjectDetectorResult> detections, const Image& image,
           int64_t timestamp_ms) {};
    absl::StatusOr<std::unique_ptr<ObjectDetector>> object_detector =
        ObjectDetector::Create(std::move(options));
    EXPECT_EQ(object_detector.status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_THAT(
        object_detector.status().message(),
        HasSubstr("a user-defined result callback shouldn't be provided"));
    EXPECT_THAT(object_detector.status().GetPayload(kMediaPipeTasksPayload),
                Optional(absl::Cord(absl::StrCat(
                    MediaPipeTasksStatus::kInvalidTaskGraphConfigError))));
  }
}

TEST_F(CreateFromOptionsTest, FailsWithMissingCallbackInLiveStreamMode) {
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  absl::StatusOr<std::unique_ptr<ObjectDetector>> object_detector =
      ObjectDetector::Create(std::move(options));

  EXPECT_EQ(object_detector.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(object_detector.status().message(),
              HasSubstr("a user-defined result callback must be provided"));
  EXPECT_THAT(object_detector.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kInvalidTaskGraphConfigError))));
}

TEST_F(CreateFromOptionsTest, InputTensorSpecsForMobileSsdModel) {
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ObjectDetector> object_detector,
                          ObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto image_tensor_specs,
                          object_detector->GetInputImageTensorSpecs());
  EXPECT_EQ(image_tensor_specs.image_width, 300);
  EXPECT_EQ(image_tensor_specs.image_height, 300);
  EXPECT_EQ(image_tensor_specs.color_space,
            tflite::ColorSpaceType::ColorSpaceType_RGB);
  EXPECT_EQ(image_tensor_specs.tensor_type, tflite::TensorType_UINT8);
}

TEST_F(CreateFromOptionsTest, InputTensorSpecsForEfficientDetModel) {
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kEfficientDetWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ObjectDetector> object_detector,
                          ObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto image_tensor_specs,
                          object_detector->GetInputImageTensorSpecs());
  EXPECT_EQ(image_tensor_specs.image_width, 320);
  EXPECT_EQ(image_tensor_specs.image_height, 320);
  EXPECT_EQ(image_tensor_specs.color_space,
            tflite::ColorSpaceType::ColorSpaceType_RGB);
  EXPECT_EQ(image_tensor_specs.tensor_type, tflite::TensorType_UINT8);
}

// TODO: Add NumThreadsTest back after having an
// "acceleration configuration" field in the ObjectDetectorOptions.

class ImageModeTest : public tflite::testing::Test {};

TEST_F(ImageModeTest, FailsWithCallingWrongMethod) {
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(JoinPath(
                                           "./", kTestDataDirectory,
                                           "cats_and_dogs_no_resizing.jpg")));
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ObjectDetector> object_detector,
                          ObjectDetector::Create(std::move(options)));
  auto results = object_detector->DetectForVideo(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the video mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));

  results = object_detector->DetectAsync(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the live stream mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));
  MP_ASSERT_OK(object_detector->Close());
}

TEST_F(ImageModeTest, Succeeds) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                       "cats_and_dogs.jpg")));
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->max_results = 4;
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ObjectDetector> object_detector,
                          ObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto results, object_detector->Detect(image));
  MP_ASSERT_OK(object_detector->Close());
  ExpectApproximatelyEqual(
      results,
      ConvertToDetectionResult(
          {ParseTextProtoOrDie<DetectionProto>(R"pb(
             label: "cat"
             score: 0.69921875
             location_data {
               format: BOUNDING_BOX
               bounding_box { xmin: 608 ymin: 164 width: 381 height: 432 }
             })pb"),
           ParseTextProtoOrDie<DetectionProto>(R"pb(
             label: "cat"
             score: 0.65625
             location_data {
               format: BOUNDING_BOX
               bounding_box { xmin: 57 ymin: 398 width: 386 height: 196 }
             })pb"),
           ParseTextProtoOrDie<DetectionProto>(R"pb(
             label: "cat"
             score: 0.51171875
             location_data {
               format: BOUNDING_BOX
               bounding_box { xmin: 256 ymin: 394 width: 173 height: 202 }
             })pb"),
           ParseTextProtoOrDie<DetectionProto>(R"pb(
             label: "cat"
             score: 0.48828125
             location_data {
               format: BOUNDING_BOX
               bounding_box { xmin: 360 ymin: 195 width: 330 height: 412 }
             })pb")}));
}

TEST_F(ImageModeTest, SucceedsEfficientDetModel) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                       "cats_and_dogs.jpg")));
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->max_results = 4;
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kEfficientDetWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ObjectDetector> object_detector,
                          ObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto results, object_detector->Detect(image));
  MP_ASSERT_OK(object_detector->Close());
  ExpectApproximatelyEqual(
      results,
      ConvertToDetectionResult(
          {ParseTextProtoOrDie<DetectionProto>(R"pb(
             label: "cat"
             score: 0.7578125
             location_data {
               format: BOUNDING_BOX
               bounding_box { xmin: 858 ymin: 408 width: 225 height: 187 }
             })pb"),
           ParseTextProtoOrDie<DetectionProto>(R"pb(
             label: "cat"
             score: 0.72265625
             location_data {
               format: BOUNDING_BOX
               bounding_box { xmin: 67 ymin: 401 width: 399 height: 192 }
             })pb"),
           ParseTextProtoOrDie<DetectionProto>(R"pb(
             label: "cat"
             score: 0.6289063
             location_data {
               format: BOUNDING_BOX
               bounding_box { xmin: 368 ymin: 210 width: 272 height: 385 }
             })pb"),
           ParseTextProtoOrDie<DetectionProto>(R"pb(
             label: "cat"
             score: 0.5859375
             location_data {
               format: BOUNDING_BOX
               bounding_box { xmin: 601 ymin: 166 width: 298 height: 437 }
             })pb")}));
}

TEST_F(ImageModeTest, SucceedsEfficientDetNoNmsModel) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                       "cats_and_dogs.jpg")));
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->max_results = 4;
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kEfficientDetWithoutNms);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ObjectDetector> object_detector,
                          ObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto results, object_detector->Detect(image));
  MP_ASSERT_OK(object_detector->Close());
  ExpectApproximatelyEqual(
      results,
      ConvertToDetectionResult(
          {ParseTextProtoOrDie<DetectionProto>(R"pb(
             label: "dog"
             score: 0.733542
             location_data {
               format: BOUNDING_BOX
               bounding_box { xmin: 636 ymin: 160 width: 282 height: 451 }
             })pb"),
           ParseTextProtoOrDie<DetectionProto>(R"pb(
             label: "cat"
             score: 0.699751
             location_data {
               format: BOUNDING_BOX
               bounding_box { xmin: 870 ymin: 411 width: 208 height: 187 }
             })pb"),
           ParseTextProtoOrDie<DetectionProto>(R"pb(
             label: "dog"
             score: 0.682425
             location_data {
               format: BOUNDING_BOX
               bounding_box { xmin: 386 ymin: 216 width: 256 height: 376 }
             })pb"),
           ParseTextProtoOrDie<DetectionProto>(R"pb(
             label: "cat"
             score: 0.646585
             location_data {
               format: BOUNDING_BOX
               bounding_box { xmin: 83 ymin: 399 width: 347 height: 198 }
             })pb")}));
}

TEST_F(ImageModeTest, SucceedsNoObjectDetected) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                       "cats_and_dogs.jpg")));
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->max_results = 4;
  options->score_threshold = 1.0f;
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kEfficientDetWithoutNms);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ObjectDetector> object_detector,
                          ObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto results, object_detector->Detect(image));
  MP_ASSERT_OK(object_detector->Close());
  EXPECT_THAT(results.detections, testing::IsEmpty());
}

TEST_F(ImageModeTest, SucceedsWithoutImageResizing) {
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(JoinPath(
                                           "./", kTestDataDirectory,
                                           "cats_and_dogs_no_resizing.jpg")));
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->max_results = 4;
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ObjectDetector> object_detector,
                          ObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto results, object_detector->Detect(image));
  MP_ASSERT_OK(object_detector->Close());
  ExpectApproximatelyEqual(
      results, ConvertToDetectionResult(
                   GenerateMobileSsdNoImageResizingFullExpectedResults()));
}

TEST_F(ImageModeTest, SucceedsWithScoreCalibration) {
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(JoinPath(
                                           "./", kTestDataDirectory,
                                           "cats_and_dogs_no_resizing.jpg")));
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->max_results = 1;
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithDummyScoreCalibration);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ObjectDetector> object_detector,
                          ObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto results, object_detector->Detect(image));
  MP_ASSERT_OK(object_detector->Close());
  ExpectApproximatelyEqual(
      results,
      ConvertToDetectionResult({ParseTextProtoOrDie<DetectionProto>(R"pb(
        label: "cat"
        score: 0.650467276
        location_data {
          format: BOUNDING_BOX
          bounding_box { xmin: 15 ymin: 197 width: 98 height: 99 }
        })pb")}));
}

TEST_F(ImageModeTest, SucceedsWithScoreThresholdOption) {
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(JoinPath(
                                           "./", kTestDataDirectory,
                                           "cats_and_dogs_no_resizing.jpg")));
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->score_threshold = 0.5;
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ObjectDetector> object_detector,
                          ObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto results, object_detector->Detect(image));
  MP_ASSERT_OK(object_detector->Close());
  std::vector<DetectionProto> full_expected_results =
      GenerateMobileSsdNoImageResizingFullExpectedResults();

  ExpectApproximatelyEqual(
      results, ConvertToDetectionResult(
                   {full_expected_results[0], full_expected_results[1],
                    full_expected_results[2], full_expected_results[3]}));
}

TEST_F(ImageModeTest, SucceedsWithMaxResultsOption) {
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(JoinPath(
                                           "./", kTestDataDirectory,
                                           "cats_and_dogs_no_resizing.jpg")));
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->max_results = 2;
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ObjectDetector> object_detector,
                          ObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto results, object_detector->Detect(image));
  MP_ASSERT_OK(object_detector->Close());
  std::vector<DetectionProto> full_expected_results =
      GenerateMobileSsdNoImageResizingFullExpectedResults();
  ExpectApproximatelyEqual(
      results, ConvertToDetectionResult(
                   {full_expected_results[0], full_expected_results[1]}));
}

TEST_F(ImageModeTest, SucceedsWithAllowlistOption) {
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(JoinPath(
                                           "./", kTestDataDirectory,
                                           "cats_and_dogs_no_resizing.jpg")));
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->max_results = 1;
  options->category_allowlist.push_back("dog");
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ObjectDetector> object_detector,
                          ObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto results, object_detector->Detect(image));
  MP_ASSERT_OK(object_detector->Close());
  std::vector<DetectionProto> full_expected_results =
      GenerateMobileSsdNoImageResizingFullExpectedResults();
  ExpectApproximatelyEqual(
      results, ConvertToDetectionResult({full_expected_results[3]}));
}

TEST_F(ImageModeTest, SucceedsWithDenylistOption) {
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(JoinPath(
                                           "./", kTestDataDirectory,
                                           "cats_and_dogs_no_resizing.jpg")));
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->max_results = 1;
  options->category_denylist.push_back("cat");
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ObjectDetector> object_detector,
                          ObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto results, object_detector->Detect(image));
  MP_ASSERT_OK(object_detector->Close());
  std::vector<DetectionProto> full_expected_results =
      GenerateMobileSsdNoImageResizingFullExpectedResults();
  ExpectApproximatelyEqual(
      results, ConvertToDetectionResult({full_expected_results[3]}));
}

TEST_F(ImageModeTest, SucceedsWithRotation) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image, DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                "cats_and_dogs_rotated.jpg")));
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->max_results = 1;
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ObjectDetector> object_detector,
                          ObjectDetector::Create(std::move(options)));
  ImageProcessingOptions image_processing_options;
  image_processing_options.rotation_degrees = -90;
  MP_ASSERT_OK_AND_ASSIGN(
      auto results, object_detector->Detect(image, image_processing_options));
  MP_ASSERT_OK(object_detector->Close());
  ExpectApproximatelyEqual(
      results,
      ConvertToDetectionResult({ParseTextProtoOrDie<DetectionProto>(R"pb(
        label: "cat"
        score: 0.69921875
        location_data {
          format: BOUNDING_BOX
          bounding_box { xmin: 0 ymin: 608 width: 439 height: 387 }
        })pb")}));
}

TEST_F(ImageModeTest, FailsWithRegionOfInterest) {
  MP_ASSERT_OK_AND_ASSIGN(Image image,
                          DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                                       "cats_and_dogs.jpg")));
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->max_results = 1;
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ObjectDetector> object_detector,
                          ObjectDetector::Create(std::move(options)));
  RectF roi{/*left=*/0.1, /*top=*/0, /*right=*/0.9, /*bottom=*/1};
  ImageProcessingOptions image_processing_options{roi, /*rotation_degrees=*/0};

  auto results = object_detector->Detect(image, image_processing_options);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("This task doesn't support region-of-interest"));
  EXPECT_THAT(
      results.status().GetPayload(kMediaPipeTasksPayload),
      Optional(absl::Cord(absl::StrCat(
          MediaPipeTasksStatus::kImageProcessingInvalidArgumentError))));
}

class VideoModeTest : public tflite::testing::Test {};

TEST_F(VideoModeTest, FailsWithCallingWrongMethod) {
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(JoinPath(
                                           "./", kTestDataDirectory,
                                           "cats_and_dogs_no_resizing.jpg")));
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
  options->running_mode = core::RunningMode::VIDEO;

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ObjectDetector> object_detector,
                          ObjectDetector::Create(std::move(options)));
  auto results = object_detector->Detect(image);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the image mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));

  results = object_detector->DetectAsync(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the live stream mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));
  MP_ASSERT_OK(object_detector->Close());
}

TEST_F(VideoModeTest, Succeeds) {
  int iterations = 100;
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(JoinPath(
                                           "./", kTestDataDirectory,
                                           "cats_and_dogs_no_resizing.jpg")));
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->max_results = 2;
  options->running_mode = core::RunningMode::VIDEO;
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ObjectDetector> object_detector,
                          ObjectDetector::Create(std::move(options)));
  for (int i = 0; i < iterations; ++i) {
    MP_ASSERT_OK_AND_ASSIGN(auto results,
                            object_detector->DetectForVideo(image, i));
    std::vector<DetectionProto> full_expected_results =
        GenerateMobileSsdNoImageResizingFullExpectedResults();
    ExpectApproximatelyEqual(
        results, ConvertToDetectionResult(
                     {full_expected_results[0], full_expected_results[1]}));
  }
  MP_ASSERT_OK(object_detector->Close());
}

class LiveStreamModeTest : public tflite::testing::Test {};

TEST_F(LiveStreamModeTest, FailsWithCallingWrongMethod) {
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(JoinPath(
                                           "./", kTestDataDirectory,
                                           "cats_and_dogs_no_resizing.jpg")));
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->result_callback = [](absl::StatusOr<ObjectDetectorResult> detections,
                                const Image& image, int64_t timestamp_ms) {};

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ObjectDetector> object_detector,
                          ObjectDetector::Create(std::move(options)));
  auto results = object_detector->Detect(image);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the image mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));

  results = object_detector->DetectForVideo(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the video mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));
  MP_ASSERT_OK(object_detector->Close());
}

TEST_F(LiveStreamModeTest, FailsWithOutOfOrderInputTimestamps) {
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(JoinPath(
                                           "./", kTestDataDirectory,
                                           "cats_and_dogs_no_resizing.jpg")));
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
  options->result_callback = [](absl::StatusOr<ObjectDetectorResult> detections,
                                const Image& image, int64_t timestamp_ms) {};
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ObjectDetector> object_detector,
                          ObjectDetector::Create(std::move(options)));
  MP_ASSERT_OK(object_detector->DetectAsync(image, 1));

  auto status = object_detector->DetectAsync(image, 0);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("timestamp must be monotonically increasing"));
  EXPECT_THAT(status.GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInvalidTimestampError))));
  MP_ASSERT_OK(object_detector->DetectAsync(image, 2));
  MP_ASSERT_OK(object_detector->Close());
}

TEST_F(LiveStreamModeTest, Succeeds) {
  int iterations = 100;
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(JoinPath(
                                           "./", kTestDataDirectory,
                                           "cats_and_dogs_no_resizing.jpg")));
  auto options = std::make_unique<ObjectDetectorOptions>();
  options->max_results = 2;
  options->running_mode = core::RunningMode::LIVE_STREAM;
  std::vector<ObjectDetectorResult> detection_results;
  std::vector<std::pair<int, int>> image_sizes;
  std::vector<int64_t> timestamps;
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileSsdWithMetadata);
  options->result_callback =
      [&detection_results, &image_sizes, &timestamps](
          absl::StatusOr<ObjectDetectorResult> detections, const Image& image,
          int64_t timestamp_ms) {
        MP_ASSERT_OK(detections.status());
        detection_results.push_back(std::move(detections).value());
        image_sizes.push_back({image.width(), image.height()});
        timestamps.push_back(timestamp_ms);
      };
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ObjectDetector> object_detector,
                          ObjectDetector::Create(std::move(options)));
  for (int i = 0; i < iterations; ++i) {
    MP_ASSERT_OK(object_detector->DetectAsync(image, i));
  }
  MP_ASSERT_OK(object_detector->Close());
  // Due to the flow limiter, the total of outputs will be smaller than the
  // number of iterations.
  ASSERT_LE(detection_results.size(), iterations);
  ASSERT_GT(detection_results.size(), 0);
  std::vector<DetectionProto> full_expected_results =
      GenerateMobileSsdNoImageResizingFullExpectedResults();
  for (const auto& detection_result : detection_results) {
    ExpectApproximatelyEqual(
        detection_result, ConvertToDetectionResult({full_expected_results[0],
                                                    full_expected_results[1]}));
  }
  for (const auto& image_size : image_sizes) {
    EXPECT_EQ(image_size.first, image.width());
    EXPECT_EQ(image_size.second, image.height());
  }
  int64_t timestamp_ms = -1;
  for (const auto& timestamp : timestamps) {
    EXPECT_GT(timestamp, timestamp_ms);
    timestamp_ms = timestamp;
  }
}

}  // namespace
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
