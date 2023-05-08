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

#include "mediapipe/tasks/cc/vision/image_segmenter/image_segmenter.h"

#include <array>
#include <cstdint>
#include <memory>

#include "absl/flags/flag.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/test_util.h"
#include "mediapipe/tasks/cc/components/containers/rect.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/calculators/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/image_segmenter_result.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/image_segmenter_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/test_util.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_segmenter {
namespace {

using ::mediapipe::Image;
using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::components::containers::RectF;
using ::mediapipe::tasks::vision::core::ImageProcessingOptions;
using ::testing::HasSubstr;
using ::testing::Optional;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kDeeplabV3WithMetadata[] = "deeplabv3.tflite";

constexpr char kSelfie128x128WithMetadata[] = "selfie_segm_128_128_3.tflite";

constexpr char kSelfie144x256WithMetadata[] = "selfie_segm_144_256_3.tflite";

constexpr char kSelfieSegmentation[] = "selfie_segmentation.tflite";

constexpr char kSelfieSegmentationLandscape[] =
    "selfie_segmentation_landscape.tflite";

constexpr char kHairSegmentationWithMetadata[] = "hair_segmentation.tflite";

constexpr float kGoldenMaskSimilarity = 0.98;

// Magnification factor used when creating the golden category masks to make
// them more human-friendly. Each pixel in the golden masks has its value
// multiplied by this factor, i.e. a value of 10 means class index 1, a value of
// 20 means class index 2, etc.
constexpr int kGoldenMaskMagnificationFactor = 10;

constexpr std::array<absl::string_view, 21> kDeeplabLabelNames = {
    "background", "aeroplane",    "bicycle", "bird",  "boat",
    "bottle",     "bus",          "car",     "cat",   "chair",
    "cow",        "dining table", "dog",     "horse", "motorbike",
    "person",     "potted plant", "sheep",   "sofa",  "train",
    "tv"};

// Intentionally converting output into CV_8UC1 and then again into CV_32FC1
// as expected outputs are stored in CV_8UC1, so this conversion allows to do
// fair comparison.
cv::Mat PostProcessResultMask(const cv::Mat& mask) {
  cv::Mat mask_float;
  mask.convertTo(mask_float, CV_8UC1, 255);
  mask_float.convertTo(mask_float, CV_32FC1, 1 / 255.f);
  return mask_float;
}

Image GetSRGBImage(const std::string& image_path) {
  cv::Mat image_mat = cv::imread(image_path);
  cv::cvtColor(image_mat, image_mat, cv::COLOR_BGR2RGB);
  mediapipe::ImageFrame image_frame(
      mediapipe::ImageFormat::SRGB, image_mat.cols, image_mat.rows,
      image_mat.step, image_mat.data, [image_mat](uint8_t[]) {});
  Image image(std::make_shared<mediapipe::ImageFrame>(std::move(image_frame)));
  return image;
}

Image GetSRGBAImage(const std::string& image_path) {
  cv::Mat image_mat = cv::imread(image_path);
  cv::cvtColor(image_mat, image_mat, cv::COLOR_BGR2RGBA);
  std::vector<cv::Mat> channels(4);
  cv::split(image_mat, channels);
  channels[3].setTo(0);
  cv::merge(channels.data(), 4, image_mat);
  mediapipe::ImageFrame image_frame(
      mediapipe::ImageFormat::SRGBA, image_mat.cols, image_mat.rows,
      image_mat.step, image_mat.data, [image_mat](uint8_t[]) {});
  Image image(std::make_shared<mediapipe::ImageFrame>(std::move(image_frame)));
  return image;
}

double CalculateSum(const cv::Mat& m) {
  double sum = 0.0;
  cv::Scalar s = cv::sum(m);
  for (int i = 0; i < m.channels(); ++i) {
    sum += s.val[i];
  }
  return sum;
}

double CalculateSoftIOU(const cv::Mat& m1, const cv::Mat& m2) {
  cv::Mat intersection;
  cv::multiply(m1, m2, intersection);
  double intersection_value = CalculateSum(intersection);
  double union_value =
      CalculateSum(m1.mul(m1)) + CalculateSum(m2.mul(m2)) - intersection_value;
  return union_value > 0.0 ? intersection_value / union_value : 0.0;
}

MATCHER_P2(SimilarToFloatMask, expected_mask, similarity_threshold, "") {
  cv::Mat actual_mask = PostProcessResultMask(arg);
  return arg.rows == expected_mask.rows && arg.cols == expected_mask.cols &&
         CalculateSoftIOU(arg, expected_mask) > similarity_threshold;
}

MATCHER_P3(SimilarToUint8Mask, expected_mask, similarity_threshold,
           magnification_factor, "") {
  if (arg.rows != expected_mask.rows || arg.cols != expected_mask.cols) {
    return false;
  }
  int consistent_pixels = 0;
  const int num_pixels = expected_mask.rows * expected_mask.cols;
  for (int i = 0; i < num_pixels; ++i) {
    consistent_pixels +=
        (arg.data[i] * magnification_factor == expected_mask.data[i]);
  }
  return static_cast<float>(consistent_pixels) / num_pixels >=
         similarity_threshold;
}

class DeepLabOpResolver : public ::tflite::MutableOpResolver {
 public:
  DeepLabOpResolver() {
    AddBuiltin(::tflite::BuiltinOperator_ADD,
               ::tflite::ops::builtin::Register_ADD());
    AddBuiltin(::tflite::BuiltinOperator_AVERAGE_POOL_2D,
               ::tflite::ops::builtin::Register_AVERAGE_POOL_2D());
    AddBuiltin(::tflite::BuiltinOperator_CONCATENATION,
               ::tflite::ops::builtin::Register_CONCATENATION());
    AddBuiltin(::tflite::BuiltinOperator_CONV_2D,
               ::tflite::ops::builtin::Register_CONV_2D());
    // DeepLab uses different versions of DEPTHWISE_CONV_2D.
    AddBuiltin(::tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
               ::tflite::ops::builtin::Register_DEPTHWISE_CONV_2D(),
               /*min_version=*/1, /*max_version=*/2);
    AddBuiltin(::tflite::BuiltinOperator_RESIZE_BILINEAR,
               ::tflite::ops::builtin::Register_RESIZE_BILINEAR());
  }

  DeepLabOpResolver(const DeepLabOpResolver& r) = delete;
};

class CreateFromOptionsTest : public tflite::testing::Test {};

class DeepLabOpResolverMissingOps : public ::tflite::MutableOpResolver {
 public:
  DeepLabOpResolverMissingOps() {
    AddBuiltin(::tflite::BuiltinOperator_ADD,
               ::tflite::ops::builtin::Register_ADD());
  }

  DeepLabOpResolverMissingOps(const DeepLabOpResolverMissingOps& r) = delete;
};

TEST_F(CreateFromOptionsTest, SucceedsWithSelectiveOpResolver) {
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kDeeplabV3WithMetadata);
  options->base_options.op_resolver = absl::make_unique<DeepLabOpResolver>();
  MP_ASSERT_OK(ImageSegmenter::Create(std::move(options)));
}

TEST_F(CreateFromOptionsTest, FailsWithSelectiveOpResolverMissingOps) {
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kDeeplabV3WithMetadata);
  options->base_options.op_resolver =
      absl::make_unique<DeepLabOpResolverMissingOps>();
  auto segmenter_or = ImageSegmenter::Create(std::move(options));
  // TODO: Make MediaPipe InferenceCalculator report the detailed
  // interpreter errors (e.g., "Encountered unresolved custom op").
  EXPECT_EQ(segmenter_or.status().code(), absl::StatusCode::kInternal);
  EXPECT_THAT(
      segmenter_or.status().message(),
      testing::HasSubstr("interpreter_builder(&interpreter) == kTfLiteOk"));
}

TEST_F(CreateFromOptionsTest, FailsWithMissingModel) {
  absl::StatusOr<std::unique_ptr<ImageSegmenter>> segmenter_or =
      ImageSegmenter::Create(std::make_unique<ImageSegmenterOptions>());

  EXPECT_EQ(segmenter_or.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      segmenter_or.status().message(),
      HasSubstr("ExternalFile must specify at least one of 'file_content', "
                "'file_name', 'file_pointer_meta' or 'file_descriptor_meta'."));
  EXPECT_THAT(segmenter_or.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(CreateFromOptionsTest, FailsWithInputDimsTwoModel) {
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, "dense.tflite");
  absl::StatusOr<std::unique_ptr<ImageSegmenter>> result =
      ImageSegmenter::Create(std::move(options));
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(result.status().message(),
              HasSubstr("Expect segmentation model has input image tensor to "
                        "be 4 dims."));
}

TEST_F(CreateFromOptionsTest, FailsWithInputChannelOneModel) {
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, "conv2d_input_channel_1.tflite");
  absl::StatusOr<std::unique_ptr<ImageSegmenter>> result =
      ImageSegmenter::Create(std::move(options));
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(result.status().message(),
              HasSubstr("Expect segmentation model has input image tensor with "
                        "channels = 3 or 4."));
}

TEST(GetLabelsTest, SucceedsWithLabelsInModel) {
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kDeeplabV3WithMetadata);

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageSegmenter> segmenter,
                          ImageSegmenter::Create(std::move(options)));
  const auto& labels = segmenter->GetLabels();
  ASSERT_FALSE(labels.empty());
  ASSERT_EQ(labels.size(), kDeeplabLabelNames.size());
  for (int i = 0; i < labels.size(); ++i) {
    EXPECT_EQ(labels[i], kDeeplabLabelNames[i]);
  }
}

class ImageModeTest : public tflite::testing::Test {};

TEST_F(ImageModeTest, SucceedsWithCategoryMask) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                   "segmentation_input_rotation0.jpg")));
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kDeeplabV3WithMetadata);
  options->output_confidence_masks = false;
  options->output_category_mask = true;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageSegmenter> segmenter,
                          ImageSegmenter::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto result, segmenter->Segment(image));
  EXPECT_TRUE(result.category_mask.has_value());

  cv::Mat actual_mask = mediapipe::formats::MatView(
      result.category_mask->GetImageFrameSharedPtr().get());

  cv::Mat expected_mask = cv::imread(
      JoinPath("./", kTestDataDirectory, "segmentation_golden_rotation0.png"),
      cv::IMREAD_GRAYSCALE);
  EXPECT_THAT(actual_mask,
              SimilarToUint8Mask(expected_mask, kGoldenMaskSimilarity,
                                 kGoldenMaskMagnificationFactor));
}

TEST_F(ImageModeTest, SucceedsWithConfidenceMask) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "cat.jpg")));
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kDeeplabV3WithMetadata);

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageSegmenter> segmenter,
                          ImageSegmenter::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto result, segmenter->Segment(image));
  EXPECT_EQ(result.confidence_masks->size(), 21);

  cv::Mat expected_mask = cv::imread(
      JoinPath("./", kTestDataDirectory, "cat_mask.jpg"), cv::IMREAD_GRAYSCALE);
  cv::Mat expected_mask_float;
  expected_mask.convertTo(expected_mask_float, CV_32FC1, 1 / 255.f);

  // Cat category index 8.
  cv::Mat cat_mask = mediapipe::formats::MatView(
      result.confidence_masks->at(8).GetImageFrameSharedPtr().get());
  EXPECT_THAT(cat_mask,
              SimilarToFloatMask(expected_mask_float, kGoldenMaskSimilarity));
}

// TODO: fix this unit test after image segmenter handled post
// processing correctly with rotated image.
TEST_F(ImageModeTest, DISABLED_SucceedsWithRotation) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "cat.jpg")));
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kDeeplabV3WithMetadata);

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageSegmenter> segmenter,
                          ImageSegmenter::Create(std::move(options)));
  ImageProcessingOptions image_processing_options;
  image_processing_options.rotation_degrees = -90;
  MP_ASSERT_OK_AND_ASSIGN(auto result,
                          segmenter->Segment(image, image_processing_options));
  EXPECT_EQ(result.confidence_masks->size(), 21);

  cv::Mat expected_mask =
      cv::imread(JoinPath("./", kTestDataDirectory, "cat_rotated_mask.jpg"),
                 cv::IMREAD_GRAYSCALE);
  cv::Mat expected_mask_float;
  expected_mask.convertTo(expected_mask_float, CV_32FC1, 1 / 255.f);

  // Cat category index 8.
  cv::Mat cat_mask = mediapipe::formats::MatView(
      result.confidence_masks->at(8).GetImageFrameSharedPtr().get());
  EXPECT_THAT(cat_mask,
              SimilarToFloatMask(expected_mask_float, kGoldenMaskSimilarity));
}

TEST_F(ImageModeTest, FailsWithRegionOfInterest) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, "cat.jpg")));
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kDeeplabV3WithMetadata);

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageSegmenter> segmenter,
                          ImageSegmenter::Create(std::move(options)));
  RectF roi{/*left=*/0.1, /*top=*/0, /*right=*/0.9, /*bottom=*/1};
  ImageProcessingOptions image_processing_options{roi, /*rotation_degrees=*/0};

  auto results = segmenter->Segment(image, image_processing_options);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("This task doesn't support region-of-interest"));
  EXPECT_THAT(
      results.status().GetPayload(kMediaPipeTasksPayload),
      Optional(absl::Cord(absl::StrCat(
          MediaPipeTasksStatus::kImageProcessingInvalidArgumentError))));
}

TEST_F(ImageModeTest, SucceedsSelfie128x128Segmentation) {
  Image image =
      GetSRGBImage(JoinPath("./", kTestDataDirectory, "mozart_square.jpg"));
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kSelfie128x128WithMetadata);

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageSegmenter> segmenter,
                          ImageSegmenter::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto result, segmenter->Segment(image));
  EXPECT_EQ(result.confidence_masks->size(), 2);

  cv::Mat expected_mask =
      cv::imread(JoinPath("./", kTestDataDirectory,
                          "selfie_segm_128_128_3_expected_mask.jpg"),
                 cv::IMREAD_GRAYSCALE);
  cv::Mat expected_mask_float;
  expected_mask.convertTo(expected_mask_float, CV_32FC1, 1 / 255.f);

  // Selfie category index 1.
  cv::Mat selfie_mask = mediapipe::formats::MatView(
      result.confidence_masks->at(1).GetImageFrameSharedPtr().get());
  EXPECT_THAT(selfie_mask,
              SimilarToFloatMask(expected_mask_float, kGoldenMaskSimilarity));
}

TEST_F(ImageModeTest, SucceedsSelfie144x256Segmentations) {
  Image image =
      GetSRGBImage(JoinPath("./", kTestDataDirectory, "mozart_square.jpg"));
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kSelfie144x256WithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageSegmenter> segmenter,
                          ImageSegmenter::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto result, segmenter->Segment(image));
  EXPECT_EQ(result.confidence_masks->size(), 1);

  cv::Mat expected_mask =
      cv::imread(JoinPath("./", kTestDataDirectory,
                          "selfie_segm_144_256_3_expected_mask.jpg"),
                 cv::IMREAD_GRAYSCALE);
  cv::Mat expected_mask_float;
  expected_mask.convertTo(expected_mask_float, CV_32FC1, 1 / 255.f);

  cv::Mat selfie_mask = mediapipe::formats::MatView(
      result.confidence_masks->at(0).GetImageFrameSharedPtr().get());
  EXPECT_THAT(selfie_mask,
              SimilarToFloatMask(expected_mask_float, kGoldenMaskSimilarity));
}

TEST_F(ImageModeTest, SucceedsSelfieSegmentationSingleLabel) {
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kSelfieSegmentation);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageSegmenter> segmenter,
                          ImageSegmenter::Create(std::move(options)));
  ASSERT_EQ(segmenter->GetLabels().size(), 1);
  EXPECT_EQ(segmenter->GetLabels()[0], "selfie");
  MP_ASSERT_OK(segmenter->Close());
}

TEST_F(ImageModeTest, SucceedsSelfieSegmentationLandscapeSingleLabel) {
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kSelfieSegmentationLandscape);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageSegmenter> segmenter,
                          ImageSegmenter::Create(std::move(options)));
  ASSERT_EQ(segmenter->GetLabels().size(), 1);
  EXPECT_EQ(segmenter->GetLabels()[0], "selfie");
  MP_ASSERT_OK(segmenter->Close());
}

TEST_F(ImageModeTest, SucceedsPortraitSelfieSegmentationConfidenceMask) {
  Image image =
      GetSRGBImage(JoinPath("./", kTestDataDirectory, "portrait.jpg"));
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kSelfieSegmentation);

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageSegmenter> segmenter,
                          ImageSegmenter::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto result, segmenter->Segment(image));
  EXPECT_EQ(result.confidence_masks->size(), 1);
  MP_ASSERT_OK(segmenter->Close());

  cv::Mat expected_mask = cv::imread(
      JoinPath("./", kTestDataDirectory,
               "portrait_selfie_segmentation_expected_confidence_mask.jpg"),
      cv::IMREAD_GRAYSCALE);
  cv::Mat expected_mask_float;
  expected_mask.convertTo(expected_mask_float, CV_32FC1, 1 / 255.f);

  cv::Mat selfie_mask = mediapipe::formats::MatView(
      result.confidence_masks->at(0).GetImageFrameSharedPtr().get());
  EXPECT_THAT(selfie_mask,
              SimilarToFloatMask(expected_mask_float, kGoldenMaskSimilarity));
}

TEST_F(ImageModeTest, SucceedsPortraitSelfieSegmentationCategoryMask) {
  Image image =
      GetSRGBImage(JoinPath("./", kTestDataDirectory, "portrait.jpg"));
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kSelfieSegmentation);
  options->output_category_mask = true;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageSegmenter> segmenter,
                          ImageSegmenter::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto result, segmenter->Segment(image));
  EXPECT_TRUE(result.category_mask.has_value());
  MP_ASSERT_OK(segmenter->Close());

  MP_EXPECT_OK(
      SavePngTestOutput(*result.category_mask->GetImageFrameSharedPtr(),
                        "portrait_selfie_segmentation_expected_category_mask"));
  cv::Mat selfie_mask = mediapipe::formats::MatView(
      result.category_mask->GetImageFrameSharedPtr().get());
  cv::Mat expected_mask = cv::imread(
      JoinPath("./", kTestDataDirectory,
               "portrait_selfie_segmentation_expected_category_mask.jpg"),
      cv::IMREAD_GRAYSCALE);
  EXPECT_THAT(selfie_mask,
              SimilarToUint8Mask(expected_mask, kGoldenMaskSimilarity, 1));
}

TEST_F(ImageModeTest, SucceedsPortraitSelfieSegmentationLandscapeCategoryMask) {
  Image image =
      GetSRGBImage(JoinPath("./", kTestDataDirectory, "portrait.jpg"));
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kSelfieSegmentationLandscape);
  options->output_category_mask = true;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageSegmenter> segmenter,
                          ImageSegmenter::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto result, segmenter->Segment(image));
  EXPECT_TRUE(result.category_mask.has_value());
  MP_ASSERT_OK(segmenter->Close());

  MP_EXPECT_OK(SavePngTestOutput(
      *result.category_mask->GetImageFrameSharedPtr(),
      "portrait_selfie_segmentation_landscape_expected_category_mask"));
  cv::Mat selfie_mask = mediapipe::formats::MatView(
      result.category_mask->GetImageFrameSharedPtr().get());
  cv::Mat expected_mask = cv::imread(
      JoinPath(
          "./", kTestDataDirectory,
          "portrait_selfie_segmentation_landscape_expected_category_mask.jpg"),
      cv::IMREAD_GRAYSCALE);
  EXPECT_THAT(selfie_mask,
              SimilarToUint8Mask(expected_mask, kGoldenMaskSimilarity, 1));
}

TEST_F(ImageModeTest, SucceedsHairSegmentation) {
  Image image =
      GetSRGBAImage(JoinPath("./", kTestDataDirectory, "portrait.jpg"));
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kHairSegmentationWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageSegmenter> segmenter,
                          ImageSegmenter::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto result, segmenter->Segment(image));
  EXPECT_EQ(result.confidence_masks->size(), 2);

  cv::Mat hair_mask = mediapipe::formats::MatView(
      result.confidence_masks->at(1).GetImageFrameSharedPtr().get());
  MP_ASSERT_OK(segmenter->Close());
  cv::Mat expected_mask = cv::imread(
      JoinPath("./", kTestDataDirectory, "portrait_hair_expected_mask.jpg"),
      cv::IMREAD_GRAYSCALE);
  cv::Mat expected_mask_float;
  expected_mask.convertTo(expected_mask_float, CV_32FC1, 1 / 255.f);
  EXPECT_THAT(hair_mask,
              SimilarToFloatMask(expected_mask_float, kGoldenMaskSimilarity));
}

class VideoModeTest : public tflite::testing::Test {};

TEST_F(VideoModeTest, FailsWithCallingWrongMethod) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                   "segmentation_input_rotation0.jpg")));
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kDeeplabV3WithMetadata);
  options->running_mode = core::RunningMode::VIDEO;

  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageSegmenter> segmenter,
                          ImageSegmenter::Create(std::move(options)));
  auto results = segmenter->Segment(image);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the image mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));

  results = segmenter->SegmentAsync(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the live stream mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));
  MP_ASSERT_OK(segmenter->Close());
}

TEST_F(VideoModeTest, Succeeds) {
  constexpr int iterations = 100;
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                   "segmentation_input_rotation0.jpg")));
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kDeeplabV3WithMetadata);
  options->output_category_mask = true;
  options->running_mode = core::RunningMode::VIDEO;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageSegmenter> segmenter,
                          ImageSegmenter::Create(std::move(options)));
  cv::Mat expected_mask = cv::imread(
      JoinPath("./", kTestDataDirectory, "segmentation_golden_rotation0.png"),
      cv::IMREAD_GRAYSCALE);
  for (int i = 0; i < iterations; ++i) {
    MP_ASSERT_OK_AND_ASSIGN(auto result, segmenter->SegmentForVideo(image, i));
    EXPECT_TRUE(result.category_mask.has_value());
    cv::Mat actual_mask = mediapipe::formats::MatView(
        result.category_mask->GetImageFrameSharedPtr().get());
    EXPECT_THAT(actual_mask,
                SimilarToUint8Mask(expected_mask, kGoldenMaskSimilarity,
                                   kGoldenMaskMagnificationFactor));
  }
  MP_ASSERT_OK(segmenter->Close());
}

class LiveStreamModeTest : public tflite::testing::Test {};

TEST_F(LiveStreamModeTest, FailsWithCallingWrongMethod) {
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(JoinPath(
                                           "./", kTestDataDirectory,
                                           "cats_and_dogs_no_resizing.jpg")));
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kDeeplabV3WithMetadata);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->result_callback =
      [](absl::StatusOr<ImageSegmenterResult> segmented_masks,
         const Image& image, int64_t timestamp_ms) {};
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageSegmenter> segmenter,
                          ImageSegmenter::Create(std::move(options)));

  auto results = segmenter->Segment(image);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the image mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));

  results = segmenter->SegmentForVideo(image, 0);
  EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(results.status().message(),
              HasSubstr("not initialized with the video mode"));
  EXPECT_THAT(results.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerApiCalledInWrongModeError))));
  MP_ASSERT_OK(segmenter->Close());
}

TEST_F(LiveStreamModeTest, FailsWithOutOfOrderInputTimestamps) {
  MP_ASSERT_OK_AND_ASSIGN(Image image, DecodeImageFromFile(JoinPath(
                                           "./", kTestDataDirectory,
                                           "cats_and_dogs_no_resizing.jpg")));
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kDeeplabV3WithMetadata);
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->result_callback = [](absl::StatusOr<ImageSegmenterResult> result,
                                const Image& image, int64_t timestamp_ms) {};
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageSegmenter> segmenter,
                          ImageSegmenter::Create(std::move(options)));
  MP_ASSERT_OK(segmenter->SegmentAsync(image, 1));

  auto status = segmenter->SegmentAsync(image, 0);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("timestamp must be monotonically increasing"));
  EXPECT_THAT(status.GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInvalidTimestampError))));
  MP_ASSERT_OK(segmenter->SegmentAsync(image, 2));
  MP_ASSERT_OK(segmenter->Close());
}

TEST_F(LiveStreamModeTest, Succeeds) {
  constexpr int iterations = 100;
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory,
                                   "segmentation_input_rotation0.jpg")));
  std::vector<Image> segmented_masks_results;
  std::vector<std::pair<int, int>> image_sizes;
  std::vector<int64_t> timestamps;
  auto options = std::make_unique<ImageSegmenterOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kDeeplabV3WithMetadata);
  options->output_category_mask = true;
  options->running_mode = core::RunningMode::LIVE_STREAM;
  options->result_callback = [&segmented_masks_results, &image_sizes,
                              &timestamps](
                                 absl::StatusOr<ImageSegmenterResult> result,
                                 const Image& image, int64_t timestamp_ms) {
    MP_ASSERT_OK(result.status());
    segmented_masks_results.push_back(std::move(*result->category_mask));
    image_sizes.push_back({image.width(), image.height()});
    timestamps.push_back(timestamp_ms);
  };
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageSegmenter> segmenter,
                          ImageSegmenter::Create(std::move(options)));
  for (int i = 0; i < iterations; ++i) {
    MP_ASSERT_OK(segmenter->SegmentAsync(image, i));
  }
  MP_ASSERT_OK(segmenter->Close());
  // Due to the flow limiter, the total of outputs will be smaller than the
  // number of iterations.
  ASSERT_LE(segmented_masks_results.size(), iterations);
  ASSERT_GT(segmented_masks_results.size(), 0);
  cv::Mat expected_mask = cv::imread(
      JoinPath("./", kTestDataDirectory, "segmentation_golden_rotation0.png"),
      cv::IMREAD_GRAYSCALE);
  for (const auto& category_mask : segmented_masks_results) {
    cv::Mat actual_mask = mediapipe::formats::MatView(
        category_mask.GetImageFrameSharedPtr().get());
    EXPECT_THAT(actual_mask,
                SimilarToUint8Mask(expected_mask, kGoldenMaskSimilarity,
                                   kGoldenMaskMagnificationFactor));
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
}  // namespace image_segmenter
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
