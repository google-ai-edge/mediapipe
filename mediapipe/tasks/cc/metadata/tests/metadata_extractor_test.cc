/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/common.h"

namespace mediapipe {
namespace tasks {
namespace metadata {
namespace {

using ::testing::Optional;

constexpr char kTestDataDirectory[] = "mediapipe/tasks/testdata/metadata";
constexpr char kEnLocale[] = "en";
constexpr char kFrLocale[] = "fr";
constexpr char kEnLabels[] = "0-labels-en.txt";
constexpr char kMobileIcaWithoutTfLiteMetadata[] =
    "mobile_ica_8bit-without-model-metadata.tflite";
constexpr char kMobileIcaWithTfLiteMetadata[] =
    "mobile_ica_8bit-with-metadata.tflite";

constexpr char kMobileIcaWithUnsupportedMetadataVersion[] =
    "mobile_ica_8bit-with-unsupported-metadata-version.tflite";
constexpr char kMobileIcaWithMetadataContainingNoName[] =
    "mobile_object_classifier_v0_2_3-metadata-no-name.tflite";
constexpr char kMobileNetWithNoMetadata[] =
    "mobilenet_v1_0.25_224_1_default_1.tflite";
// Text file not in FlatBuffer format.
constexpr char kRandomTextFile[] = "external_file";

absl::StatusOr<std::unique_ptr<ModelMetadataExtractor>> CreateMetadataExtractor(
    std::string model_name, std::string* file_contents) {
  MP_RETURN_IF_ERROR(file::GetContents(
      file::JoinPath("./", kTestDataDirectory, model_name), file_contents));
  return ModelMetadataExtractor::CreateFromModelBuffer(file_contents->data(),
                                                       file_contents->length());
}

TEST(ModelMetadataExtractorTest, CreateFailsWithInvalidFlatBuffer) {
  std::string buffer;
  absl::StatusOr<std::unique_ptr<ModelMetadataExtractor>> extractor =
      CreateMetadataExtractor(kRandomTextFile, &buffer);

  EXPECT_THAT(extractor.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(extractor.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kInvalidFlatBufferError))));
}

TEST(ModelMetadataExtractorTest, CreateFailsWithUnsupportedMetadataVersion) {
  std::string buffer;
  absl::StatusOr<std::unique_ptr<ModelMetadataExtractor>> extractor =
      CreateMetadataExtractor(kMobileIcaWithUnsupportedMetadataVersion,
                              &buffer);
  EXPECT_THAT(extractor.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(extractor.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kMetadataInvalidSchemaVersionError))));
}

TEST(ModelMetadataExtractorTest, ModelCreatedWithNoNameMetadataField) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithMetadataContainingNoName, &buffer));
  EXPECT_EQ(extractor->GetModelMetadata(), nullptr);
}

// This model has no "TFLITE_METADATA" but has one metadata field for
// "min_runtime_version".
TEST(ModelMetadataExtractorTest,
     GetModelMetadataSucceedsWithoutTfLiteMetadata) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithoutTfLiteMetadata, &buffer));
  EXPECT_EQ(extractor->GetModelMetadata(), nullptr);
}

// This model has no metadata at all. Source:
// https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.25_224/1/default/1
TEST(ModelMetadataExtractorTest, GetModelMetadataSucceedsWithBlankMetadata) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileNetWithNoMetadata, &buffer));
  EXPECT_EQ(extractor->GetModelMetadata(), nullptr);
}

TEST(ModelMetadataExtractorTest, GetModelMetadataSucceedsWithMetadata) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithTfLiteMetadata, &buffer));
  ASSERT_TRUE(extractor->GetModelMetadata() != nullptr);
  ASSERT_TRUE(extractor->GetModelMetadata()->name() != nullptr);
  EXPECT_STREQ(extractor->GetModelMetadata()->name()->c_str(),
               "image_understanding/classifier/mobile_ica_V1");
}

TEST(ModelMetadataExtractorTest, GetAssociatedFileSucceeds) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithTfLiteMetadata, &buffer));
  MP_EXPECT_OK(extractor->GetAssociatedFile("0-labels.txt").status());
}

TEST(ModelMetadataExtractorTest, GetAssociatedFileFailsWithNoSuchFile) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithTfLiteMetadata, &buffer));
  absl::StatusOr<absl::string_view> file_contents =
      extractor->GetAssociatedFile("foo");
  EXPECT_THAT(file_contents.status().code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(
      file_contents.status().GetPayload(kMediaPipeTasksPayload),
      Optional(absl::Cord(absl::StrCat(
          MediaPipeTasksStatus::kMetadataAssociatedFileNotFoundError))));
}

TEST(ModelMetadataExtractorTest, FindFirstProcessUnitSucceeds) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithTfLiteMetadata, &buffer));
  const flatbuffers::Vector<flatbuffers::Offset<tflite::TensorMetadata>>*
      output_tensor_metadata = extractor->GetOutputTensorMetadata();
  ASSERT_EQ(output_tensor_metadata->size(), 1);
  absl::StatusOr<const tflite::ProcessUnit*> process_unit =
      ModelMetadataExtractor::FindFirstProcessUnit(
          *output_tensor_metadata->Get(0),
          tflite::ProcessUnitOptions_ScoreCalibrationOptions);
  MP_ASSERT_OK(process_unit.status());
  EXPECT_TRUE(process_unit.value() != nullptr);
}

TEST(ModelMetadataExtractorTest, FindFirstProcessUnitNonExistentReturnsNull) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithTfLiteMetadata, &buffer));
  const flatbuffers::Vector<flatbuffers::Offset<tflite::TensorMetadata>>*
      output_tensor_metadata = extractor->GetOutputTensorMetadata();
  ASSERT_EQ(output_tensor_metadata->size(), 1);
  absl::StatusOr<const tflite::ProcessUnit*> process_unit =
      ModelMetadataExtractor::FindFirstProcessUnit(
          *output_tensor_metadata->Get(0),
          tflite::ProcessUnitOptions_NormalizationOptions);
  MP_ASSERT_OK(process_unit.status());
  EXPECT_TRUE(process_unit.value() == nullptr);
}

TEST(ModelMetadataExtractorTest, FindFirstAssociatedFileNameSucceeds) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithTfLiteMetadata, &buffer));
  const flatbuffers::Vector<flatbuffers::Offset<tflite::TensorMetadata>>*
      output_tensor_metadata = extractor->GetOutputTensorMetadata();
  ASSERT_EQ(output_tensor_metadata->size(), 1);
  std::string filename = ModelMetadataExtractor::FindFirstAssociatedFileName(
      *output_tensor_metadata->Get(0),
      tflite::AssociatedFileType_TENSOR_AXIS_LABELS, kEnLocale);
  EXPECT_EQ(filename, kEnLabels);
}

TEST(ModelMetadataExtractorTest,
     FindFirstAssociatedFileNameWithUnknownLocaleReturnsEmpty) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithTfLiteMetadata, &buffer));
  const flatbuffers::Vector<flatbuffers::Offset<tflite::TensorMetadata>>*
      output_tensor_metadata = extractor->GetOutputTensorMetadata();
  ASSERT_EQ(output_tensor_metadata->size(), 1);
  std::string filename = ModelMetadataExtractor::FindFirstAssociatedFileName(
      *output_tensor_metadata->Get(0),
      tflite::AssociatedFileType_TENSOR_AXIS_LABELS, kFrLocale);
  EXPECT_EQ(filename, "");
}

TEST(ModelMetadataExtractorTest,
     FindFirstAssociatedFileNameNonExistentReturnsEmpty) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithTfLiteMetadata, &buffer));
  const flatbuffers::Vector<flatbuffers::Offset<tflite::TensorMetadata>>*
      output_tensor_metadata = extractor->GetOutputTensorMetadata();
  ASSERT_EQ(output_tensor_metadata->size(), 1);
  std::string filename = ModelMetadataExtractor::FindFirstAssociatedFileName(
      *output_tensor_metadata->Get(0),
      tflite::AssociatedFileType_TENSOR_VALUE_LABELS);
  EXPECT_EQ(filename, "");
}

TEST(ModelMetadataExtractorTest, GetInputTensorMetadataWorks) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithTfLiteMetadata, &buffer));
  EXPECT_TRUE(extractor->GetInputTensorMetadata() != nullptr);
}

TEST(ModelMetadataExtractorTest,
     GetInputTensorMetadataWithoutTfLiteMetadataWorks) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithoutTfLiteMetadata, &buffer));
  EXPECT_TRUE(extractor->GetInputTensorMetadata() == nullptr);
}

TEST(ModelMetadataExtractorTest, GetInputTensorMetadataWithIndexWorks) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithTfLiteMetadata, &buffer));
  EXPECT_TRUE(extractor->GetInputTensorMetadata(0) != nullptr);
}

TEST(ModelMetadataExtractorTest,
     GetInputTensorMetadataWithIndexAndWithoutTfLiteMetadataWorks) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithoutTfLiteMetadata, &buffer));
  EXPECT_TRUE(extractor->GetInputTensorMetadata(0) == nullptr);
}

TEST(ModelMetadataExtractorTest,
     GetInputTensorMetadataWithOutOfRangeIndexWorks) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithoutTfLiteMetadata, &buffer));
  EXPECT_TRUE(extractor->GetInputTensorMetadata(-1) == nullptr);
  EXPECT_TRUE(extractor->GetInputTensorMetadata(2) == nullptr);
}

TEST(ModelMetadataExtractorTest, GetInputTensorCountWorks) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithTfLiteMetadata, &buffer));
  EXPECT_EQ(extractor->GetInputTensorCount(), 1);
}

TEST(ModelMetadataExtractorTest,
     GetInputTensorWithoutTfLiteMetadataCountWorks) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithoutTfLiteMetadata, &buffer));
  EXPECT_EQ(extractor->GetInputTensorCount(), 0);
}

TEST(ModelMetadataExtractorTest, GetOutputTensorMetadataWithIndexWorks) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithTfLiteMetadata, &buffer));
  EXPECT_TRUE(extractor->GetOutputTensorMetadata(0) != nullptr);
}

TEST(ModelMetadataExtractorTest,
     GetOutputTensorMetadataWithIndexAndWithoutTfLiteMetadataWorks) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithoutTfLiteMetadata, &buffer));
  EXPECT_TRUE(extractor->GetOutputTensorMetadata(0) == nullptr);
}

TEST(ModelMetadataExtractorTest,
     GetOutputTensorMetadataWithOutOfRangeIndexWorks) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithoutTfLiteMetadata, &buffer));
  EXPECT_TRUE(extractor->GetOutputTensorMetadata(-1) == nullptr);
  EXPECT_TRUE(extractor->GetOutputTensorMetadata(2) == nullptr);
}

TEST(ModelMetadataExtractorTest, GetOutputTensorCountWorks) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithTfLiteMetadata, &buffer));
  EXPECT_EQ(extractor->GetOutputTensorCount(), 1);
}

TEST(ModelMetadataExtractorTest,
     GetOutputTensorWithoutTfLiteMetadataCountWorks) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithoutTfLiteMetadata, &buffer));
  EXPECT_EQ(extractor->GetOutputTensorCount(), 0);
}

TEST(ModelMetadataExtractorTest, GetModelVersionWorks) {
  std::string buffer;
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> extractor,
      CreateMetadataExtractor(kMobileIcaWithTfLiteMetadata, &buffer));
  MP_EXPECT_OK(extractor->GetModelVersion().status());
}

}  // namespace
}  // namespace metadata
}  // namespace tasks
}  // namespace mediapipe
