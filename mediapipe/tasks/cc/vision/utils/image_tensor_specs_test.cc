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

#include "mediapipe/tasks/cc/vision/utils/image_tensor_specs.h"

#include <array>
#include <memory>
#include <string>
#include <type_traits>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"
#include "tensorflow/lite/core/shims/cc/shims_test_util.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::metadata::ModelMetadataExtractor;
using ::testing::ContainerEq;
using ::testing::Optional;
using ::tflite::ColorSpaceType_RGB;
using ::tflite::EnumNameTensorType;

constexpr char kTestModelResourcesTag[] = "test_model_resources";
constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kMobileNetDefault[] = "mobilenet_v1_0.25_224_1_default_1.tflite";
constexpr char kMobileNetMetadata[] =
    "mobilenet_v1_0.25_224_1_metadata_1.tflite";
// This model has partial metadata, namely (in JSON format):
//
// {
//   "name": "MobileNetV1 image classifier (quantized)",
//   "description": "Identify the most prominent object in the image from a set
//   of 1,001 categories such as trees, animals, food, vehicles, person etc.",
//   "version": "v1",
//   "author": "TensorFlow",
//   "license": "Apache License. Version 2.0
//   http://www.apache.org/licenses/LICENSE-2.0.",
//   "min_parser_version": "1.0.0"
// }
constexpr char kMobileNetQuantizedPartialMetadata[] =
    "mobilenet_v1_0.25_224_quant_without_subgraph_metadata.tflite";

class ImageTensorSpecsTest : public tflite_shims::testing::Test {};

TEST_F(ImageTensorSpecsTest, BuildInputImageTensorSpecsWorks) {
  auto model_file = std::make_unique<core::proto::ExternalFile>();
  model_file->set_file_name(
      JoinPath("./", kTestDataDirectory, kMobileNetMetadata));
  MP_ASSERT_OK_AND_ASSIGN(auto model_resources,
                          core::ModelResources::Create(kTestModelResourcesTag,
                                                       std::move(model_file)));

  const tflite::Model& model = *model_resources->GetTfLiteModel();
  ASSERT_EQ(model.subgraphs()->size(), 1);
  const auto* primary_subgraph = (*model.subgraphs())[0];
  ASSERT_EQ(primary_subgraph->inputs()->size(), 1);
  auto* input_tensor =
      (*primary_subgraph->tensors())[(*primary_subgraph->inputs())[0]];
  const ModelMetadataExtractor& metadata_extractor =
      *model_resources->GetMetadataExtractor();
  MP_ASSERT_OK_AND_ASSIGN(auto* metadata,
                          GetImageTensorMetadataIfAny(metadata_extractor, 0));
  absl::StatusOr<ImageTensorSpecs> input_specs_or =
      BuildInputImageTensorSpecs(*input_tensor, metadata);
  MP_ASSERT_OK(input_specs_or);

  const ImageTensorSpecs& input_specs = input_specs_or.value();
  EXPECT_EQ(input_specs.image_width, 224);
  EXPECT_EQ(input_specs.image_height, 224);
  EXPECT_EQ(input_specs.color_space, ColorSpaceType_RGB);
  EXPECT_STREQ(EnumNameTensorType(input_specs.tensor_type),
               EnumNameTensorType(tflite::TensorType_FLOAT32));

  EXPECT_TRUE(input_specs.normalization_options.has_value());
  EXPECT_THAT(input_specs.normalization_options.value().num_values, 1);
  std::array<float, 3> expected_values = {127.5f, 127.5, 127.5};
  EXPECT_THAT(input_specs.normalization_options.value().mean_values,
              ContainerEq(expected_values));
  EXPECT_THAT(input_specs.normalization_options.value().std_values,
              ContainerEq(expected_values));
}

TEST_F(
    ImageTensorSpecsTest,
    BuildInputImageTensorSpecsForFloatInputWithoutNormalizationOptionsFails) {
  auto model_file = std::make_unique<core::proto::ExternalFile>();
  model_file->set_file_name(
      JoinPath("./", kTestDataDirectory, kMobileNetDefault));
  MP_ASSERT_OK_AND_ASSIGN(auto model_resources,
                          core::ModelResources::Create(kTestModelResourcesTag,
                                                       std::move(model_file)));

  const tflite::Model& model = *model_resources->GetTfLiteModel();
  ASSERT_EQ(model.subgraphs()->size(), 1);
  const auto* primary_subgraph = (*model.subgraphs())[0];
  ASSERT_EQ(primary_subgraph->inputs()->size(), 1);
  auto* input_tensor =
      (*primary_subgraph->tensors())[(*primary_subgraph->inputs())[0]];
  const ModelMetadataExtractor& metadata_extractor =
      *model_resources->GetMetadataExtractor();
  MP_ASSERT_OK_AND_ASSIGN(auto* metadata,
                          GetImageTensorMetadataIfAny(metadata_extractor, 0));
  absl::StatusOr<ImageTensorSpecs> input_specs_or =
      BuildInputImageTensorSpecs(*input_tensor, metadata);

  EXPECT_THAT(input_specs_or, StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(
      input_specs_or.status().GetPayload(kMediaPipeTasksPayload),
      Optional(absl::Cord(absl::StrCat(
          MediaPipeTasksStatus::kMetadataMissingNormalizationOptionsError))));
}

TEST_F(ImageTensorSpecsTest,
       BuildInputImageTensorSpecsWithPartialMetadataWorks) {
  auto model_file = std::make_unique<core::proto::ExternalFile>();
  model_file->set_file_name(
      JoinPath("./", kTestDataDirectory, kMobileNetQuantizedPartialMetadata));
  MP_ASSERT_OK_AND_ASSIGN(auto model_resources,
                          core::ModelResources::Create(kTestModelResourcesTag,
                                                       std::move(model_file)));

  const tflite::Model& model = *model_resources->GetTfLiteModel();
  ASSERT_EQ(model.subgraphs()->size(), 1);
  const auto* primary_subgraph = (*model.subgraphs())[0];
  ASSERT_EQ(primary_subgraph->inputs()->size(), 1);
  auto* input_tensor =
      (*primary_subgraph->tensors())[(*primary_subgraph->inputs())[0]];
  const ModelMetadataExtractor& metadata_extractor =
      *model_resources->GetMetadataExtractor();
  MP_ASSERT_OK_AND_ASSIGN(auto* metadata,
                          GetImageTensorMetadataIfAny(metadata_extractor, 0));
  absl::StatusOr<ImageTensorSpecs> input_specs_or =
      BuildInputImageTensorSpecs(*input_tensor, metadata);
  MP_ASSERT_OK(input_specs_or);

  const ImageTensorSpecs& input_specs = input_specs_or.value();
  EXPECT_EQ(input_specs.image_width, 224);
  EXPECT_EQ(input_specs.image_height, 224);
  EXPECT_EQ(input_specs.color_space, ColorSpaceType_RGB);
  EXPECT_STREQ(EnumNameTensorType(input_specs.tensor_type),
               EnumNameTensorType(tflite::TensorType_UINT8));
  EXPECT_EQ(input_specs.normalization_options, absl::nullopt);
}

}  // namespace
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
