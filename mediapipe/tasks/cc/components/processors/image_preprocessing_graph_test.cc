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

#include "mediapipe/tasks/cc/components/processors/image_preprocessing_graph.h"

#include <memory>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/gpu/gpu_origin.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/image_preprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/acceleration.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"
#include "tensorflow/lite/test_util.h"

namespace mediapipe {
namespace tasks {
namespace components {
namespace processors {
namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::core::ModelResources;
using ::mediapipe::tasks::core::TaskRunner;
using ::mediapipe::tasks::vision::DecodeImageFromFile;
using ::testing::ContainerEq;
using ::testing::HasSubstr;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::Values;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kMobileNetFloatWithMetadata[] = "mobilenet_v2_1.0_224.tflite";
constexpr char kMobileNetFloatWithoutMetadata[] =
    "mobilenet_v1_0.25_224_1_default_1.tflite";
constexpr char kMobileNetQuantizedWithMetadata[] =
    "mobilenet_v1_0.25_224_quant.tflite";
constexpr char kMobileNetQuantizedWithoutMetadata[] =
    "mobilenet_v1_0.25_192_quantized_1_default_1.tflite";

constexpr char kTestImage[] = "burger.jpg";
constexpr int kTestImageWidth = 480;
constexpr int kTestImageHeight = 325;

constexpr char kTestModelResourcesTag[] = "test_model_resources";
constexpr std::array<float, 16> kIdentityMatrix = {1, 0, 0, 0, 0, 1, 0, 0,
                                                   0, 0, 1, 0, 0, 0, 0, 1};

constexpr char kImageTag[] = "IMAGE";
constexpr char kImageName[] = "image_in";
constexpr char kMatrixTag[] = "MATRIX";
constexpr char kMatrixName[] = "matrix_out";
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kTensorsName[] = "tensors_out";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kImageSizeName[] = "image_size_out";
constexpr char kLetterboxPaddingTag[] = "LETTERBOX_PADDING";
constexpr char kLetterboxPaddingName[] = "letterbox_padding_out";

constexpr float kLetterboxMaxAbsError = 1e-5;

// Helper function to get ModelResources.
absl::StatusOr<std::unique_ptr<ModelResources>> CreateModelResourcesForModel(
    absl::string_view model_name) {
  auto external_file = std::make_unique<core::proto::ExternalFile>();
  external_file->set_file_name(JoinPath("./", kTestDataDirectory, model_name));
  return ModelResources::Create(kTestModelResourcesTag,
                                std::move(external_file));
}

// Helper function to create a TaskRunner from ModelResources.
absl::StatusOr<std::unique_ptr<TaskRunner>> CreateTaskRunner(
    const ModelResources& model_resources, bool keep_aspect_ratio) {
  Graph graph;

  auto& preprocessing = graph.AddNode(
      "mediapipe.tasks.components.processors.ImagePreprocessingGraph");
  auto& options =
      preprocessing.GetOptions<proto::ImagePreprocessingGraphOptions>();
  options.mutable_image_to_tensor_options()->set_keep_aspect_ratio(
      keep_aspect_ratio);
  MP_RETURN_IF_ERROR(
      ConfigureImagePreprocessingGraph(model_resources, false, &options));
  graph[Input<Image>(kImageTag)].SetName(kImageName) >>
      preprocessing.In(kImageTag);
  preprocessing.Out(kTensorsTag).SetName(kTensorsName) >>
      graph[Output<std::vector<Tensor>>(kTensorsTag)];
  preprocessing.Out(kMatrixTag).SetName(kMatrixName) >>
      graph[Output<std::array<float, 16>>(kMatrixTag)];
  preprocessing.Out(kImageSizeTag).SetName(kImageSizeName) >>
      graph[Output<std::pair<int, int>>(kImageSizeTag)];
  preprocessing.Out(kLetterboxPaddingTag).SetName(kLetterboxPaddingName) >>
      graph[Output<std::array<float, 4>>(kLetterboxPaddingTag)];

  return TaskRunner::Create(graph.GetConfig());
}

class ConfigureTest : public tflite::testing::Test {};

TEST_F(ConfigureTest, SucceedsWithQuantizedModelWithMetadata) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kMobileNetQuantizedWithMetadata));

  proto::ImagePreprocessingGraphOptions options;
  MP_EXPECT_OK(
      ConfigureImagePreprocessingGraph(*model_resources, false, &options));

  EXPECT_THAT(options, EqualsProto(
                           R"pb(image_to_tensor_options {
                                  output_tensor_width: 224
                                  output_tensor_height: 224
                                  output_tensor_uint_range { min: 0 max: 255 }
                                  gpu_origin: TOP_LEFT
                                }
                                backend: CPU_BACKEND)pb"));
}

TEST_F(ConfigureTest, SucceedsWithQuantizedModelWithoutMetadata) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kMobileNetQuantizedWithoutMetadata));

  proto::ImagePreprocessingGraphOptions options;
  MP_EXPECT_OK(
      ConfigureImagePreprocessingGraph(*model_resources, false, &options));

  EXPECT_THAT(options, EqualsProto(
                           R"pb(image_to_tensor_options {
                                  output_tensor_width: 192
                                  output_tensor_height: 192
                                  output_tensor_uint_range { min: 0 max: 255 }
                                  gpu_origin: TOP_LEFT
                                }
                                backend: CPU_BACKEND)pb"));
}

TEST_F(ConfigureTest, SucceedsWithFloatModelWithMetadata) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kMobileNetFloatWithMetadata));

  proto::ImagePreprocessingGraphOptions options;
  MP_EXPECT_OK(
      ConfigureImagePreprocessingGraph(*model_resources, false, &options));

  EXPECT_THAT(options, EqualsProto(
                           R"pb(image_to_tensor_options {
                                  output_tensor_width: 224
                                  output_tensor_height: 224
                                  output_tensor_float_range { min: -1 max: 1 }
                                  gpu_origin: TOP_LEFT
                                }
                                backend: CPU_BACKEND)pb"));
}

TEST_F(ConfigureTest, SucceedsWithQuantizedModelFallbacksCpuBackend) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kMobileNetQuantizedWithMetadata));

  proto::ImagePreprocessingGraphOptions options;
  core::proto::Acceleration acceleration;
  acceleration.mutable_gpu();
  bool use_gpu = DetermineImagePreprocessingGpuBackend(acceleration);
  EXPECT_TRUE(use_gpu);
  MP_EXPECT_OK(
      ConfigureImagePreprocessingGraph(*model_resources, use_gpu, &options));

  EXPECT_THAT(options, EqualsProto(
                           R"pb(image_to_tensor_options {
                                  output_tensor_width: 224
                                  output_tensor_height: 224
                                  output_tensor_uint_range { min: 0 max: 255 }
                                  gpu_origin: TOP_LEFT
                                }
                                backend: CPU_BACKEND)pb"));
}

TEST_F(ConfigureTest, SucceedsWithFloatModelGpuBackend) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kMobileNetFloatWithMetadata));

  proto::ImagePreprocessingGraphOptions options;
  core::proto::Acceleration acceleration;
  acceleration.mutable_gpu();
  bool use_gpu = DetermineImagePreprocessingGpuBackend(acceleration);
  EXPECT_TRUE(use_gpu);
  MP_EXPECT_OK(
      ConfigureImagePreprocessingGraph(*model_resources, use_gpu, &options));

  EXPECT_THAT(options, EqualsProto(
                           R"pb(image_to_tensor_options {
                                  output_tensor_width: 224
                                  output_tensor_height: 224
                                  output_tensor_float_range { min: -1 max: 1 }
                                  gpu_origin: TOP_LEFT
                                }
                                backend: GPU_BACKEND)pb"));
}

TEST_F(ConfigureTest, SucceedsGpuOriginConventional) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kMobileNetFloatWithMetadata));

  proto::ImagePreprocessingGraphOptions options;
  MP_EXPECT_OK(ConfigureImagePreprocessingGraph(
      *model_resources, true, mediapipe::GpuOrigin::CONVENTIONAL, &options));

  EXPECT_THAT(options, EqualsProto(
                           R"pb(image_to_tensor_options {
                                  output_tensor_width: 224
                                  output_tensor_height: 224
                                  output_tensor_float_range { min: -1 max: 1 }
                                  gpu_origin: CONVENTIONAL
                                }
                                backend: GPU_BACKEND)pb"));
}

TEST_F(ConfigureTest, FailsWithFloatModelWithoutMetadata) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(kMobileNetFloatWithoutMetadata));

  proto::ImagePreprocessingGraphOptions options;
  auto status =
      ConfigureImagePreprocessingGraph(*model_resources, false, &options);

  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(status.message(),
              HasSubstr("requires specifying NormalizationOptions metadata"));
}

// Struct holding the parameters for parameterized PreprocessingTest class.
struct PreprocessingParams {
  // The name of this test, for convenience when displaying test results.
  std::string test_name;
  // The filename of the model to test.
  std::string input_model_name;
  // If true, keep test image aspect ratio.
  bool keep_aspect_ratio;
  // The expected output tensor type.
  Tensor::ElementType expected_type;
  // The expected outoput tensor shape.
  std::vector<int> expected_shape;
  // The expected output letterbox padding;
  std::array<float, 4> expected_letterbox_padding;
};

class PreprocessingTest : public testing::TestWithParam<PreprocessingParams> {};

TEST_P(PreprocessingTest, Succeeds) {
  MP_ASSERT_OK_AND_ASSIGN(
      Image image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, kTestImage)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateModelResourcesForModel(GetParam().input_model_name));
  MP_ASSERT_OK_AND_ASSIGN(
      auto task_runner,
      CreateTaskRunner(*model_resources, GetParam().keep_aspect_ratio));

  auto output_packets =
      task_runner->Process({{kImageName, MakePacket<Image>(std::move(image))}});
  MP_ASSERT_OK(output_packets);

  const std::vector<Tensor>& tensors =
      (*output_packets)[kTensorsName].Get<std::vector<Tensor>>();
  EXPECT_EQ(tensors.size(), 1);
  EXPECT_EQ(tensors[0].element_type(), GetParam().expected_type);
  EXPECT_THAT(tensors[0].shape().dims, ContainerEq(GetParam().expected_shape));
  auto& matrix = (*output_packets)[kMatrixName].Get<std::array<float, 16>>();
  if (!GetParam().keep_aspect_ratio) {
    for (int i = 0; i < matrix.size(); ++i) {
      EXPECT_FLOAT_EQ(matrix[i], kIdentityMatrix[i]);
    }
  }
  auto& image_size =
      (*output_packets)[kImageSizeName].Get<std::pair<int, int>>();
  EXPECT_EQ(image_size.first, kTestImageWidth);
  EXPECT_EQ(image_size.second, kTestImageHeight);
  std::array<float, 4> letterbox_padding =
      (*output_packets)[kLetterboxPaddingName].Get<std::array<float, 4>>();
  for (int i = 0; i < letterbox_padding.size(); ++i) {
    EXPECT_NEAR(letterbox_padding[i], GetParam().expected_letterbox_padding[i],
                kLetterboxMaxAbsError);
  }
}

INSTANTIATE_TEST_SUITE_P(
    PreprocessingTest, PreprocessingTest,
    Values(
        PreprocessingParams{.test_name = "kMobileNetQuantizedWithMetadata",
                            .input_model_name = kMobileNetQuantizedWithMetadata,
                            .keep_aspect_ratio = false,
                            .expected_type = Tensor::ElementType::kUInt8,
                            .expected_shape = {1, 224, 224, 3},
                            .expected_letterbox_padding = {0, 0, 0, 0}},
        PreprocessingParams{
            .test_name = "kMobileNetQuantizedWithoutMetadata",
            .input_model_name = kMobileNetQuantizedWithoutMetadata,
            .keep_aspect_ratio = false,
            .expected_type = Tensor::ElementType::kUInt8,
            .expected_shape = {1, 192, 192, 3},
            .expected_letterbox_padding = {0, 0, 0, 0}},
        PreprocessingParams{.test_name = "kMobileNetFloatWithMetadata",
                            .input_model_name = kMobileNetFloatWithMetadata,
                            .keep_aspect_ratio = false,
                            .expected_type = Tensor::ElementType::kFloat32,
                            .expected_shape = {1, 224, 224, 3},
                            .expected_letterbox_padding = {0, 0, 0, 0}},
        PreprocessingParams{
            .test_name = "kMobileNetFloatWithMetadataKeepAspectRatio",
            .input_model_name = kMobileNetFloatWithMetadata,
            .keep_aspect_ratio = true,
            .expected_type = Tensor::ElementType::kFloat32,
            .expected_shape = {1, 224, 224, 3},
            .expected_letterbox_padding = {/*left*/ 0,
                                           /*top*/ 0.161458,
                                           /*right*/ 0,
                                           /*bottom*/ 0.161458}}),
    [](const TestParamInfo<PreprocessingTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace processors
}  // namespace components
}  // namespace tasks
}  // namespace mediapipe
