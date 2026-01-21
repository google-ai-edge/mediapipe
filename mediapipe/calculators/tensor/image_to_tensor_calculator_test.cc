// Copyright 2020 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.h"

#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/api3/function_runner.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/image_test_utils.h"

#if !MEDIAPIPE_DISABLE_GPU && !MEDIAPIPE_METAL_ENABLED
#include "mediapipe/gpu/gl_context.h"
#endif  // !MEDIAPIPE_DISABLE_GPU && !MEDIAPIPE_METAL_ENABLED

namespace mediapipe {
namespace {

using ::mediapipe::api3::GenericGraph;
using ::mediapipe::api3::ImageToTensorNode;
using ::mediapipe::api3::Packet;
using ::mediapipe::api3::Runner;
using ::mediapipe::api3::Stream;

constexpr char kTestDataDir[] =
    "/mediapipe/calculators/tensor/testdata/"
    "image_to_tensor/";

std::string GetFilePath(absl::string_view filename) {
  return file::JoinPath("./", kTestDataDir, filename);
}

template <typename T>
struct Range {
  T min;
  T max;
};

template <typename T>
absl::Status TensorAndExpectedMatch(const Tensor& tensor, const Range<T>& range,
                                    cv::Mat expected) {
  const int channels = tensor.shape().dims[3];
  RET_CHECK(channels == 1 || channels == 3);
  Tensor::CpuReadView view = tensor.GetCpuReadView();
  cv::Mat tensor_mat;
  if constexpr (std::is_same_v<T, int>) {
    RET_CHECK(tensor.element_type() == Tensor::ElementType::kInt8);
    tensor_mat =
        cv::Mat(expected.rows, expected.cols, channels == 1 ? CV_8SC1 : CV_8SC3,
                const_cast<int8_t*>(view.buffer<int8_t>()));
  } else if constexpr (std::is_same_v<T, uint>) {
    RET_CHECK(tensor.element_type() == Tensor::ElementType::kUInt8);
    tensor_mat =
        cv::Mat(expected.rows, expected.cols, channels == 1 ? CV_8UC1 : CV_8UC3,
                const_cast<uint8_t*>(view.buffer<uint8_t>()));
  } else if constexpr (std::is_same_v<T, float>) {
    RET_CHECK(tensor.element_type() == Tensor::ElementType::kFloat32);
    tensor_mat = cv::Mat(expected.rows, expected.cols,
                         channels == 1 ? CV_32FC1 : CV_32FC3,
                         const_cast<float*>(view.buffer<float>()));
  }

  cv::Mat result_rgb;
  auto transformation =
      GetValueRangeTransformation(range.min, range.max, 0.0f, 255.0f).value();
  tensor_mat.convertTo(result_rgb, channels == 1 ? CV_8UC1 : CV_8UC3,
                       transformation.scale, transformation.offset);

  cv::Mat diff;
  cv::absdiff(result_rgb, expected, diff);
  double max_diff;
  cv::minMaxLoc(diff, nullptr, &max_diff);
  // Expects the maximum absolute pixel-by-pixel difference is less than 5.
  RET_CHECK(max_diff <= 5);
  return absl::OkStatus();
}

mediapipe::ImageFormat::Format GetImageFormat(int image_channels) {
  if (image_channels == 4) {
    return ImageFormat::SRGBA;
  } else if (image_channels == 3) {
    return ImageFormat::SRGB;
  } else if (image_channels == 1) {
    return ImageFormat::GRAY8;
  }
  ABSL_CHECK(false) << "Unsupported input image channels: " << image_channels;
}

ImageFrame ReadImageFrameRgb(absl::string_view name) {
  cv::Mat input = GetRgb(GetFilePath(name));
  return ImageFrame(GetImageFormat(input.channels()), input.cols, input.rows,
                    input.step, input.data,
                    [input](uint8_t*) mutable { input.release(); });
}

Image ReadImageRgb(absl::string_view name) {
  cv::Mat input = GetRgb(GetFilePath(name));
  return Image(std::make_shared<ImageFrame>(
      GetImageFormat(input.channels()), input.cols, input.rows, input.step,
      input.data, [input](uint8_t*) mutable { input.release(); }));
}

ImageFrame ReadImageFrameRgba(absl::string_view name) {
  cv::Mat input = GetRgba(GetFilePath(name));
  return ImageFrame(GetImageFormat(input.channels()), input.cols, input.rows,
                    input.step, input.data,
                    [input](uint8_t*) mutable { input.release(); });
}

ImageFrame ReadImageFrameGray(absl::string_view name) {
  cv::Mat input = GetGray(GetFilePath(name));
  return ImageFrame(GetImageFormat(input.channels()), input.cols, input.rows,
                    input.step, input.data,
                    [input](uint8_t*) mutable { input.release(); });
}

NormalizedRect MakeRect(float x_center, float y_center, float width,
                        float height, float rotation) {
  mediapipe::NormalizedRect rect;
  rect.set_x_center(x_center);
  rect.set_y_center(y_center);
  rect.set_width(width);
  rect.set_height(height);
  rect.set_rotation(rotation);
  return rect;
}

struct TestCase {
  std::string name;
  std::optional<ImageToTensorCalculatorOptions::BorderMode> border_mode;
  std::optional<std::pair<int, int>> tensor_dims;
  bool keep_aspect_ratio;
  ImageFormat::Format image_format;
  NormalizedRect norm_rect;
  std::string expected_output;
  std::pair<float, float> range;
};

using ImageToTensorCalculatorParameterizedTest =
    testing::TestWithParam<TestCase>;

TEST_P(ImageToTensorCalculatorParameterizedTest, ConvertsImageToTensor) {
  const auto& p = GetParam();

  ImageFrame input;
  cv::Mat expected_output;
  if (p.image_format == ImageFormat::GRAY8) {
    input = ReadImageFrameGray("input.jpg");
    expected_output = GetGray(GetFilePath(p.expected_output));
  } else if (p.image_format == ImageFormat::SRGB) {
    input = ReadImageFrameRgb("input.jpg");
    expected_output = GetRgb(GetFilePath(p.expected_output));
  } else if (p.image_format == ImageFormat::SRGBA) {
    input = ReadImageFrameRgba("input.jpg");
    expected_output = GetRgb(GetFilePath(p.expected_output));
  } else {
    FAIL() << "Unsupported image format provided in test case";
  }

  const Range<float> kRange = {.min = p.range.first, .max = p.range.second};
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([&](GenericGraph& graph, Stream<ImageFrame> image,
                      Stream<NormalizedRect> norm_rect) -> Stream<Tensor> {
        auto& node = graph.AddNode<ImageToTensorNode>();
        {
          auto& opts = *node.options.Mutable();
          if (p.border_mode) {
            opts.set_border_mode(*p.border_mode);
          }
          if (p.tensor_dims) {
            opts.set_output_tensor_width(p.tensor_dims->first);
            opts.set_output_tensor_height(p.tensor_dims->second);
          }
          opts.set_keep_aspect_ratio(p.keep_aspect_ratio);
          auto& float_range = *opts.mutable_output_tensor_float_range();
          float_range.set_min(kRange.min);
          float_range.set_max(kRange.max);
        }
        node.in.Set(image);
        node.in_norm_rect.Set(norm_rect);
        return node.out_tensor.Get();
      }).Create());
  MP_ASSERT_OK_AND_ASSIGN(
      api3::Packet<Tensor> tensor_packet,
      runner.Run(api3::MakePacket<ImageFrame>(std::move(input)),
                 api3::MakePacket<NormalizedRect>(p.norm_rect)));
  ASSERT_TRUE(tensor_packet);
  EXPECT_THAT(
      TensorAndExpectedMatch(tensor_packet.GetOrDie(), kRange, expected_output),
      StatusIs(absl::StatusCode::kOk));
}

INSTANTIATE_TEST_SUITE_P(
    ImageToTensorCalculatorParameterizedTests,
    ImageToTensorCalculatorParameterizedTest,
    testing::ValuesIn<TestCase>(
        {{.name = "MediumSubRectKeepAspect",
          .border_mode = std::nullopt,
          .tensor_dims = {{256, 256}},
          .keep_aspect_ratio = true,
          .image_format = ImageFormat::SRGB,
          .norm_rect = MakeRect(0.65f, 0.4f, 0.5f, 0.5f, 0),
          .expected_output = "medium_sub_rect_keep_aspect.png",
          .range = {0.0f, 1.0f}},
         {.name = "MediumSubRectKeepAspectBorderZero",
          .border_mode = ImageToTensorCalculatorOptions::BORDER_ZERO,
          .tensor_dims = {{256, 256}},
          .keep_aspect_ratio = true,
          .image_format = ImageFormat::SRGB,
          .norm_rect = MakeRect(0.65f, 0.4f, 0.5f, 0.5f, 0),
          .expected_output = "medium_sub_rect_keep_aspect_border_zero.png",
          .range = {0.0f, 1.0f}},
         {.name = "MediumSubRectKeepAspectWithRotation",
          .border_mode = ImageToTensorCalculatorOptions::BORDER_REPLICATE,
          .tensor_dims = {{256, 256}},
          .keep_aspect_ratio = true,
          .image_format = ImageFormat::SRGB,
          .norm_rect = MakeRect(0.65f, 0.4f, 0.5f, 0.5f, M_PI * 90.0f / 180.0f),
          .expected_output = "medium_sub_rect_keep_aspect_with_rotation.png",
          .range = {0.0f, 1.0f}},
         {.name = "MediumSubRectKeepAspectWithRotationBorderZero",
          .border_mode = ImageToTensorCalculatorOptions::BORDER_ZERO,
          .tensor_dims = {{256, 256}},
          .keep_aspect_ratio = true,
          .image_format = ImageFormat::SRGB,
          .norm_rect = MakeRect(0.65f, 0.4f, 0.5f, 0.5f, M_PI * 90.0f / 180.0f),
          .expected_output =
              "medium_sub_rect_keep_aspect_with_rotation_border_zero.png",
          .range = {0.0f, 1.0f}},
         {.name = "MediumSubRectWithRotation",
          .border_mode = ImageToTensorCalculatorOptions::BORDER_REPLICATE,
          .tensor_dims = {{256, 256}},
          .keep_aspect_ratio = false,
          .image_format = ImageFormat::SRGB,
          .norm_rect = MakeRect(0.65f, 0.4f, 0.5f, 0.5f,
                                M_PI * -45.0f / 180.0f),
          .expected_output = "medium_sub_rect_with_rotation.png",
          .range = {-1.0f, 1.0f}},
         {.name = "MediumSubRectWithRotationBorderZero",
          .border_mode = ImageToTensorCalculatorOptions::BORDER_ZERO,
          .tensor_dims = {{256, 256}},
          .keep_aspect_ratio = false,
          .image_format = ImageFormat::SRGB,
          .norm_rect = MakeRect(0.65f, 0.4f, 0.5f, 0.5f,
                                M_PI * -45.0f / 180.0f),
          .expected_output = "medium_sub_rect_with_rotation_border_zero.png",
          .range = {-1.0f, 1.0f}},
         {.name = "LargeSubRect",
          .border_mode = ImageToTensorCalculatorOptions::BORDER_REPLICATE,
          .tensor_dims = {{128, 128}},
          .keep_aspect_ratio = false,
          .image_format = ImageFormat::SRGB,
          .norm_rect = MakeRect(0.5f, 0.5f, 1.5f, 1.1f, 0),
          .expected_output = "large_sub_rect.png",
          .range = {0.0f, 1.0f}},
         {.name = "LargeSubRectBorderZero",
          .border_mode = ImageToTensorCalculatorOptions::BORDER_ZERO,
          .tensor_dims = {{128, 128}},
          .keep_aspect_ratio = false,
          .image_format = ImageFormat::SRGB,
          .norm_rect = MakeRect(0.5f, 0.5f, 1.5f, 1.1f, 0),
          .expected_output = "large_sub_rect_border_zero.png",
          .range = {0.0f, 1.0f}},
         {.name = "LargeSubRectKeepAspect",
          .border_mode = ImageToTensorCalculatorOptions::BORDER_REPLICATE,
          .tensor_dims = {{128, 128}},
          .keep_aspect_ratio = true,
          .image_format = ImageFormat::SRGB,
          .norm_rect = MakeRect(0.5f, 0.5f, 1.5f, 1.1f, 0),
          .expected_output = "large_sub_rect_keep_aspect.png",
          .range = {0.0f, 1.0f}},
         {.name = "LargeSubRectKeepAspectBorderZero",
          .border_mode = ImageToTensorCalculatorOptions::BORDER_ZERO,
          .tensor_dims = {{128, 128}},
          .keep_aspect_ratio = true,
          .image_format = ImageFormat::SRGB,
          .norm_rect = MakeRect(0.5f, 0.5f, 1.5f, 1.1f, 0),
          .expected_output = "large_sub_rect_keep_aspect_border_zero.png",
          .range = {0.0f, 1.0f}},
         {.name = "LargeSubRectKeepAspectWithRotation",
          .border_mode = std::nullopt,
          .tensor_dims = {{128, 128}},
          .keep_aspect_ratio = true,
          .image_format = ImageFormat::SRGBA,
          .norm_rect = MakeRect(0.5f, 0.5f, 1.5f, 1.1f, M_PI * -15.0f / 180.0f),
          .expected_output = "large_sub_rect_keep_aspect_with_rotation.png",
          .range = {0.0f, 1.0f}},
         {.name = "LargeSubRectKeepAspectWithRotationGray",
          .border_mode = std::nullopt,
          .tensor_dims = {{128, 128}},
          .keep_aspect_ratio = true,
          .image_format = ImageFormat::GRAY8,
          .norm_rect = MakeRect(0.5f, 0.5f, 1.5f, 1.1f, M_PI * -15.0f / 180.0f),
          .expected_output = "large_sub_rect_keep_aspect_with_rotation.png",
          .range = {0.0f, 1.0f}},
         {.name = "LargeSubRectKeepAspectWithRotationBorderZero",
          .border_mode = ImageToTensorCalculatorOptions::BORDER_ZERO,
          .tensor_dims = {{128, 128}},
          .keep_aspect_ratio = true,
          .image_format = ImageFormat::SRGBA,
          .norm_rect = MakeRect(0.5f, 0.5f, 1.5f, 1.1f, M_PI * -15.0f / 180.0f),
          .expected_output =
              "large_sub_rect_keep_aspect_with_rotation_border_zero.png",
          .range = {0.0f, 1.0f}},
         {.name = "LargeSubRectKeepAspectWithRotationBorderZeroGray",
          .border_mode = ImageToTensorCalculatorOptions::BORDER_ZERO,
          .tensor_dims = {{128, 128}},
          .keep_aspect_ratio = true,
          .image_format = ImageFormat::GRAY8,
          .norm_rect = MakeRect(0.5f, 0.5f, 1.5f, 1.1f, M_PI * -15.0f / 180.0f),
          .expected_output =
              "large_sub_rect_keep_aspect_with_rotation_border_zero.png",
          .range = {-0.5f, 0.5f}},
         {.name = "NoOpExceptRange",
          .border_mode = ImageToTensorCalculatorOptions::BORDER_REPLICATE,
          .tensor_dims = {{64, 128}},
          .keep_aspect_ratio = true,
          .image_format = ImageFormat::SRGBA,
          .norm_rect = MakeRect(0.5f, 0.5f, 1.0f, 1.0f, 0),
          .expected_output = "noop_except_range.png",
          .range = {-10.0f, 10.0f}},
         {.name = "NoOpExceptRangeBorderZero",
          .border_mode = ImageToTensorCalculatorOptions::BORDER_ZERO,
          .tensor_dims = {{64, 128}},
          .keep_aspect_ratio = true,
          .image_format = ImageFormat::SRGBA,
          .norm_rect = MakeRect(0.5f, 0.5f, 1.0f, 1.0f, 0),
          .expected_output = "noop_except_range.png",
          .range = {0.0f, 1.0f}},
         {.name = "NoOpExceptRangeAndUseInputImageDims",
          .border_mode = ImageToTensorCalculatorOptions::BORDER_ZERO,
          .tensor_dims = std::nullopt,
          .keep_aspect_ratio = false,
          .image_format = ImageFormat::SRGB,
          .norm_rect = MakeRect(0.5f, 0.5f, 1.0f, 1.0f, 0),
          .expected_output = "noop_except_range.png",
          .range = {-1.0f, 1.0f}}}),
    [](const testing::TestParamInfo<
        ImageToTensorCalculatorParameterizedTest::ParamType>& info) {
      return info.param.name;
    });

TEST(ImageToTensorCalculatorTest, MediumSubRectKeepAspectUIntRange) {
  const Range<uint> kRange = {.min = 0, .max = 255};
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([&](GenericGraph& graph, Stream<ImageFrame> image,
                      Stream<NormalizedRect> norm_rect) -> Stream<Tensor> {
        auto& node = graph.AddNode<ImageToTensorNode>();
        {
          auto& opts = *node.options.Mutable();
          opts.set_output_tensor_width(256);
          opts.set_output_tensor_height(256);
          opts.set_keep_aspect_ratio(true);

          auto& uint_range = *opts.mutable_output_tensor_uint_range();
          uint_range.set_min(kRange.min);
          uint_range.set_max(kRange.max);
        }
        node.in.Set(image);
        node.in_norm_rect.Set(norm_rect);
        return node.out_tensor.Get();
      }).Create());

  MP_ASSERT_OK_AND_ASSIGN(
      api3::Packet<Tensor> tensor_packet,
      runner.Run(api3::MakePacket<ImageFrame>(ReadImageFrameRgb("input.jpg")),
                 api3::MakePacket<NormalizedRect>(
                     MakeRect(0.65f, 0.4f, 0.5f, 0.5f, 0))));

  ASSERT_TRUE(tensor_packet);
  EXPECT_THAT(TensorAndExpectedMatch(
                  tensor_packet.GetOrDie(), kRange,
                  GetRgb(GetFilePath("medium_sub_rect_keep_aspect.png"))),
              StatusIs(absl::StatusCode::kOk));
}

TEST(ImageToTensorCalculatorTest, MediumSubRectKeepAspectIntRange) {
  const Range<int> kRange = {.min = -128, .max = 127};
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([&](GenericGraph& graph, Stream<ImageFrame> image,
                      Stream<NormalizedRect> norm_rect) -> Stream<Tensor> {
        auto& node = graph.AddNode<ImageToTensorNode>();
        {
          auto& opts = *node.options.Mutable();
          opts.set_output_tensor_width(256);
          opts.set_output_tensor_height(256);
          opts.set_keep_aspect_ratio(true);

          auto& int_range = *opts.mutable_output_tensor_int_range();
          int_range.set_min(kRange.min);
          int_range.set_max(kRange.max);
        }
        node.in.Set(image);
        node.in_norm_rect.Set(norm_rect);
        return node.out_tensor.Get();
      }).Create());

  MP_ASSERT_OK_AND_ASSIGN(
      api3::Packet<Tensor> tensor_packet,
      runner.Run(api3::MakePacket<ImageFrame>(ReadImageFrameRgb("input.jpg")),
                 api3::MakePacket<NormalizedRect>(
                     MakeRect(0.65f, 0.4f, 0.5f, 0.5f, 0))));

  ASSERT_TRUE(tensor_packet);
  EXPECT_THAT(TensorAndExpectedMatch(
                  tensor_packet.GetOrDie(), kRange,
                  GetRgb(GetFilePath("medium_sub_rect_keep_aspect.png"))),
              StatusIs(absl::StatusCode::kOk));
}

TEST(ImageToTensorCalculatorTest, MediumSubRectKeepAspectImageInput) {
  const Range<int> kRange = {.min = -128, .max = 127};
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([&](GenericGraph& graph, Stream<Image> image,
                      Stream<NormalizedRect> norm_rect) -> Stream<Tensor> {
        auto& node = graph.AddNode<ImageToTensorNode>();
        {
          auto& opts = *node.options.Mutable();
          opts.set_output_tensor_width(256);
          opts.set_output_tensor_height(256);
          opts.set_keep_aspect_ratio(true);

          auto& int_range = *opts.mutable_output_tensor_int_range();
          int_range.set_min(kRange.min);
          int_range.set_max(kRange.max);
        }
        node.in.Set(image);
        node.in_norm_rect.Set(norm_rect);
        return node.out_tensor.Get();
      }).Create());

  MP_ASSERT_OK_AND_ASSIGN(
      api3::Packet<Tensor> tensor_packet,
      runner.Run(api3::MakePacket<Image>(ReadImageRgb("input.jpg")),
                 api3::MakePacket<NormalizedRect>(
                     MakeRect(0.65f, 0.4f, 0.5f, 0.5f, 0))));

  ASSERT_TRUE(tensor_packet);
  EXPECT_THAT(TensorAndExpectedMatch(
                  tensor_packet.GetOrDie(), kRange,
                  GetRgb(GetFilePath("medium_sub_rect_keep_aspect.png"))),
              StatusIs(absl::StatusCode::kOk));
}

TEST(ImageToTensorCalculatorTest, CanBeUsedWithoutRect) {
  const Range<int> kRange = {.min = -128, .max = 127};
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For([&](GenericGraph& graph,
                                   Stream<Image> image) -> Stream<Tensor> {
                     auto& node = graph.AddNode<ImageToTensorNode>();
                     {
                       auto& opts = *node.options.Mutable();
                       opts.set_output_tensor_width(64);
                       opts.set_output_tensor_height(128);
                       opts.set_keep_aspect_ratio(true);

                       auto& int_range =
                           *opts.mutable_output_tensor_int_range();
                       int_range.set_min(kRange.min);
                       int_range.set_max(kRange.max);
                     }
                     node.in.Set(image);
                     return node.out_tensor.Get();
                   }).Create());

  MP_ASSERT_OK_AND_ASSIGN(
      api3::Packet<Tensor> tensor_packet,
      runner.Run(api3::MakePacket<Image>(ReadImageRgb("input.jpg"))));

  ASSERT_TRUE(tensor_packet);
  EXPECT_THAT(
      TensorAndExpectedMatch(tensor_packet.GetOrDie(), kRange,
                             GetRgb(GetFilePath("noop_except_range.png"))),
      StatusIs(absl::StatusCode::kOk));
}

TEST(ImageToTensorCalculatorTest, CanBeUsedWithoutGpuServiceSet) {
  auto graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input_image"
        node {
          calculator: "ImageToTensorCalculator"
          input_stream: "IMAGE:input_image"
          output_stream: "TENSORS:tensor"
          options {
            [mediapipe.ImageToTensorCalculatorOptions.ext] {
              output_tensor_float_range { min: 0.0f max: 1.0f }
            }
          }
        }
      )pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.DisallowServiceDefaultInitialization());
  MP_ASSERT_OK(graph.StartRun({}));
  auto image_frame =
      std::make_shared<ImageFrame>(ImageFormat::SRGBA, 128, 256, 4);
  Image image = Image(std::move(image_frame));
  mediapipe::Packet packet = mediapipe::MakePacket<Image>(std::move(image));
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("input_image", packet.At(Timestamp(1))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

#if !MEDIAPIPE_DISABLE_GPU && !MEDIAPIPE_METAL_ENABLED

TEST(ImageToTensorCalculatorTest,
     FailsGracefullyWhenGpuServiceNeededButNotAvailable) {
  auto graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input_image"
        node {
          calculator: "ImageToTensorCalculator"
          input_stream: "IMAGE:input_image"
          output_stream: "TENSORS:tensor"
          options {
            [mediapipe.ImageToTensorCalculatorOptions.ext] {
              output_tensor_float_range { min: 0.0f max: 1.0f }
            }
          }
        }
      )pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.DisallowServiceDefaultInitialization());
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK_AND_ASSIGN(auto context,
                          GlContext::Create(nullptr, /*create_thread=*/true));
  mediapipe::Packet packet;
  context->Run([&packet]() {
    auto image_frame =
        std::make_shared<ImageFrame>(ImageFormat::SRGBA, 128, 256, 4);
    Image image = Image(std::move(image_frame));
    // Ensure image is available on GPU to force ImageToTensorCalculator to
    // run on GPU.
    ASSERT_TRUE(image.ConvertToGpu());
    packet = MakePacket<Image>(std::move(image));
  });
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("input_image", packet.At(Timestamp(1))));
  EXPECT_THAT(graph.WaitUntilIdle(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("GPU service not available")));
}
#endif  // !MEDIAPIPE_DISABLE_GPU && !MEDIAPIPE_METAL_ENABLED

}  // namespace
}  // namespace mediapipe
