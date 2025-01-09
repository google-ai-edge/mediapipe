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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/optional.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/image_test_utils.h"

#if !MEDIAPIPE_DISABLE_GPU && !MEDIAPIPE_METAL_ENABLED
#include "mediapipe/gpu/gl_context.h"
#endif  // !MEDIAPIPE_DISABLE_GPU && !MEDIAPIPE_METAL_ENABLED

namespace mediapipe {
namespace {

constexpr char kTestDataDir[] =
    "/mediapipe/calculators/tensor/testdata/"
    "image_to_tensor/";

std::string GetFilePath(absl::string_view filename) {
  return file::JoinPath("./", kTestDataDir, filename);
}

// Image to tensor test template.
// No processing/assertions should be done after the function is invoked.
void RunTestWithInputImagePacket(
    const Packet& input_image_packet, cv::Mat expected_result, float range_min,
    float range_max, std::optional<int> tensor_width,
    std::optional<int> tensor_height, bool keep_aspect,
    absl::optional<BorderMode> border_mode,
    const mediapipe::NormalizedRect& roi, bool output_int_tensor,
    bool use_tensor_vector_output) {
  std::string border_mode_str;
  if (border_mode) {
    switch (*border_mode) {
      case BorderMode::kReplicate:
        border_mode_str = "border_mode: BORDER_REPLICATE";
        break;
      case BorderMode::kZero:
        border_mode_str = "border_mode: BORDER_ZERO";
        break;
    }
  }
  std::string output_tensor_range;
  if (output_int_tensor) {
    if (range_min < 0) {
      output_tensor_range = absl::Substitute(R"(output_tensor_int_range {
                min: $0
                max: $1
              })",
                                             static_cast<int>(range_min),
                                             static_cast<int>(range_max));
    } else {
      output_tensor_range = absl::Substitute(R"(output_tensor_uint_range {
                min: $0
                max: $1
              })",
                                             static_cast<uint>(range_min),
                                             static_cast<uint>(range_max));
    }
  } else {
    output_tensor_range = absl::Substitute(R"(output_tensor_float_range {
                min: $0
                max: $1
              })",
                                           range_min, range_max);
  }
  auto graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
          R"(
        input_stream: "input_image"
        input_stream: "roi"
        node {
          calculator: "ImageToTensorCalculator"
          input_stream: "IMAGE:input_image"
          input_stream: "NORM_RECT:roi"
          output_stream: "$0:tensor"
          options {
            [mediapipe.ImageToTensorCalculatorOptions.ext] {
              $1 # output tensor width
              $2 # output tensor height
              keep_aspect_ratio: $3
              $4 # output range
              $5 # border mode
            }
          }
        }
        )",
          /*$0=*/use_tensor_vector_output ? "TENSORS" : "TENSOR",
          /*$1=*/tensor_width.has_value()
              ? absl::StrFormat("output_tensor_width: %d", tensor_width.value())
              : "",
          /*$2=*/tensor_height.has_value()
              ? absl::StrFormat("output_tensor_height: %d",
                                tensor_height.value())
              : "",
          /*$3=*/keep_aspect ? "true" : "false",
          /*$4=*/output_tensor_range,
          /*$5=*/border_mode_str));

  std::vector<Packet> output_packets;
  tool::AddVectorSink("tensor", &graph_config, &output_packets);

  // Run the graph.
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(graph.AddPacketToInputStream("input_image", input_image_packet));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "roi",
      MakePacket<mediapipe::NormalizedRect>(std::move(roi)).At(Timestamp(0))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_THAT(output_packets, testing::SizeIs(1));

  // Get and process results.
  const Tensor* tensor;
  if (use_tensor_vector_output) {
    const std::vector<Tensor>& tensor_vec =
        output_packets[0].Get<std::vector<Tensor>>();
    ASSERT_THAT(tensor_vec, testing::SizeIs(1));
    tensor = &(tensor_vec[0]);
  } else {
    tensor = &(output_packets[0].Get<Tensor>());
  }
  const int channels = tensor->shape().dims[3];
  ASSERT_TRUE(channels == 1 || channels == 3);
  auto view = tensor->GetCpuReadView();
  cv::Mat tensor_mat;
  if (output_int_tensor) {
    if (range_min < 0) {
      EXPECT_EQ(tensor->element_type(), Tensor::ElementType::kInt8);
      tensor_mat = cv::Mat(expected_result.rows, expected_result.cols,
                           channels == 1 ? CV_8SC1 : CV_8SC3,
                           const_cast<int8_t*>(view.buffer<int8_t>()));
    } else {
      EXPECT_EQ(tensor->element_type(), Tensor::ElementType::kUInt8);
      tensor_mat = cv::Mat(expected_result.rows, expected_result.cols,
                           channels == 1 ? CV_8UC1 : CV_8UC3,
                           const_cast<uint8_t*>(view.buffer<uint8_t>()));
    }
  } else {
    EXPECT_EQ(tensor->element_type(), Tensor::ElementType::kFloat32);
    tensor_mat = cv::Mat(expected_result.rows, expected_result.cols,
                         channels == 1 ? CV_32FC1 : CV_32FC3,
                         const_cast<float*>(view.buffer<float>()));
  }

  cv::Mat result_rgb;
  auto transformation =
      GetValueRangeTransformation(range_min, range_max, 0.0f, 255.0f).value();
  tensor_mat.convertTo(result_rgb, channels == 1 ? CV_8UC1 : CV_8UC3,
                       transformation.scale, transformation.offset);

  cv::Mat diff;
  cv::absdiff(result_rgb, expected_result, diff);
  double max_val;
  cv::minMaxLoc(diff, nullptr, &max_val);
  // Expects the maximum absolute pixel-by-pixel difference is less than 5.
  EXPECT_LE(max_val, 5);

  // Fully close graph at end, otherwise calculator+tensors are destroyed
  // after calling WaitUntilDone().
  MP_ASSERT_OK(graph.CloseInputStream("input_image"));
  MP_ASSERT_OK(graph.CloseInputStream("roi"));
  MP_ASSERT_OK(graph.WaitUntilDone());
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

Packet MakeImageFramePacket(cv::Mat input) {
  ImageFrame input_image(GetImageFormat(input.channels()), input.cols,
                         input.rows, input.step, input.data, [](uint8_t*) {});
  return MakePacket<ImageFrame>(std::move(input_image)).At(Timestamp(0));
}

Packet MakeImagePacket(cv::Mat input) {
  mediapipe::Image input_image(std::make_shared<mediapipe::ImageFrame>(
      GetImageFormat(input.channels()), input.cols, input.rows, input.step,
      input.data, [](uint8_t*) {}));
  return MakePacket<mediapipe::Image>(std::move(input_image)).At(Timestamp(0));
}

enum class InputType { kImageFrame, kImage };

const std::vector<InputType> kInputTypesToTest = {InputType::kImageFrame,
                                                  InputType::kImage};

void RunTest(cv::Mat input, cv::Mat expected_result,
             std::vector<std::pair<float, float>> float_ranges,
             std::vector<std::pair<int, int>> int_ranges,
             std::optional<int> tensor_width, std::optional<int> tensor_height,
             bool keep_aspect, absl::optional<BorderMode> border_mode,
             const mediapipe::NormalizedRect& roi) {
  for (auto input_type : kInputTypesToTest) {
    for (auto float_range : float_ranges) {
      RunTestWithInputImagePacket(
          input_type == InputType::kImageFrame ? MakeImageFramePacket(input)
                                               : MakeImagePacket(input),
          expected_result, float_range.first, float_range.second, tensor_width,
          tensor_height, keep_aspect, border_mode, roi,
          /*output_int_tensor=*/false,
          /*use_tensor_vector_output=*/true);
    }
    for (auto int_range : int_ranges) {
      RunTestWithInputImagePacket(
          input_type == InputType::kImageFrame ? MakeImageFramePacket(input)
                                               : MakeImagePacket(input),
          expected_result, int_range.first, int_range.second, tensor_width,
          tensor_height, keep_aspect, border_mode, roi,
          /*output_int_tensor=*/true,
          /*use_tensor_vector_output=*/true);
    }
  }

  // Run test with single output tensor instead of std::vector<Tensor>.
  RunTestWithInputImagePacket(MakeImageFramePacket(input), expected_result, 0,
                              100, tensor_width, tensor_height, keep_aspect,
                              border_mode, roi,
                              /*output_int_tensor=*/true,
                              /*use_tensor_vector_output=*/false);
}

TEST(ImageToTensorCalculatorTest, MediumSubRectKeepAspect) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.65f);
  roi.set_y_center(0.4f);
  roi.set_width(0.5f);
  roi.set_height(0.5f);
  roi.set_rotation(0);
  RunTest(GetRgb(GetFilePath("input.jpg")),
          GetRgb(GetFilePath("medium_sub_rect_keep_aspect.png")),
          /*float_ranges=*/{{0.0f, 1.0f}},
          /*int_ranges=*/{{0, 255}, {-128, 127}},
          /*tensor_width=*/256, /*tensor_height=*/256, /*keep_aspect=*/true,
          /*border mode*/ {}, roi);
}

TEST(ImageToTensorCalculatorTest, MediumSubRectKeepAspectBorderZero) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.65f);
  roi.set_y_center(0.4f);
  roi.set_width(0.5f);
  roi.set_height(0.5f);
  roi.set_rotation(0);
  RunTest(GetRgb(GetFilePath("input.jpg")),
          GetRgb(GetFilePath("medium_sub_rect_keep_aspect_border_zero.png")),
          /*float_ranges=*/{{0.0f, 1.0f}},
          /*int_ranges=*/{{0, 255}, {-128, 127}},
          /*tensor_width=*/256, /*tensor_height=*/256, /*keep_aspect=*/true,
          BorderMode::kZero, roi);
}

TEST(ImageToTensorCalculatorTest, MediumSubRectKeepAspectWithRotation) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.65f);
  roi.set_y_center(0.4f);
  roi.set_width(0.5f);
  roi.set_height(0.5f);
  roi.set_rotation(M_PI * 90.0f / 180.0f);
  RunTest(GetRgb(GetFilePath("input.jpg")),
          GetRgb(GetFilePath("medium_sub_rect_keep_aspect_with_rotation.png")),
          /*float_ranges=*/{{0.0f, 1.0f}},
          /*int_ranges=*/{{0, 255}},
          /*tensor_width=*/256, /*tensor_height=*/256, /*keep_aspect=*/true,
          BorderMode::kReplicate, roi);
}

TEST(ImageToTensorCalculatorTest,
     MediumSubRectKeepAspectWithRotationBorderZero) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.65f);
  roi.set_y_center(0.4f);
  roi.set_width(0.5f);
  roi.set_height(0.5f);
  roi.set_rotation(M_PI * 90.0f / 180.0f);
  RunTest(GetRgb(GetFilePath("input.jpg")),
          GetRgb(GetFilePath(
              "medium_sub_rect_keep_aspect_with_rotation_border_zero.png")),
          /*float_ranges=*/{{0.0f, 1.0f}},
          /*int_ranges=*/{{0, 255}, {-128, 127}},
          /*tensor_width=*/256, /*tensor_height=*/256, /*keep_aspect=*/true,
          BorderMode::kZero, roi);
}

TEST(ImageToTensorCalculatorTest, MediumSubRectWithRotation) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.65f);
  roi.set_y_center(0.4f);
  roi.set_width(0.5f);
  roi.set_height(0.5f);
  roi.set_rotation(M_PI * -45.0f / 180.0f);
  RunTest(GetRgb(GetFilePath("input.jpg")),
          GetRgb(GetFilePath("medium_sub_rect_with_rotation.png")),
          /*float_ranges=*/{{-1.0f, 1.0f}},
          /*int_ranges=*/{{0, 255}, {-128, 127}},
          /*tensor_width=*/256, /*tensor_height=*/256, /*keep_aspect=*/false,
          BorderMode::kReplicate, roi);
}

TEST(ImageToTensorCalculatorTest, MediumSubRectWithRotationBorderZero) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.65f);
  roi.set_y_center(0.4f);
  roi.set_width(0.5f);
  roi.set_height(0.5f);
  roi.set_rotation(M_PI * -45.0f / 180.0f);
  RunTest(GetRgb(GetFilePath("input.jpg")),
          GetRgb(GetFilePath("medium_sub_rect_with_rotation_border_zero.png")),
          /*float_ranges=*/{{-1.0f, 1.0f}},
          /*int_ranges=*/{{0, 255}, {-128, 127}},
          /*tensor_width=*/256, /*tensor_height=*/256, /*keep_aspect=*/false,
          BorderMode::kZero, roi);
}

TEST(ImageToTensorCalculatorTest, LargeSubRect) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.5f);
  roi.set_height(1.1f);
  roi.set_rotation(0);
  RunTest(GetRgb(GetFilePath("input.jpg")),
          GetRgb(GetFilePath("large_sub_rect.png")),
          /*float_ranges=*/{{0.0f, 1.0f}},
          /*int_ranges=*/{{0, 255}},
          /*tensor_width=*/128, /*tensor_height=*/128, /*keep_aspect=*/false,
          BorderMode::kReplicate, roi);
}

TEST(ImageToTensorCalculatorTest, LargeSubRectBorderZero) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.5f);
  roi.set_height(1.1f);
  roi.set_rotation(0);
  RunTest(GetRgb(GetFilePath("input.jpg")),
          GetRgb(GetFilePath("large_sub_rect_border_zero.png")),
          /*float_ranges=*/{{0.0f, 1.0f}},
          /*int_ranges=*/{{0, 255}, {-128, 127}},
          /*tensor_width=*/128, /*tensor_height=*/128, /*keep_aspect=*/false,
          BorderMode::kZero, roi);
}

TEST(ImageToTensorCalculatorTest, LargeSubRectKeepAspect) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.5f);
  roi.set_height(1.1f);
  roi.set_rotation(0);
  RunTest(GetRgb(GetFilePath("input.jpg")),
          GetRgb(GetFilePath("large_sub_rect_keep_aspect.png")),
          /*float_ranges=*/{{0.0f, 1.0f}},
          /*int_ranges=*/{{0, 255}, {-128, 127}},
          /*tensor_width=*/128, /*tensor_height=*/128, /*keep_aspect=*/true,
          BorderMode::kReplicate, roi);
}

TEST(ImageToTensorCalculatorTest, LargeSubRectKeepAspectBorderZero) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.5f);
  roi.set_height(1.1f);
  roi.set_rotation(0);
  RunTest(GetRgb(GetFilePath("input.jpg")),
          GetRgb(GetFilePath("large_sub_rect_keep_aspect_border_zero.png")),
          /*float_ranges=*/{{0.0f, 1.0f}},
          /*int_ranges=*/{{0, 255}, {-128, 127}},
          /*tensor_width=*/128, /*tensor_height=*/128, /*keep_aspect=*/true,
          BorderMode::kZero, roi);
}

TEST(ImageToTensorCalculatorTest, LargeSubRectKeepAspectWithRotation) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.5f);
  roi.set_height(1.1f);
  roi.set_rotation(M_PI * -15.0f / 180.0f);
  RunTest(GetRgba(GetFilePath("input.jpg")),
          GetRgb(GetFilePath("large_sub_rect_keep_aspect_with_rotation.png")),
          /*float_ranges=*/{{0.0f, 1.0f}},
          /*int_ranges=*/{{0, 255}, {-128, 127}},
          /*tensor_width=*/128, /*tensor_height=*/128, /*keep_aspect=*/true,
          /*border_mode=*/{}, roi);
}

TEST(ImageToTensorCalculatorTest, LargeSubRectKeepAspectWithRotationGray) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.5f);
  roi.set_height(1.1f);
  roi.set_rotation(M_PI * -15.0f / 180.0f);
  RunTest(GetGray(GetFilePath("input.jpg")),
          GetGray(GetFilePath("large_sub_rect_keep_aspect_with_rotation.png")),
          /*float_ranges=*/{{0.0f, 1.0f}},
          /*int_ranges=*/{{0, 255}, {-128, 127}},
          /*tensor_width=*/128, /*tensor_height=*/128, /*keep_aspect=*/true,
          /*border_mode=*/{}, roi);
}

TEST(ImageToTensorCalculatorTest,
     LargeSubRectKeepAspectWithRotationBorderZero) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.5f);
  roi.set_height(1.1f);
  roi.set_rotation(M_PI * -15.0f / 180.0f);
  RunTest(GetRgba(GetFilePath("input.jpg")),
          GetRgb(GetFilePath(
              "large_sub_rect_keep_aspect_with_rotation_border_zero.png")),
          /*float_ranges=*/{{0.0f, 1.0f}},
          /*int_ranges=*/{{0, 255}},
          /*tensor_width=*/128, /*tensor_height=*/128, /*keep_aspect=*/true,
          /*border_mode=*/BorderMode::kZero, roi);
}

TEST(ImageToTensorCalculatorTest,
     LargeSubRectKeepAspectWithRotationBorderZeroGray) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.5f);
  roi.set_height(1.1f);
  roi.set_rotation(M_PI * -15.0f / 180.0f);
  RunTest(GetGray(GetFilePath("input.jpg")),
          GetGray(GetFilePath(
              "large_sub_rect_keep_aspect_with_rotation_border_zero.png")),
          /*float_ranges=*/{{0.0f, 1.0f}},
          /*int_ranges=*/{{0, 255}},
          /*tensor_width=*/128, /*tensor_height=*/128, /*keep_aspect=*/true,
          /*border_mode=*/BorderMode::kZero, roi);
}

TEST(ImageToTensorCalculatorTest, NoOpExceptRange) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.0f);
  roi.set_height(1.0f);
  roi.set_rotation(0);
  RunTest(GetRgba(GetFilePath("input.jpg")),
          GetRgb(GetFilePath("noop_except_range.png")),
          /*float_ranges=*/{{0.0f, 1.0f}},
          /*int_ranges=*/{{0, 255}, {-128, 127}},
          /*tensor_width=*/64, /*tensor_height=*/128, /*keep_aspect=*/true,
          BorderMode::kReplicate, roi);
}

TEST(ImageToTensorCalculatorTest, NoOpExceptRangeBorderZero) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.0f);
  roi.set_height(1.0f);
  roi.set_rotation(0);
  RunTest(GetRgba(GetFilePath("input.jpg")),
          GetRgb(GetFilePath("noop_except_range.png")),
          /*float_ranges=*/{{0.0f, 1.0f}},
          /*int_ranges=*/{{0, 255}, {-128, 127}},
          /*tensor_width=*/64, /*tensor_height=*/128, /*keep_aspect=*/true,
          BorderMode::kZero, roi);
}

TEST(ImageToTensorCalculatorTest, NoOpExceptRangeAndUseInputImageDims) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.0f);
  roi.set_height(1.0f);
  RunTest(GetRgb(GetFilePath("input.jpg")),
          GetRgb(GetFilePath("noop_except_range.png")),
          /*float_ranges=*/{{-1.0f, 1.0f}},
          /*int_ranges=*/{{0, 255}, {-128, 127}},
          /*tensor_width=*/std::nullopt, /*tensor_height=*/std::nullopt,
          /*keep_aspect=*/false, BorderMode::kZero, roi);
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
  Packet packet = MakePacket<Image>(std::move(image));
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
  Packet packet;
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
