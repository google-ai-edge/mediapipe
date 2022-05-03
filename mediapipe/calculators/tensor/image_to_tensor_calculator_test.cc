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

#include <cmath>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

cv::Mat GetRgb(absl::string_view path) {
  cv::Mat bgr = cv::imread(file::JoinPath("./", path));
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return rgb;
}

cv::Mat GetRgba(absl::string_view path) {
  cv::Mat bgr = cv::imread(file::JoinPath("./", path));
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGBA);
  return rgb;
}

// Image to tensor test template.
// No processing/assertions should be done after the function is invoked.
void RunTestWithInputImagePacket(const Packet& input_image_packet,
                                 cv::Mat expected_result, float range_min,
                                 float range_max, int tensor_width,
                                 int tensor_height, bool keep_aspect,
                                 absl::optional<BorderMode> border_mode,
                                 const mediapipe::NormalizedRect& roi,
                                 bool output_int_tensor) {
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
  auto graph_config = mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
      absl::Substitute(R"(
        input_stream: "input_image"
        input_stream: "roi"
        node {
          calculator: "ImageToTensorCalculator"
          input_stream: "IMAGE:input_image"
          input_stream: "NORM_RECT:roi"
          output_stream: "TENSORS:tensor"
          options {
            [mediapipe.ImageToTensorCalculatorOptions.ext] {
              output_tensor_width: $0
              output_tensor_height: $1
              keep_aspect_ratio: $2
              $3 # output range
              $4 # border mode
            }
          }
        }
        )",
                       /*$0=*/tensor_width,
                       /*$1=*/tensor_height,
                       /*$2=*/keep_aspect ? "true" : "false",
                       /*$3=*/output_tensor_range,
                       /*$4=*/border_mode_str));

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
  const std::vector<Tensor>& tensor_vec =
      output_packets[0].Get<std::vector<Tensor>>();
  ASSERT_THAT(tensor_vec, testing::SizeIs(1));

  const Tensor& tensor = tensor_vec[0];
  auto view = tensor.GetCpuReadView();
  cv::Mat tensor_mat;
  if (output_int_tensor) {
    if (range_min < 0) {
      EXPECT_EQ(tensor.element_type(), Tensor::ElementType::kInt8);
      tensor_mat = cv::Mat(tensor_height, tensor_width, CV_8SC3,
                           const_cast<int8*>(view.buffer<int8>()));
    } else {
      EXPECT_EQ(tensor.element_type(), Tensor::ElementType::kUInt8);
      tensor_mat = cv::Mat(tensor_height, tensor_width, CV_8UC3,
                           const_cast<uint8*>(view.buffer<uint8>()));
    }
  } else {
    EXPECT_EQ(tensor.element_type(), Tensor::ElementType::kFloat32);
    tensor_mat = cv::Mat(tensor_height, tensor_width, CV_32FC3,
                         const_cast<float*>(view.buffer<float>()));
  }

  cv::Mat result_rgb;
  auto transformation =
      GetValueRangeTransformation(range_min, range_max, 0.0f, 255.0f).value();
  tensor_mat.convertTo(result_rgb, CV_8UC3, transformation.scale,
                       transformation.offset);

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

Packet MakeImageFramePacket(cv::Mat input) {
  ImageFrame input_image(
      input.channels() == 4 ? ImageFormat::SRGBA : ImageFormat::SRGB,
      input.cols, input.rows, input.step, input.data, [](uint8*) {});
  return MakePacket<ImageFrame>(std::move(input_image)).At(Timestamp(0));
}

Packet MakeImagePacket(cv::Mat input) {
  mediapipe::Image input_image(std::make_shared<mediapipe::ImageFrame>(
      input.channels() == 4 ? ImageFormat::SRGBA : ImageFormat::SRGB,
      input.cols, input.rows, input.step, input.data, [](uint8*) {}));
  return MakePacket<mediapipe::Image>(std::move(input_image)).At(Timestamp(0));
}

enum class InputType { kImageFrame, kImage };

const std::vector<InputType> kInputTypesToTest = {InputType::kImageFrame,
                                                  InputType::kImage};

void RunTest(cv::Mat input, cv::Mat expected_result,
             std::vector<std::pair<float, float>> float_ranges,
             std::vector<std::pair<int, int>> int_ranges, int tensor_width,
             int tensor_height, bool keep_aspect,
             absl::optional<BorderMode> border_mode,
             const mediapipe::NormalizedRect& roi) {
  for (auto input_type : kInputTypesToTest) {
    for (auto float_range : float_ranges) {
      RunTestWithInputImagePacket(
          input_type == InputType::kImageFrame ? MakeImageFramePacket(input)
                                               : MakeImagePacket(input),
          expected_result, float_range.first, float_range.second, tensor_width,
          tensor_height, keep_aspect, border_mode, roi,
          /*output_int_tensor=*/false);
    }
    for (auto int_range : int_ranges) {
      RunTestWithInputImagePacket(
          input_type == InputType::kImageFrame ? MakeImageFramePacket(input)
                                               : MakeImagePacket(input),
          expected_result, int_range.first, int_range.second, tensor_width,
          tensor_height, keep_aspect, border_mode, roi,
          /*output_int_tensor=*/true);
    }
  }
}

TEST(ImageToTensorCalculatorTest, MediumSubRectKeepAspect) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.65f);
  roi.set_y_center(0.4f);
  roi.set_width(0.5f);
  roi.set_height(0.5f);
  roi.set_rotation(0);
  RunTest(
      GetRgb("/mediapipe/calculators/"
             "tensor/testdata/image_to_tensor/input.jpg"),
      GetRgb("/mediapipe/calculators/"
             "tensor/testdata/image_to_tensor/medium_sub_rect_keep_aspect.png"),
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
  RunTest(GetRgb("/mediapipe/calculators/"
                 "tensor/testdata/image_to_tensor/input.jpg"),
          GetRgb("/mediapipe/calculators/"
                 "tensor/testdata/image_to_tensor/"
                 "medium_sub_rect_keep_aspect_border_zero.png"),
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
  RunTest(GetRgb("/mediapipe/calculators/"
                 "tensor/testdata/image_to_tensor/input.jpg"),
          GetRgb("/mediapipe/calculators/"
                 "tensor/testdata/image_to_tensor/"
                 "medium_sub_rect_keep_aspect_with_rotation.png"),
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
  RunTest(GetRgb("/mediapipe/calculators/"
                 "tensor/testdata/image_to_tensor/input.jpg"),
          GetRgb("/mediapipe/calculators/"
                 "tensor/testdata/image_to_tensor/"
                 "medium_sub_rect_keep_aspect_with_rotation_border_zero.png"),
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
  RunTest(
      GetRgb("/mediapipe/calculators/"
             "tensor/testdata/image_to_tensor/input.jpg"),
      GetRgb(
          "/mediapipe/calculators/"
          "tensor/testdata/image_to_tensor/medium_sub_rect_with_rotation.png"),
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
  RunTest(GetRgb("/mediapipe/calculators/"
                 "tensor/testdata/image_to_tensor/input.jpg"),
          GetRgb("/mediapipe/calculators/"
                 "tensor/testdata/image_to_tensor/"
                 "medium_sub_rect_with_rotation_border_zero.png"),
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
  RunTest(GetRgb("/mediapipe/calculators/"
                 "tensor/testdata/image_to_tensor/input.jpg"),
          GetRgb("/mediapipe/calculators/"
                 "tensor/testdata/image_to_tensor/large_sub_rect.png"),
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
  RunTest(
      GetRgb("/mediapipe/calculators/"
             "tensor/testdata/image_to_tensor/input.jpg"),
      GetRgb("/mediapipe/calculators/"
             "tensor/testdata/image_to_tensor/large_sub_rect_border_zero.png"),
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
  RunTest(
      GetRgb("/mediapipe/calculators/"
             "tensor/testdata/image_to_tensor/input.jpg"),
      GetRgb("/mediapipe/calculators/"
             "tensor/testdata/image_to_tensor/large_sub_rect_keep_aspect.png"),
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
  RunTest(GetRgb("/mediapipe/calculators/"
                 "tensor/testdata/image_to_tensor/input.jpg"),
          GetRgb("/mediapipe/calculators/"
                 "tensor/testdata/image_to_tensor/"
                 "large_sub_rect_keep_aspect_border_zero.png"),
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
  RunTest(GetRgba("/mediapipe/calculators/"
                  "tensor/testdata/image_to_tensor/input.jpg"),
          GetRgb("/mediapipe/calculators/"
                 "tensor/testdata/image_to_tensor/"
                 "large_sub_rect_keep_aspect_with_rotation.png"),
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
  RunTest(GetRgba("/mediapipe/calculators/"
                  "tensor/testdata/image_to_tensor/input.jpg"),
          GetRgb("/mediapipe/calculators/"
                 "tensor/testdata/image_to_tensor/"
                 "large_sub_rect_keep_aspect_with_rotation_border_zero.png"),
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
  RunTest(GetRgba("/mediapipe/calculators/"
                  "tensor/testdata/image_to_tensor/input.jpg"),
          GetRgb("/mediapipe/calculators/"
                 "tensor/testdata/image_to_tensor/noop_except_range.png"),
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
  RunTest(GetRgba("/mediapipe/calculators/"
                  "tensor/testdata/image_to_tensor/input.jpg"),
          GetRgb("/mediapipe/calculators/"
                 "tensor/testdata/image_to_tensor/noop_except_range.png"),
          /*float_ranges=*/{{0.0f, 1.0f}},
          /*int_ranges=*/{{0, 255}, {-128, 127}},
          /*tensor_width=*/64, /*tensor_height=*/128, /*keep_aspect=*/true,
          BorderMode::kZero, roi);
}

}  // namespace
}  // namespace mediapipe
