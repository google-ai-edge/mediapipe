// Copyright 2021 The MediaPipe Authors.
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
#include "mediapipe/calculators/image/affine_transformation.h"
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
  cv::Mat rgb(bgr.rows, bgr.cols, CV_8UC3);
  int from_to[] = {0, 2, 1, 1, 2, 0};
  cv::mixChannels(&bgr, 1, &rgb, 1, from_to, 3);
  return rgb;
}

cv::Mat GetRgba(absl::string_view path) {
  cv::Mat bgr = cv::imread(file::JoinPath("./", path));
  cv::Mat rgba(bgr.rows, bgr.cols, CV_8UC4, cv::Scalar(0, 0, 0, 0));
  int from_to[] = {0, 2, 1, 1, 2, 0};
  cv::mixChannels(&bgr, 1, &bgr, 1, from_to, 3);
  return bgr;
}

// Test template.
// No processing/assertions should be done after the function is invoked.
void RunTest(const std::string& graph_text, const std::string& tag,
             const cv::Mat& input, cv::Mat expected_result,
             float similarity_threshold, std::array<float, 16> matrix,
             int out_width, int out_height,
             absl::optional<AffineTransformation::BorderMode> border_mode) {
  std::string border_mode_str;
  if (border_mode) {
    switch (*border_mode) {
      case AffineTransformation::BorderMode::kReplicate:
        border_mode_str = "border_mode: BORDER_REPLICATE";
        break;
      case AffineTransformation::BorderMode::kZero:
        border_mode_str = "border_mode: BORDER_ZERO";
        break;
    }
  }
  auto graph_config = mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
      absl::Substitute(graph_text, /*$0=*/border_mode_str));

  std::vector<Packet> output_packets;
  tool::AddVectorSink("output_image", &graph_config, &output_packets);

  // Run the graph.
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  ImageFrame input_image(
      input.channels() == 4 ? ImageFormat::SRGBA : ImageFormat::SRGB,
      input.cols, input.rows, input.step, input.data, [](uint8*) {});
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_image",
      MakePacket<ImageFrame>(std::move(input_image)).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "matrix",
      MakePacket<std::array<float, 16>>(std::move(matrix)).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "output_size", MakePacket<std::pair<int, int>>(
                         std::pair<int, int>(out_width, out_height))
                         .At(Timestamp(0))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_THAT(output_packets, testing::SizeIs(1));

  // Get and process results.
  const ImageFrame& out_frame = output_packets[0].Get<ImageFrame>();
  cv::Mat result = formats::MatView(&out_frame);
  double similarity =
      1.0 - cv::norm(result, expected_result, cv::NORM_RELATIVE | cv::NORM_L2);
  EXPECT_GE(similarity, similarity_threshold);

  // Fully close graph at end, otherwise calculator+tensors are destroyed
  // after calling WaitUntilDone().
  MP_ASSERT_OK(graph.CloseInputStream("input_image"));
  MP_ASSERT_OK(graph.CloseInputStream("matrix"));
  MP_ASSERT_OK(graph.CloseInputStream("output_size"));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

enum class InputType { kImageFrame, kImage };

// Similarity is checked against OpenCV results always, and due to differences
// on how OpenCV and GL treats pixels there are two thresholds.
// TODO: update to have just one threshold when OpenCV
//                    implementation is updated.
struct SimilarityConfig {
  double threshold_on_cpu;
  double threshold_on_gpu;
};

void RunTest(cv::Mat input, cv::Mat expected_result,
             const SimilarityConfig& similarity, std::array<float, 16> matrix,
             int out_width, int out_height,
             absl::optional<AffineTransformation::BorderMode> border_mode) {
  RunTest(R"(
        input_stream: "input_image"
        input_stream: "output_size"
        input_stream: "matrix"
        node {
          calculator: "WarpAffineCalculatorCpu"
          input_stream: "IMAGE:input_image"
          input_stream: "MATRIX:matrix"
          input_stream: "OUTPUT_SIZE:output_size"
          output_stream: "IMAGE:output_image"
          options {
            [mediapipe.WarpAffineCalculatorOptions.ext] {
              $0 # border mode
            }
          }
        }
        )",
          "cpu", input, expected_result, similarity.threshold_on_cpu, matrix,
          out_width, out_height, border_mode);

  RunTest(R"(
        input_stream: "input_image"
        input_stream: "output_size"
        input_stream: "matrix"
        node {
          calculator: "ToImageCalculator"
          input_stream: "IMAGE_CPU:input_image"
          output_stream: "IMAGE:input_image_unified"
        }
        node {
          calculator: "WarpAffineCalculator"
          input_stream: "IMAGE:input_image_unified"
          input_stream: "MATRIX:matrix"
          input_stream: "OUTPUT_SIZE:output_size"
          output_stream: "IMAGE:output_image_unified"
          options {
            [mediapipe.WarpAffineCalculatorOptions.ext] {
              $0 # border mode
            }
          }
        }
        node {
          calculator: "FromImageCalculator"
          input_stream: "IMAGE:output_image_unified"
          output_stream: "IMAGE_CPU:output_image"
        }
        )",
          "cpu_image", input, expected_result, similarity.threshold_on_cpu,
          matrix, out_width, out_height, border_mode);

  RunTest(R"(
        input_stream: "input_image"
        input_stream: "output_size"
        input_stream: "matrix"
        node {
          calculator: "ImageFrameToGpuBufferCalculator"
          input_stream: "input_image"
          output_stream: "input_image_gpu"
        }
        node {
          calculator: "WarpAffineCalculatorGpu"
          input_stream: "IMAGE:input_image_gpu"
          input_stream: "MATRIX:matrix"
          input_stream: "OUTPUT_SIZE:output_size"
          output_stream: "IMAGE:output_image_gpu"
          options {
            [mediapipe.WarpAffineCalculatorOptions.ext] {
              $0 # border mode
              gpu_origin: TOP_LEFT
            }
          }
        }
        node {
          calculator: "GpuBufferToImageFrameCalculator"
          input_stream: "output_image_gpu"
          output_stream: "output_image"
        }
        )",
          "gpu", input, expected_result, similarity.threshold_on_gpu, matrix,
          out_width, out_height, border_mode);

  RunTest(R"(
        input_stream: "input_image"
        input_stream: "output_size"
        input_stream: "matrix"
        node {
          calculator: "ImageFrameToGpuBufferCalculator"
          input_stream: "input_image"
          output_stream: "input_image_gpu"
        }
        node {
          calculator: "ToImageCalculator"
          input_stream: "IMAGE_GPU:input_image_gpu"
          output_stream: "IMAGE:input_image_unified"
        }
        node {
          calculator: "WarpAffineCalculator"
          input_stream: "IMAGE:input_image_unified"
          input_stream: "MATRIX:matrix"
          input_stream: "OUTPUT_SIZE:output_size"
          output_stream: "IMAGE:output_image_unified"
          options {
            [mediapipe.WarpAffineCalculatorOptions.ext] {
              $0 # border mode
              gpu_origin: TOP_LEFT
            }
          }
        }
        node {
          calculator: "FromImageCalculator"
          input_stream: "IMAGE:output_image_unified"
          output_stream: "IMAGE_GPU:output_image_gpu"
        }
        node {
          calculator: "GpuBufferToImageFrameCalculator"
          input_stream: "output_image_gpu"
          output_stream: "output_image"
        }
        )",
          "gpu_image", input, expected_result, similarity.threshold_on_gpu,
          matrix, out_width, out_height, border_mode);
}

std::array<float, 16> GetMatrix(cv::Mat input, mediapipe::NormalizedRect roi,
                                bool keep_aspect_ratio, int out_width,
                                int out_height) {
  std::array<float, 16> transform_mat;
  mediapipe::RotatedRect roi_absolute =
      mediapipe::GetRoi(input.cols, input.rows, roi);
  mediapipe::PadRoi(out_width, out_height, keep_aspect_ratio, &roi_absolute)
      .IgnoreError();
  mediapipe::GetRotatedSubRectToRectTransformMatrix(
      roi_absolute, input.cols, input.rows,
      /*flip_horizontaly=*/false, &transform_mat);
  return transform_mat;
}

TEST(WarpAffineCalculatorTest, MediumSubRectKeepAspect) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.65f);
  roi.set_y_center(0.4f);
  roi.set_width(0.5f);
  roi.set_height(0.5f);
  roi.set_rotation(0);
  auto input = GetRgb(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/input.jpg");
  auto expected_output = GetRgb(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/medium_sub_rect_keep_aspect.png");
  int out_width = 256;
  int out_height = 256;
  bool keep_aspect_ratio = true;
  std::optional<AffineTransformation::BorderMode> border_mode = {};
  RunTest(input, expected_output,
          {.threshold_on_cpu = 0.99, .threshold_on_gpu = 0.82},
          GetMatrix(input, roi, keep_aspect_ratio, out_width, out_height),
          out_width, out_height, border_mode);
}

TEST(WarpAffineCalculatorTest, MediumSubRectKeepAspectBorderZero) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.65f);
  roi.set_y_center(0.4f);
  roi.set_width(0.5f);
  roi.set_height(0.5f);
  roi.set_rotation(0);
  auto input = GetRgb(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/input.jpg");
  auto expected_output = GetRgb(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/"
      "medium_sub_rect_keep_aspect_border_zero.png");
  int out_width = 256;
  int out_height = 256;
  bool keep_aspect_ratio = true;
  std::optional<AffineTransformation::BorderMode> border_mode =
      AffineTransformation::BorderMode::kZero;
  RunTest(input, expected_output,
          {.threshold_on_cpu = 0.99, .threshold_on_gpu = 0.81},
          GetMatrix(input, roi, keep_aspect_ratio, out_width, out_height),
          out_width, out_height, border_mode);
}

TEST(WarpAffineCalculatorTest, MediumSubRectKeepAspectWithRotation) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.65f);
  roi.set_y_center(0.4f);
  roi.set_width(0.5f);
  roi.set_height(0.5f);
  roi.set_rotation(M_PI * 90.0f / 180.0f);
  auto input = GetRgb(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/input.jpg");
  auto expected_output = GetRgb(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/"
      "medium_sub_rect_keep_aspect_with_rotation.png");
  int out_width = 256;
  int out_height = 256;
  bool keep_aspect_ratio = true;
  std::optional<AffineTransformation::BorderMode> border_mode =
      AffineTransformation::BorderMode::kReplicate;
  RunTest(input, expected_output,
          {.threshold_on_cpu = 0.99, .threshold_on_gpu = 0.77},
          GetMatrix(input, roi, keep_aspect_ratio, out_width, out_height),
          out_width, out_height, border_mode);
}

TEST(WarpAffineCalculatorTest, MediumSubRectKeepAspectWithRotationBorderZero) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.65f);
  roi.set_y_center(0.4f);
  roi.set_width(0.5f);
  roi.set_height(0.5f);
  roi.set_rotation(M_PI * 90.0f / 180.0f);
  auto input = GetRgb(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/input.jpg");
  auto expected_output = GetRgb(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/"
      "medium_sub_rect_keep_aspect_with_rotation_border_zero.png");
  int out_width = 256;
  int out_height = 256;
  bool keep_aspect_ratio = true;
  std::optional<AffineTransformation::BorderMode> border_mode =
      AffineTransformation::BorderMode::kZero;
  RunTest(input, expected_output,
          {.threshold_on_cpu = 0.99, .threshold_on_gpu = 0.75},
          GetMatrix(input, roi, keep_aspect_ratio, out_width, out_height),
          out_width, out_height, border_mode);
}

TEST(WarpAffineCalculatorTest, MediumSubRectWithRotation) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.65f);
  roi.set_y_center(0.4f);
  roi.set_width(0.5f);
  roi.set_height(0.5f);
  roi.set_rotation(M_PI * -45.0f / 180.0f);
  auto input = GetRgb(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/input.jpg");
  auto expected_output = GetRgb(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/medium_sub_rect_with_rotation.png");
  int out_width = 256;
  int out_height = 256;
  bool keep_aspect_ratio = false;
  std::optional<AffineTransformation::BorderMode> border_mode =
      AffineTransformation::BorderMode::kReplicate;
  RunTest(input, expected_output,
          {.threshold_on_cpu = 0.99, .threshold_on_gpu = 0.81},
          GetMatrix(input, roi, keep_aspect_ratio, out_width, out_height),
          out_width, out_height, border_mode);
}

TEST(WarpAffineCalculatorTest, MediumSubRectWithRotationBorderZero) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.65f);
  roi.set_y_center(0.4f);
  roi.set_width(0.5f);
  roi.set_height(0.5f);
  roi.set_rotation(M_PI * -45.0f / 180.0f);
  auto input = GetRgb(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/input.jpg");
  auto expected_output = GetRgb(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/"
      "medium_sub_rect_with_rotation_border_zero.png");
  int out_width = 256;
  int out_height = 256;
  bool keep_aspect_ratio = false;
  std::optional<AffineTransformation::BorderMode> border_mode =
      AffineTransformation::BorderMode::kZero;
  RunTest(input, expected_output,
          {.threshold_on_cpu = 0.99, .threshold_on_gpu = 0.80},
          GetMatrix(input, roi, keep_aspect_ratio, out_width, out_height),
          out_width, out_height, border_mode);
}

TEST(WarpAffineCalculatorTest, LargeSubRect) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.5f);
  roi.set_height(1.1f);
  roi.set_rotation(0);
  auto input = GetRgb(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/input.jpg");
  auto expected_output = GetRgb(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/large_sub_rect.png");
  int out_width = 128;
  int out_height = 128;
  bool keep_aspect_ratio = false;
  std::optional<AffineTransformation::BorderMode> border_mode =
      AffineTransformation::BorderMode::kReplicate;
  RunTest(input, expected_output,
          {.threshold_on_cpu = 0.99, .threshold_on_gpu = 0.95},
          GetMatrix(input, roi, keep_aspect_ratio, out_width, out_height),
          out_width, out_height, border_mode);
}

TEST(WarpAffineCalculatorTest, LargeSubRectBorderZero) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.5f);
  roi.set_height(1.1f);
  roi.set_rotation(0);
  auto input = GetRgb(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/input.jpg");
  auto expected_output = GetRgb(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/large_sub_rect_border_zero.png");
  int out_width = 128;
  int out_height = 128;
  bool keep_aspect_ratio = false;
  std::optional<AffineTransformation::BorderMode> border_mode =
      AffineTransformation::BorderMode::kZero;
  RunTest(input, expected_output,
          {.threshold_on_cpu = 0.99, .threshold_on_gpu = 0.92},
          GetMatrix(input, roi, keep_aspect_ratio, out_width, out_height),
          out_width, out_height, border_mode);
}

TEST(WarpAffineCalculatorTest, LargeSubRectKeepAspect) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.5f);
  roi.set_height(1.1f);
  roi.set_rotation(0);
  auto input = GetRgb(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/input.jpg");
  auto expected_output = GetRgb(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/large_sub_rect_keep_aspect.png");
  int out_width = 128;
  int out_height = 128;
  bool keep_aspect_ratio = true;
  std::optional<AffineTransformation::BorderMode> border_mode =
      AffineTransformation::BorderMode::kReplicate;
  RunTest(input, expected_output,
          {.threshold_on_cpu = 0.99, .threshold_on_gpu = 0.97},
          GetMatrix(input, roi, keep_aspect_ratio, out_width, out_height),
          out_width, out_height, border_mode);
}

TEST(WarpAffineCalculatorTest, LargeSubRectKeepAspectBorderZero) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.5f);
  roi.set_height(1.1f);
  roi.set_rotation(0);
  auto input = GetRgb(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/input.jpg");
  auto expected_output = GetRgb(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/"
      "large_sub_rect_keep_aspect_border_zero.png");
  int out_width = 128;
  int out_height = 128;
  bool keep_aspect_ratio = true;
  std::optional<AffineTransformation::BorderMode> border_mode =
      AffineTransformation::BorderMode::kZero;
  RunTest(input, expected_output,
          {.threshold_on_cpu = 0.99, .threshold_on_gpu = 0.97},
          GetMatrix(input, roi, keep_aspect_ratio, out_width, out_height),
          out_width, out_height, border_mode);
}

TEST(WarpAffineCalculatorTest, LargeSubRectKeepAspectWithRotation) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.5f);
  roi.set_height(1.1f);
  roi.set_rotation(M_PI * -15.0f / 180.0f);
  auto input = GetRgba(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/input.jpg");
  auto expected_output = GetRgba(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/"
      "large_sub_rect_keep_aspect_with_rotation.png");
  int out_width = 128;
  int out_height = 128;
  bool keep_aspect_ratio = true;
  std::optional<AffineTransformation::BorderMode> border_mode = {};
  RunTest(input, expected_output,
          {.threshold_on_cpu = 0.99, .threshold_on_gpu = 0.91},
          GetMatrix(input, roi, keep_aspect_ratio, out_width, out_height),
          out_width, out_height, border_mode);
}

TEST(WarpAffineCalculatorTest, LargeSubRectKeepAspectWithRotationBorderZero) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.5f);
  roi.set_height(1.1f);
  roi.set_rotation(M_PI * -15.0f / 180.0f);
  auto input = GetRgba(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/input.jpg");
  auto expected_output = GetRgba(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/"
      "large_sub_rect_keep_aspect_with_rotation_border_zero.png");
  int out_width = 128;
  int out_height = 128;
  bool keep_aspect_ratio = true;
  std::optional<AffineTransformation::BorderMode> border_mode =
      AffineTransformation::BorderMode::kZero;
  RunTest(input, expected_output,
          {.threshold_on_cpu = 0.99, .threshold_on_gpu = 0.88},
          GetMatrix(input, roi, keep_aspect_ratio, out_width, out_height),
          out_width, out_height, border_mode);
}

TEST(WarpAffineCalculatorTest, NoOp) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.0f);
  roi.set_height(1.0f);
  roi.set_rotation(0);
  auto input = GetRgba(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/input.jpg");
  auto expected_output = GetRgba(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/noop_except_range.png");
  int out_width = 64;
  int out_height = 128;
  bool keep_aspect_ratio = true;
  std::optional<AffineTransformation::BorderMode> border_mode =
      AffineTransformation::BorderMode::kReplicate;
  RunTest(input, expected_output,
          {.threshold_on_cpu = 0.99, .threshold_on_gpu = 0.99},
          GetMatrix(input, roi, keep_aspect_ratio, out_width, out_height),
          out_width, out_height, border_mode);
}

TEST(WarpAffineCalculatorTest, NoOpBorderZero) {
  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.0f);
  roi.set_height(1.0f);
  roi.set_rotation(0);
  auto input = GetRgba(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/input.jpg");
  auto expected_output = GetRgba(
      "/mediapipe/calculators/"
      "tensor/testdata/image_to_tensor/noop_except_range.png");
  int out_width = 64;
  int out_height = 128;
  bool keep_aspect_ratio = true;
  std::optional<AffineTransformation::BorderMode> border_mode =
      AffineTransformation::BorderMode::kZero;
  RunTest(input, expected_output,
          {.threshold_on_cpu = 0.99, .threshold_on_gpu = 0.99},
          GetMatrix(input, roi, keep_aspect_ratio, out_width, out_height),
          out_width, out_height, border_mode);
}

}  // namespace
}  // namespace mediapipe
