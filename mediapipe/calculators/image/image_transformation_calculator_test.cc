#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/gpu/multi_pool.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/googletest.h"
#include "testing/base/public/gunit.h"
#include "third_party/OpenCV/core.hpp"  // IWYU pragma: keep
#include "third_party/OpenCV/core/base.hpp"
#include "third_party/OpenCV/core/hal/interface.h"
#include "third_party/OpenCV/core/mat.hpp"
#include "third_party/OpenCV/core/types.hpp"

namespace mediapipe {

namespace {

template <typename T, typename U>
absl::flat_hash_set<T> computeUniqueValues(const cv::Mat& mat) {
  // Compute the unique values in cv::Mat
  absl::flat_hash_set<T> unique_values;
  for (int i = 0; i < mat.rows; i++) {
    for (int j = 0; j < mat.cols; j++) {
      unique_values.insert(mat.at<U>(i, j));
    }
  }
  return unique_values;
}

TEST(ImageTransformationCalculatorTest, NearestNeighborResizing) {
  cv::Mat input_mat;
  cv::cvtColor(cv::imread(file::JoinPath("./",
                                         "/mediapipe/calculators/"
                                         "image/testdata/binary_mask.png")),
               input_mat, cv::COLOR_BGR2GRAY);
  Packet input_image_packet = MakePacket<ImageFrame>(
      ImageFormat::GRAY8, input_mat.size().width, input_mat.size().height);
  input_mat.copyTo(formats::MatView(&(input_image_packet.Get<ImageFrame>())));

  std::vector<std::pair<int, int>> output_dims{
      {256, 333}, {512, 512}, {1024, 1024}};

  for (auto& output_dim : output_dims) {
    Packet input_output_dim_packet =
        MakePacket<std::pair<int, int>>(output_dim);
    std::vector<std::string> scale_modes{"FIT", "STRETCH"};
    for (const auto& scale_mode : scale_modes) {
      CalculatorGraphConfig::Node node_config =
          ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
              absl::Substitute(R"(
          calculator: "ImageTransformationCalculator"
          input_stream: "IMAGE:input_image"
          input_stream: "OUTPUT_DIMENSIONS:image_size"
          output_stream: "IMAGE:output_image"
          options: {
            [mediapipe.ImageTransformationCalculatorOptions.ext]: {
              scale_mode: $0
              interpolation_mode: NEAREST
            }
          })",
                               scale_mode));

      CalculatorRunner runner(node_config);
      runner.MutableInputs()->Tag("IMAGE").packets.push_back(
          input_image_packet.At(Timestamp(0)));
      runner.MutableInputs()
          ->Tag("OUTPUT_DIMENSIONS")
          .packets.push_back(input_output_dim_packet.At(Timestamp(0)));

      ABSL_QCHECK_OK(runner.Run());
      const auto& outputs = runner.Outputs();
      ABSL_QCHECK_EQ(outputs.NumEntries(), 1);
      const std::vector<Packet>& packets = outputs.Tag("IMAGE").packets;
      ABSL_QCHECK_EQ(packets.size(), 1);

      const auto& result = packets[0].Get<ImageFrame>();
      ASSERT_EQ(output_dim.first, result.Width());
      ASSERT_EQ(output_dim.second, result.Height());

      auto unique_input_values =
          computeUniqueValues<int, unsigned char>(input_mat);
      auto unique_output_values =
          computeUniqueValues<int, unsigned char>(formats::MatView(&result));
      EXPECT_THAT(unique_input_values,
                  ::testing::ContainerEq(unique_output_values));
    }
  }
}

TEST(ImageTransformationCalculatorTest,
     NearestNeighborResizingWorksForFloatInput) {
  cv::Mat input_mat;
  cv::cvtColor(cv::imread(file::JoinPath("./",
                                         "/mediapipe/calculators/"
                                         "image/testdata/binary_mask.png")),
               input_mat, cv::COLOR_BGR2GRAY);
  Packet input_image_packet = MakePacket<ImageFrame>(
      ImageFormat::VEC32F1, input_mat.size().width, input_mat.size().height);
  cv::Mat packet_mat_view =
      formats::MatView(&(input_image_packet.Get<ImageFrame>()));
  input_mat.convertTo(packet_mat_view, CV_32FC1, 1 / 255.f);

  std::vector<std::pair<int, int>> output_dims{
      {256, 333}, {512, 512}, {1024, 1024}};

  for (auto& output_dim : output_dims) {
    Packet input_output_dim_packet =
        MakePacket<std::pair<int, int>>(output_dim);
    std::vector<std::string> scale_modes{"FIT", "STRETCH"};
    for (const auto& scale_mode : scale_modes) {
      CalculatorGraphConfig::Node node_config =
          ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
              absl::Substitute(R"(
          calculator: "ImageTransformationCalculator"
          input_stream: "IMAGE:input_image"
          input_stream: "OUTPUT_DIMENSIONS:image_size"
          output_stream: "IMAGE:output_image"
          options: {
            [mediapipe.ImageTransformationCalculatorOptions.ext]: {
              scale_mode: $0
              interpolation_mode: NEAREST
            }
          })",
                               scale_mode));

      CalculatorRunner runner(node_config);
      runner.MutableInputs()->Tag("IMAGE").packets.push_back(
          input_image_packet.At(Timestamp(0)));
      runner.MutableInputs()
          ->Tag("OUTPUT_DIMENSIONS")
          .packets.push_back(input_output_dim_packet.At(Timestamp(0)));

      ABSL_QCHECK_OK(runner.Run());
      const auto& outputs = runner.Outputs();
      ABSL_QCHECK_EQ(outputs.NumEntries(), 1);
      const std::vector<Packet>& packets = outputs.Tag("IMAGE").packets;
      ABSL_QCHECK_EQ(packets.size(), 1);

      const auto& result = packets[0].Get<ImageFrame>();
      ASSERT_EQ(output_dim.first, result.Width());
      ASSERT_EQ(output_dim.second, result.Height());

      auto unique_input_values =
          computeUniqueValues<float, float>(packet_mat_view);
      auto unique_output_values =
          computeUniqueValues<float, float>(formats::MatView(&result));
      EXPECT_THAT(unique_input_values,
                  ::testing::ContainerEq(unique_output_values));
    }
  }
}

TEST(ImageTransformationCalculatorTest, NearestNeighborResizingGpu) {
  cv::Mat input_mat;
  cv::cvtColor(cv::imread(file::JoinPath("./",
                                         "/mediapipe/calculators/"
                                         "image/testdata/binary_mask.png")),
               input_mat, cv::COLOR_BGR2RGBA);

  std::vector<std::pair<int, int>> output_dims{
      {256, 333}, {512, 512}, {1024, 1024}};

  for (auto& output_dim : output_dims) {
    std::vector<std::string> scale_modes{"FIT"};  //, "STRETCH"};
    for (const auto& scale_mode : scale_modes) {
      CalculatorGraphConfig graph_config =
          ParseTextProtoOrDie<CalculatorGraphConfig>(
              absl::Substitute(R"(
          input_stream: "input_image"
          input_stream: "image_size"
          output_stream: "output_image"

          node {
            calculator: "ImageFrameToGpuBufferCalculator"
            input_stream: "input_image"
            output_stream: "input_image_gpu"
          }

          node {
            calculator: "ImageTransformationCalculator"
            input_stream: "IMAGE_GPU:input_image_gpu"
            input_stream: "OUTPUT_DIMENSIONS:image_size"
            output_stream: "IMAGE_GPU:output_image_gpu"
            options: {
              [mediapipe.ImageTransformationCalculatorOptions.ext]: {
                scale_mode: $0
                interpolation_mode: NEAREST
              }
            }
          }
          node {
            calculator: "GpuBufferToImageFrameCalculator"
            input_stream: "output_image_gpu"
            output_stream: "output_image"
          })",
                               scale_mode));
      ImageFrame input_image(ImageFormat::SRGBA, input_mat.size().width,
                             input_mat.size().height);
      input_mat.copyTo(formats::MatView(&input_image));

      std::vector<Packet> output_image_packets;
      tool::AddVectorSink("output_image", &graph_config, &output_image_packets);

      CalculatorGraph graph(graph_config);
      ABSL_QCHECK_OK(graph.StartRun({}));

      ABSL_QCHECK_OK(graph.AddPacketToInputStream(
          "input_image",
          MakePacket<ImageFrame>(std::move(input_image)).At(Timestamp(0))));
      ABSL_QCHECK_OK(graph.AddPacketToInputStream(
          "image_size",
          MakePacket<std::pair<int, int>>(output_dim).At(Timestamp(0))));

      ABSL_QCHECK_OK(graph.WaitUntilIdle());
      ABSL_QCHECK_EQ(output_image_packets.size(), 1);

      const auto& output_image = output_image_packets[0].Get<ImageFrame>();
      ASSERT_EQ(output_dim.first, output_image.Width());
      ASSERT_EQ(output_dim.second, output_image.Height());

      auto unique_input_values =
          computeUniqueValues<int, unsigned char>(input_mat);
      auto unique_output_values = computeUniqueValues<int, unsigned char>(
          formats::MatView(&output_image));
      EXPECT_THAT(unique_input_values,
                  ::testing::ContainerEq(unique_output_values));
    }
  }
}

TEST(ImageTransformationCalculatorTest,
     NearestNeighborResizingWorksForFloatTexture) {
  cv::Mat input_mat;
  cv::cvtColor(cv::imread(file::JoinPath("./",
                                         "/mediapipe/calculators/"
                                         "image/testdata/binary_mask.png")),
               input_mat, cv::COLOR_BGR2GRAY);
  Packet input_image_packet = MakePacket<ImageFrame>(
      ImageFormat::VEC32F1, input_mat.size().width, input_mat.size().height);
  cv::Mat packet_mat_view =
      formats::MatView(&(input_image_packet.Get<ImageFrame>()));
  input_mat.convertTo(packet_mat_view, CV_32FC1, 1 / 255.f);

  std::vector<std::pair<int, int>> output_dims{
      {256, 333}, {512, 512}, {1024, 1024}};

  for (auto& output_dim : output_dims) {
    std::vector<std::string> scale_modes{"FIT"};  //, "STRETCH"};
    for (const auto& scale_mode : scale_modes) {
      CalculatorGraphConfig graph_config =
          ParseTextProtoOrDie<CalculatorGraphConfig>(
              absl::Substitute(R"(
          input_stream: "input_image"
          input_stream: "image_size"
          output_stream: "output_image"

          node {
            calculator: "ImageFrameToGpuBufferCalculator"
            input_stream: "input_image"
            output_stream: "input_image_gpu"
          }

          node {
            calculator: "ImageTransformationCalculator"
            input_stream: "IMAGE_GPU:input_image_gpu"
            input_stream: "OUTPUT_DIMENSIONS:image_size"
            output_stream: "IMAGE_GPU:output_image_gpu"
            options: {
              [mediapipe.ImageTransformationCalculatorOptions.ext]: {
                scale_mode: $0
                interpolation_mode: NEAREST
              }
            }
          }
          node {
            calculator: "GpuBufferToImageFrameCalculator"
            input_stream: "output_image_gpu"
            output_stream: "output_image"
          })",
                               scale_mode));

      std::vector<Packet> output_image_packets;
      tool::AddVectorSink("output_image", &graph_config, &output_image_packets);

      CalculatorGraph graph(graph_config);
      ABSL_QCHECK_OK(graph.StartRun({}));

      ABSL_QCHECK_OK(graph.AddPacketToInputStream(
          "input_image", input_image_packet.At(Timestamp(0))));
      ABSL_QCHECK_OK(graph.AddPacketToInputStream(
          "image_size",
          MakePacket<std::pair<int, int>>(output_dim).At(Timestamp(0))));

      ABSL_QCHECK_OK(graph.WaitUntilIdle());
      ABSL_QCHECK_EQ(output_image_packets.size(), 1);

      const auto& output_image = output_image_packets[0].Get<ImageFrame>();
      ASSERT_EQ(output_dim.first, output_image.Width());
      ASSERT_EQ(output_dim.second, output_image.Height());

      auto unique_input_values =
          computeUniqueValues<float, float>(packet_mat_view);
      auto unique_output_values =
          computeUniqueValues<float, float>(formats::MatView(&output_image));
      EXPECT_THAT(unique_input_values,
                  ::testing::ContainerEq(unique_output_values));
    }
  }
}

TEST(ImageTransformationCalculatorTest, FitScalingClearsBackground) {
  // Regression test for not clearing the background in FIT scaling mode.
  // First scale an all-red (=r) image from 8x4 to 8x4, so it's a plain copy:
  //   rrrrrrrr
  //   rrrrrrrr
  //   rrrrrrrr
  //   rrrrrrrr
  // Then scale an all-blue image from 4x4 to 8x4 in FIT mode. This should
  // introduce dark yellow (=y) letterboxes left and right due to padding_color:
  //   yybbbbyy
  //   yybbbbyy
  //   yybbbbyy
  //   yybbbbyy
  // We make sure that the all-red buffer gets reused. Without clearing the
  // background, the blue (=b) image will have red letterboxes:
  //   rrbbbbrr
  //   rrbbbbrr
  //   rrbbbbrr
  //   rrbbbbrr

  constexpr int kSmall = 4, kLarge = 8;
  ImageFrame input_image_red(ImageFormat::SRGBA, kLarge, kSmall);
  cv::Mat input_image_red_mat = formats::MatView(&input_image_red);
  input_image_red_mat = cv::Scalar(255, 0, 0, 255);

  ImageFrame input_image_blue(ImageFormat::SRGBA, kSmall, kSmall);
  cv::Mat input_image_blue_mat = formats::MatView(&input_image_blue);
  input_image_blue_mat = cv::Scalar(0, 0, 255, 255);

  Packet input_image_red_packet =
      MakePacket<ImageFrame>(std::move(input_image_red));
  Packet input_image_blue_packet =
      MakePacket<ImageFrame>(std::move(input_image_blue));

  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
          R"pb(
            input_stream: "input_image"
            output_stream: "output_image"

            node {
              calculator: "ImageFrameToGpuBufferCalculator"
              input_stream: "input_image"
              output_stream: "input_image_gpu"
            }

            node {
              calculator: "ImageTransformationCalculator"
              input_stream: "IMAGE_GPU:input_image_gpu"
              output_stream: "IMAGE_GPU:output_image_gpu"
              options: {
                [mediapipe.ImageTransformationCalculatorOptions.ext]: {
                  scale_mode: FIT
                  output_width: $0,
                  output_height: $1,
                  padding_color: { red: 128, green: 128, blue: 0 }
                }
              }
            }

            node {
              calculator: "GpuBufferToImageFrameCalculator"
              input_stream: "output_image_gpu"
              output_stream: "output_image"
            })pb",
          kLarge, kSmall));

  std::vector<Packet> output_image_packets;
  tool::AddVectorSink("output_image", &graph_config, &output_image_packets);

  CalculatorGraph graph(graph_config);
  ABSL_QCHECK_OK(graph.StartRun({}));

  // Send the red image multiple times to cause the GPU pool to actually use
  // a pool.
  int num_red_packets =
      std::max(kDefaultMultiPoolOptions.min_requests_before_pool, 1);
  for (int n = 0; n < num_red_packets; ++n) {
    ABSL_QCHECK_OK(graph.AddPacketToInputStream(
        "input_image", input_image_red_packet.At(Timestamp(n))));
  }
  ABSL_QCHECK_OK(graph.AddPacketToInputStream(
      "input_image", input_image_blue_packet.At(Timestamp(num_red_packets))));

  ABSL_QCHECK_OK(graph.WaitUntilIdle());
  ABSL_QCHECK_EQ(output_image_packets.size(), num_red_packets + 1);

  const auto& output_image_red = output_image_packets[0].Get<ImageFrame>();
  const auto& output_image_blue =
      output_image_packets[num_red_packets].Get<ImageFrame>();

  ABSL_QCHECK_EQ(output_image_red.Width(), kLarge);
  ABSL_QCHECK_EQ(output_image_red.Height(), kSmall);
  ABSL_QCHECK_EQ(output_image_blue.Width(), kLarge);
  ABSL_QCHECK_EQ(output_image_blue.Height(), kSmall);

  cv::Mat output_image_blue_mat = formats::MatView(&output_image_blue);
  ImageFrame expected_image_blue(ImageFormat::SRGBA, kLarge, kSmall);
  cv::Mat expected_image_blue_mat = formats::MatView(&expected_image_blue);
  expected_image_blue_mat = cv::Scalar(128, 128, 0, 255);
  cv::Rect rect((kLarge - kSmall) / 2, 0, kSmall, kSmall);
  cv::rectangle(expected_image_blue_mat, rect, cv::Scalar(0, 0, 255, 255),
                cv::FILLED);
  EXPECT_EQ(cv::sum(cv::sum(output_image_blue_mat != expected_image_blue_mat)),
            cv::Scalar(0));
}

}  // namespace
}  // namespace mediapipe
