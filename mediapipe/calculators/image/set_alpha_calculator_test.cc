#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"  // IWYU pragma: keep
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "testing/base/public/benchmark.h"

namespace mediapipe {

namespace {

constexpr int input_width = 100;
constexpr int input_height = 100;

std::unique_ptr<ImageFrame> GetInputFrame(int width, int height, int channel) {
  const int total_size = width * height * channel;

  ImageFormat::Format image_format;
  if (channel == 4) {
    image_format = ImageFormat::SRGBA;
  } else if (channel == 3) {
    image_format = ImageFormat::SRGB;
  } else {
    image_format = ImageFormat::GRAY8;
  }

  auto input_frame = std::make_unique<ImageFrame>(image_format, width, height,
                                                  /*alignment_boundary =*/1);
  for (int i = 0; i < total_size; ++i) {
    input_frame->MutablePixelData()[i] = i % 256;
  }
  return input_frame;
}

// Test SetAlphaCalculator with RGB IMAGE input.
TEST(SetAlphaCalculatorTest, CpuRgb) {
  auto calculator_node = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
      R"pb(
        calculator: "SetAlphaCalculator"
        input_stream: "IMAGE:input_frames"
        input_stream: "ALPHA:masks"
        output_stream: "IMAGE:output_frames"
      )pb");
  CalculatorRunner runner(calculator_node);

  // Input frames.
  const auto input_frame = GetInputFrame(input_width, input_height, 3);
  const auto mask_frame = GetInputFrame(input_width, input_height, 1);
  auto input_frame_packet = MakePacket<ImageFrame>(std::move(*input_frame));
  auto mask_frame_packet = MakePacket<ImageFrame>(std::move(*mask_frame));
  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      input_frame_packet.At(Timestamp(1)));
  runner.MutableInputs()->Tag("ALPHA").packets.push_back(
      mask_frame_packet.At(Timestamp(1)));

  MP_ASSERT_OK(runner.Run());

  const auto& outputs = runner.Outputs();
  EXPECT_EQ(outputs.NumEntries(), 1);
  const auto& output_image = outputs.Tag("IMAGE").packets[0].Get<ImageFrame>();

  // Generate ground truth (expected_mat).
  const auto image = GetInputFrame(input_width, input_height, 3);
  const auto input_mat = formats::MatView(image.get());
  const auto mask = GetInputFrame(input_width, input_height, 1);
  const auto mask_mat = formats::MatView(mask.get());
  const std::array<cv::Mat, 2> input_mats = {input_mat, mask_mat};
  cv::Mat expected_mat(input_width, input_height, CV_8UC4);
  cv::mixChannels(input_mats, {expected_mat}, {0, 0, 1, 1, 2, 2, 3, 3});

  cv::Mat output_mat = formats::MatView(&output_image);
  double max_diff = cv::norm(expected_mat, output_mat, cv::NORM_INF);
  EXPECT_FLOAT_EQ(max_diff, 0);
}  // TEST

// Test SetAlphaCalculator with RGBA IMAGE input.
TEST(SetAlphaCalculatorTest, CpuRgba) {
  auto calculator_node = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
      R"pb(
        calculator: "SetAlphaCalculator"
        input_stream: "IMAGE:input_frames"
        input_stream: "ALPHA:masks"
        output_stream: "IMAGE:output_frames"
      )pb");
  CalculatorRunner runner(calculator_node);

  // Input frames.
  const auto input_frame = GetInputFrame(input_width, input_height, 4);
  const auto mask_frame = GetInputFrame(input_width, input_height, 1);
  auto input_frame_packet = MakePacket<ImageFrame>(std::move(*input_frame));
  auto mask_frame_packet = MakePacket<ImageFrame>(std::move(*mask_frame));
  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      input_frame_packet.At(Timestamp(1)));
  runner.MutableInputs()->Tag("ALPHA").packets.push_back(
      mask_frame_packet.At(Timestamp(1)));

  MP_ASSERT_OK(runner.Run());

  const auto& outputs = runner.Outputs();
  EXPECT_EQ(outputs.NumEntries(), 1);
  const auto& output_image = outputs.Tag("IMAGE").packets[0].Get<ImageFrame>();

  // Generate ground truth (expected_mat).
  const auto image = GetInputFrame(input_width, input_height, 4);
  const auto input_mat = formats::MatView(image.get());
  const auto mask = GetInputFrame(input_width, input_height, 1);
  const auto mask_mat = formats::MatView(mask.get());
  const std::array<cv::Mat, 2> input_mats = {input_mat, mask_mat};
  cv::Mat expected_mat(input_width, input_height, CV_8UC4);
  cv::mixChannels(input_mats, {expected_mat}, {0, 0, 1, 1, 2, 2, 4, 3});

  cv::Mat output_mat = formats::MatView(&output_image);
  double max_diff = cv::norm(expected_mat, output_mat, cv::NORM_INF);
  EXPECT_FLOAT_EQ(max_diff, 0);
}  // TEST

#if !MEDIAPIPE_DISABLE_GPU && defined(__linux__)
TEST(SetAlphaCalculatorTest, GpuRgbaWithAlphaValueOption) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "input_frames"
    output_stream: "output_frames"
    node {
      calculator: "ImageFrameToGpuBufferCalculator"
      input_stream: "input_frames"
      output_stream: "input_frames_gpu"
    }
    node {
      calculator: "SetAlphaCalculator"
      input_stream: "IMAGE_GPU:input_frames_gpu"
      output_stream: "IMAGE_GPU:output_frames_gpu"
      options {
        [mediapipe.SetAlphaCalculatorOptions.ext] { alpha_value: 0 }
      }
    }
    node {
      calculator: "GpuBufferToImageFrameCalculator"
      input_stream: "output_frames_gpu"
      output_stream: "output_frames"
    }
  )pb");

  std::vector<Packet> output_packets;
  tool::AddVectorSink("output_frames", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  for (int ts = 1; ts <= 2; ++ts) {
    auto input_frame = GetInputFrame(input_width, input_height, 4);
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "input_frames",
        MakePacket<ImageFrame>(std::move(*input_frame)).At(Timestamp(ts))));
    MP_ASSERT_OK(graph.WaitUntilIdle());
  }
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(output_packets.size(), 2);
  const auto expected_src = GetInputFrame(input_width, input_height, 4);
  cv::Mat expected_mat = formats::MatView(expected_src.get()).clone();
  for (int r = 0; r < expected_mat.rows; ++r) {
    for (int c = 0; c < expected_mat.cols; ++c) {
      expected_mat.at<cv::Vec4b>(r, c)[3] = 0;
    }
  }
  for (const Packet& packet : output_packets) {
    const ImageFrame& output_image = packet.Get<ImageFrame>();
    cv::Mat output_mat = formats::MatView(&output_image);
    EXPECT_FLOAT_EQ(cv::norm(expected_mat, output_mat, cv::NORM_INF), 0);
  }
}
#endif  // !MEDIAPIPE_DISABLE_GPU && defined(__linux__)

static void BM_SetAlpha3ChannelImage(benchmark::State& state) {
  auto calculator_node = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
      R"pb(
        calculator: "SetAlphaCalculator"
        input_stream: "IMAGE:input_frames"
        input_stream: "ALPHA:masks"
        output_stream: "IMAGE:output_frames"
      )pb");
  CalculatorRunner runner(calculator_node);

  // Input frames.
  const auto input_frame = GetInputFrame(input_width, input_height, 3);
  const auto mask_frame = GetInputFrame(input_width, input_height, 1);
  auto input_frame_packet = MakePacket<ImageFrame>(std::move(*input_frame));
  auto mask_frame_packet = MakePacket<ImageFrame>(std::move(*mask_frame));
  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      input_frame_packet.At(Timestamp(1)));
  runner.MutableInputs()->Tag("ALPHA").packets.push_back(
      mask_frame_packet.At(Timestamp(1)));

  MP_ASSERT_OK(runner.Run());
  const auto& outputs = runner.Outputs();
  ASSERT_EQ(1, outputs.NumEntries());

  for (const auto _ : state) {
    MP_ASSERT_OK(runner.Run());
  }
}

BENCHMARK(BM_SetAlpha3ChannelImage);

}  // namespace
}  // namespace mediapipe
