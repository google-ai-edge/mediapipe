#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/yuv_image.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/benchmark.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/util/image_frame_util.h"

namespace mediapipe {
namespace {

using ::testing::HasSubstr;
using ::testing::status::StatusIs;

mediapipe::ImageFrame GetInputFrame(
    const int width, const int height, const int channel,
    const mediapipe::ImageFormat::Format image_format) {
  const int total_size = width * height * channel;

  mediapipe::ImageFrame input_frame(image_format, width, height,
                                    /*alignment_boundary =*/1);
  uint8_t* pixel_data = input_frame.MutablePixelData();
  for (int i = 0; i < total_size; ++i) {
    pixel_data[i] = i % 256;
  }

  return input_frame;
}

mediapipe::CalculatorGraphConfig::Node GetTestingGraphNode() {
  return ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
      R"pb(
        calculator: "ScaleImageCalculator"
        input_stream: "input_frames"
        output_stream: "scaled_frames"
        options {
          [mediapipe.ScaleImageCalculatorOptions.ext] {
            input_format: SRGB
            output_format: SRGB
            target_width: 720
            target_height: 720
            preserve_aspect_ratio: true
          }
        }
      )pb");
}

TEST(ScaleImageCalculatorTest, ScaleRegualrSize) {
  auto calculator_node = GetTestingGraphNode();
  mediapipe::CalculatorRunner runner(calculator_node);

  // Vertical 9:16 720P input frame
  auto input_frame = GetInputFrame(720, 1280, 3, mediapipe::ImageFormat::SRGB);
  auto input_frame_packet =
      mediapipe::MakePacket<mediapipe::ImageFrame>(std::move(input_frame));
  runner.MutableInputs()->Index(0).packets.push_back(
      input_frame_packet.At(mediapipe::Timestamp(1)));
  MP_ASSERT_OK(runner.Run());
}

TEST(ScaleImageCalculatorTest, ScaleOddSize) {
  auto calculator_node = GetTestingGraphNode();
  mediapipe::CalculatorRunner runner(calculator_node);

  // 1 x 512 input frame
  auto input_frame = GetInputFrame(1, 512, 3, mediapipe::ImageFormat::SRGB);
  auto input_frame_packet =
      mediapipe::MakePacket<mediapipe::ImageFrame>(std::move(input_frame));
  runner.MutableInputs()->Index(0).packets.push_back(
      input_frame_packet.At(mediapipe::Timestamp(1)));
  ASSERT_THAT(runner.Run(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Image frame is empty before rescaling.")));
}

TEST(ScaleImageCalculatorTest, ScaleRgbaToRgb) {
  auto calculator_node =
      ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"pb(
            calculator: "ScaleImageCalculator"
            input_stream: "input_frames"
            output_stream: "scaled_frames"
            options {
              [mediapipe.ScaleImageCalculatorOptions.ext] {
                input_format: SRGBA
                output_format: SRGB
                target_width: 720
                target_height: 720
                preserve_aspect_ratio: true
              }
            }
          )pb");
  mediapipe::CalculatorRunner runner(calculator_node);

  // Vertical 9:16 720P input frame
  auto input_frame = GetInputFrame(720, 1280, 4, mediapipe::ImageFormat::SRGBA);
  auto input_frame_packet =
      mediapipe::MakePacket<mediapipe::ImageFrame>(std::move(input_frame));
  runner.MutableInputs()->Index(0).packets.push_back(
      input_frame_packet.At(mediapipe::Timestamp(1)));
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets = runner.Outputs().Index(0).packets;
  ASSERT_EQ(output_packets.size(), 1);
  const auto& output_frame = output_packets[0].Get<mediapipe::ImageFrame>();
  EXPECT_EQ(output_frame.Format(), mediapipe::ImageFormat::SRGB);
  EXPECT_EQ(output_frame.Width(), 404);
  EXPECT_EQ(output_frame.Height(), 720);
}

TEST(ScaleImageCalculatorTest, ScaleRgbaToRgbNoInputFormatInOption) {
  // This is exactly the same as ScaleRgbaToRgb except that we don't specify
  // the input_format in the option. It should therfore be obtained from the
  // frame itself.
  auto calculator_node =
      ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"pb(
            calculator: "ScaleImageCalculator"
            input_stream: "input_frames"
            output_stream: "scaled_frames"
            options {
              [mediapipe.ScaleImageCalculatorOptions.ext] {
                # input_format: SRGBA
                output_format: SRGB
                target_width: 720
                target_height: 720
                preserve_aspect_ratio: true
              }
            }
          )pb");
  mediapipe::CalculatorRunner runner(calculator_node);

  // Vertical 9:16 720P input frame
  auto input_frame = GetInputFrame(720, 1280, 4, mediapipe::ImageFormat::SRGBA);
  auto input_frame_packet =
      mediapipe::MakePacket<mediapipe::ImageFrame>(std::move(input_frame));
  runner.MutableInputs()->Index(0).packets.push_back(
      input_frame_packet.At(mediapipe::Timestamp(1)));
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets = runner.Outputs().Index(0).packets;
  ASSERT_EQ(output_packets.size(), 1);
  const auto& output_frame = output_packets[0].Get<mediapipe::ImageFrame>();
  EXPECT_EQ(output_frame.Format(), mediapipe::ImageFormat::SRGB);
  EXPECT_EQ(output_frame.Width(), 404);
  EXPECT_EQ(output_frame.Height(), 720);
}

TEST(ScaleImageCalculatorTest, ScaleRgbToRgb) {
  // This is exactly the same as ScaleRgbaToRgb except that we use SRGB instead
  // of SRGBA.
  auto calculator_node =
      ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"pb(
            calculator: "ScaleImageCalculator"
            input_stream: "input_frames"
            output_stream: "scaled_frames"
            options {
              [mediapipe.ScaleImageCalculatorOptions.ext] {
                input_format: SRGB
                output_format: SRGB
                target_width: 720
                target_height: 720
                preserve_aspect_ratio: true
              }
            }
          )pb");
  mediapipe::CalculatorRunner runner(calculator_node);

  // Vertical 9:16 720P input frame
  auto input_frame = GetInputFrame(720, 1280, 3, mediapipe::ImageFormat::SRGB);
  auto input_frame_packet =
      mediapipe::MakePacket<mediapipe::ImageFrame>(std::move(input_frame));
  runner.MutableInputs()->Index(0).packets.push_back(
      input_frame_packet.At(mediapipe::Timestamp(1)));
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets = runner.Outputs().Index(0).packets;
  ASSERT_EQ(output_packets.size(), 1);
  const auto& output_frame = output_packets[0].Get<mediapipe::ImageFrame>();
  EXPECT_EQ(output_frame.Format(), mediapipe::ImageFormat::SRGB);
  EXPECT_EQ(output_frame.Width(), 404);
  EXPECT_EQ(output_frame.Height(), 720);
}

void ExpectEmptyImageFails(int width, int height) {
  auto calculator_node = GetTestingGraphNode();
  mediapipe::CalculatorRunner runner(calculator_node);

  mediapipe::ImageFrame input_frame(mediapipe::ImageFormat::SRGB, width, height,
                                    1);
  auto input_frame_packet =
      mediapipe::MakePacket<mediapipe::ImageFrame>(std::move(input_frame));
  runner.MutableInputs()->Index(0).packets.push_back(
      input_frame_packet.At(mediapipe::Timestamp(1)));
  ASSERT_THAT(runner.Run(), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Input image frame is empty.")));
}

TEST(ScaleImageCalculatorTest, ScaleEmptyImage) { ExpectEmptyImageFails(0, 0); }

TEST(ScaleImageCalculatorTest, ScaleImageWithZeroWidth) {
  ExpectEmptyImageFails(0, 100);
}

TEST(ScaleImageCalculatorTest, ScaleImageWithZeroHeight) {
  ExpectEmptyImageFails(100, 0);
}

TEST(ScaleImageCalculatorTest, ScaleYuvI420AndNv12ToSrgb) {
  auto calculator_node =
      ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"pb(
            calculator: "ScaleImageCalculator"
            input_stream: "input_frames"
            output_stream: "scaled_frames"
            options {
              [mediapipe.ScaleImageCalculatorOptions.ext] {
                input_format: YCBCR420P
                output_format: SRGB
                target_width: 320
                target_height: 180
                preserve_aspect_ratio: true
              }
            }
          )pb");
  auto src_rgb = GetInputFrame(640, 360, 3, mediapipe::ImageFormat::SRGB);
  auto i420 = std::make_unique<YUVImage>();
  image_frame_util::ImageFrameToYUVImage(src_rgb, i420.get());
  auto nv12 = std::make_unique<YUVImage>();
  image_frame_util::ImageFrameToYUVNV12Image(src_rgb, nv12.get());

  mediapipe::CalculatorRunner runner(calculator_node);
  runner.MutableInputs()->Index(0).packets.push_back(
      mediapipe::Adopt(i420.release()).At(mediapipe::Timestamp(1)));
  runner.MutableInputs()->Index(0).packets.push_back(
      mediapipe::Adopt(nv12.release()).At(mediapipe::Timestamp(2)));
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets = runner.Outputs().Index(0).packets;
  ASSERT_EQ(output_packets.size(), 2);
  for (int i = 0; i < 2; ++i) {
    const auto& output_frame = output_packets[i].Get<mediapipe::ImageFrame>();
    EXPECT_EQ(output_frame.Format(), mediapipe::ImageFormat::SRGB);
    EXPECT_EQ(output_frame.Width(), 320);
    EXPECT_EQ(output_frame.Height(), 180);
  }
}

void RunScaleYuvToSrgbBenchmark(benchmark::State& state, bool nv12) {
  const int in_w = state.range(0);
  const int in_h = state.range(1);
  const int out_w = state.range(2);
  const int out_h = state.range(3);

  auto src_rgb = GetInputFrame(in_w, in_h, 3, mediapipe::ImageFormat::SRGB);
  auto yuv_image = std::make_shared<YUVImage>();
  if (nv12) {
    image_frame_util::ImageFrameToYUVNV12Image(src_rgb, yuv_image.get());
  } else {
    image_frame_util::ImageFrameToYUVImage(src_rgb, yuv_image.get());
  }
  Packet input_packet = PointToForeign(yuv_image.get());

  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
          R"pb(
            input_stream: "input_frames"
            output_stream: "scaled_frames"
            node {
              calculator: "ScaleImageCalculator"
              input_stream: "input_frames"
              output_stream: "scaled_frames"
              options {
                [mediapipe.ScaleImageCalculatorOptions.ext] {
                  input_format: YCBCR420P
                  output_format: SRGB
                  target_width: $0
                  target_height: $1
                  preserve_aspect_ratio: true
                }
              }
            }
          )pb",
          out_w, out_h));

  CalculatorGraph graph;
  ABSL_CHECK_OK(graph.Initialize(config));
  ABSL_CHECK_OK(graph.ObserveOutputStream(
      "scaled_frames", [](const Packet& p) { return absl::OkStatus(); }));
  ABSL_CHECK_OK(graph.StartRun({}));

  int64_t ts = 0;
  for (auto s : state) {
    ABSL_CHECK_OK(graph.AddPacketToInputStream(
        "input_frames", input_packet.At(Timestamp(ts++))));
    ABSL_CHECK_OK(graph.WaitUntilIdle());
  }
  ABSL_CHECK_OK(graph.CloseAllInputStreams());
  ABSL_CHECK_OK(graph.WaitUntilDone());
}

void BM_ScaleYuvI420ToSrgb(benchmark::State& state) {
  RunScaleYuvToSrgbBenchmark(state, /*nv12=*/false);
}
BENCHMARK(BM_ScaleYuvI420ToSrgb)
    ->Args({1920, 1080, 1920, 1080})
    ->Args({1920, 1080, 1280, 720})
    ->Args({1920, 1080, 640, 360})
    ->Args({1920, 1080, 320, 180})
    ->Args({3840, 2160, 640, 360});

void BM_ScaleYuvNv12ToSrgb(benchmark::State& state) {
  RunScaleYuvToSrgbBenchmark(state, /*nv12=*/true);
}
BENCHMARK(BM_ScaleYuvNv12ToSrgb)
    ->Args({1920, 1080, 1280, 720})
    ->Args({1920, 1080, 320, 180});

}  // namespace
}  // namespace mediapipe
