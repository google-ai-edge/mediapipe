#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"

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

}  // namespace
}  // namespace mediapipe
