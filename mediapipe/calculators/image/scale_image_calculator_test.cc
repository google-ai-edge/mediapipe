#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "libyuv/video_common.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/yuv_image.h"
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

// A solid-color I420 image. The chroma is deliberately far from neutral so
// that the BT.601 and BT.709 conversions give visibly different RGB.
YUVImage MakeSolidYuvImage(int width, int height,
                           YUVImage::ColorMatrixCoefficients matrix) {
  const int uv_width = (width + 1) / 2;
  const int uv_height = (height + 1) / 2;
  const int y_size = width * height;
  const int uv_size = uv_width * uv_height;

  uint8_t* buffer = new uint8_t[y_size + 2 * uv_size];
  std::fill_n(buffer, y_size, uint8_t{100});
  std::fill_n(buffer + y_size, uv_size, uint8_t{60});
  std::fill_n(buffer + y_size + uv_size, uv_size, uint8_t{200});

  YUVImage image;
  image.Initialize(
      libyuv::FOURCC_I420, [buffer]() { delete[] buffer; },
      absl::MakeSpan(buffer, y_size), width,
      absl::MakeSpan(buffer + y_size, uv_size), uv_width,
      absl::MakeSpan(buffer + y_size + uv_size, uv_size), uv_width, width,
      height);
  image.set_matrix_coefficients(matrix);
  return image;
}

// Runs the calculator over a single solid-color YUV frame and returns the RGB
// value every output pixel should have.
std::array<uint8_t, 3> ConvertSolidYuvFrame(
    YUVImage::ColorMatrixCoefficients matrix, bool use_bt709,
    bool use_input_matrix_coefficients) {
  constexpr int kSize = 16;
  auto calculator_node =
      ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(absl::StrCat(
          R"pb(
            calculator: "ScaleImageCalculator"
            input_stream: "input_frames"
            output_stream: "scaled_frames"
            options {
              [mediapipe.ScaleImageCalculatorOptions.ext] {
                input_format: YCBCR420P
                output_format: SRGB
                target_width: 16
                target_height: 16
                min_aspect_ratio: "0/1"
                max_aspect_ratio: "0/1"
          )pb",
          " use_bt709: ", use_bt709 ? "true" : "false",
          " use_input_matrix_coefficients: ",
          use_input_matrix_coefficients ? "true" : "false", " } } "));

  mediapipe::CalculatorRunner runner(calculator_node);
  runner.MutableInputs()->Index(0).packets.push_back(
      mediapipe::MakePacket<YUVImage>(MakeSolidYuvImage(kSize, kSize, matrix))
          .At(mediapipe::Timestamp(1)));
  ABSL_CHECK_OK(runner.Run());

  const auto& output_packets = runner.Outputs().Index(0).packets;
  ABSL_CHECK_EQ(output_packets.size(), 1);
  const uint8_t* pixels =
      output_packets[0].Get<mediapipe::ImageFrame>().PixelData();
  return {pixels[0], pixels[1], pixels[2]};
}

TEST(ScaleImageCalculatorTest, YuvMatrixCoefficients) {
  // References for the two conversions, selected purely by the option, with
  // the input declaring nothing.
  const auto bt601 = ConvertSolidYuvFrame(
      YUVImage::COLOR_MATRIX_COEFFICIENTS_UNSPECIFIED,
      /*use_bt709=*/false, /*use_input_matrix_coefficients=*/false);
  const auto bt709 = ConvertSolidYuvFrame(
      YUVImage::COLOR_MATRIX_COEFFICIENTS_UNSPECIFIED,
      /*use_bt709=*/true, /*use_input_matrix_coefficients=*/false);
  // Otherwise the rest of this test proves nothing.
  ASSERT_NE(bt601, bt709);

  // Opted out: the input's matrix is ignored, exactly as before this option
  // existed.
  EXPECT_EQ(ConvertSolidYuvFrame(YUVImage::COLOR_MATRIX_COEFFICIENTS_BT470BG,
                                 /*use_bt709=*/true,
                                 /*use_input_matrix_coefficients=*/false),
            bt709);

  // Opted in: the input's matrix wins over the option, in both directions.
  EXPECT_EQ(ConvertSolidYuvFrame(YUVImage::COLOR_MATRIX_COEFFICIENTS_BT470BG,
                                 /*use_bt709=*/true,
                                 /*use_input_matrix_coefficients=*/true),
            bt601);
  EXPECT_EQ(ConvertSolidYuvFrame(YUVImage::COLOR_MATRIX_COEFFICIENTS_SMPTE170M,
                                 /*use_bt709=*/true,
                                 /*use_input_matrix_coefficients=*/true),
            bt601);
  EXPECT_EQ(ConvertSolidYuvFrame(YUVImage::COLOR_MATRIX_COEFFICIENTS_BT709,
                                 /*use_bt709=*/false,
                                 /*use_input_matrix_coefficients=*/true),
            bt709);

  // Opted in, with a matrix that has no conversion of its own: FCC and
  // SMPTE 240M are converted as BT.601.
  EXPECT_EQ(ConvertSolidYuvFrame(YUVImage::COLOR_MATRIX_COEFFICIENTS_FCC,
                                 /*use_bt709=*/true,
                                 /*use_input_matrix_coefficients=*/true),
            bt601);
  EXPECT_EQ(ConvertSolidYuvFrame(YUVImage::COLOR_MATRIX_COEFFICIENTS_SMPTE240M,
                                 /*use_bt709=*/true,
                                 /*use_input_matrix_coefficients=*/true),
            bt601);

  // Opted in but the input declares nothing usable: fall back to the option.
  EXPECT_EQ(
      ConvertSolidYuvFrame(YUVImage::COLOR_MATRIX_COEFFICIENTS_UNSPECIFIED,
                           /*use_bt709=*/true,
                           /*use_input_matrix_coefficients=*/true),
      bt709);
  EXPECT_EQ(
      ConvertSolidYuvFrame(YUVImage::COLOR_MATRIX_COEFFICIENTS_UNSPECIFIED,
                           /*use_bt709=*/false,
                           /*use_input_matrix_coefficients=*/true),
      bt601);
  EXPECT_EQ(ConvertSolidYuvFrame(YUVImage::COLOR_MATRIX_COEFFICIENTS_BT2020_NCL,
                                 /*use_bt709=*/true,
                                 /*use_input_matrix_coefficients=*/true),
            bt709);
  EXPECT_EQ(ConvertSolidYuvFrame(YUVImage::COLOR_MATRIX_COEFFICIENTS_BT2020_NCL,
                                 /*use_bt709=*/false,
                                 /*use_input_matrix_coefficients=*/true),
            bt601);
}

TEST(ScaleImageCalculatorTest, YuvScalingPreservesMatrixCoefficients) {
  auto calculator_node =
      ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(R"pb(
        calculator: "ScaleImageCalculator"
        input_stream: "input_frames"
        output_stream: "scaled_frames"
        options {
          [mediapipe.ScaleImageCalculatorOptions.ext] {
            input_format: YCBCR420P
            output_format: YCBCR420P
            target_width: 8
            target_height: 8
            min_aspect_ratio: "0/1"
            max_aspect_ratio: "0/1"
          }
        }
      )pb");

  mediapipe::CalculatorRunner runner(calculator_node);
  runner.MutableInputs()->Index(0).packets.push_back(
      mediapipe::MakePacket<YUVImage>(
          MakeSolidYuvImage(16, 16, YUVImage::COLOR_MATRIX_COEFFICIENTS_BT709))
          .At(mediapipe::Timestamp(1)));
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets = runner.Outputs().Index(0).packets;
  ASSERT_EQ(output_packets.size(), 1);
  const auto& scaled = output_packets[0].Get<YUVImage>();
  EXPECT_EQ(scaled.width(), 8);
  EXPECT_EQ(scaled.height(), 8);
  EXPECT_EQ(scaled.matrix_coefficients(),
            YUVImage::COLOR_MATRIX_COEFFICIENTS_BT709);
}

}  // namespace
}  // namespace mediapipe
