#include <array>
#include <vector>

#include "absl/memory/memory.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

constexpr char kProjectionMatrixTag[] = "PROJECTION_MATRIX";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";

absl::StatusOr<mediapipe::NormalizedLandmarkList> RunCalculator(
    mediapipe::NormalizedLandmarkList input, mediapipe::NormalizedRect rect) {
  mediapipe::CalculatorRunner runner(
      ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(R"pb(
        calculator: "LandmarkProjectionCalculator"
        input_stream: "NORM_LANDMARKS:landmarks"
        input_stream: "NORM_RECT:rect"
        output_stream: "NORM_LANDMARKS:projected_landmarks"
      )pb"));
  runner.MutableInputs()
      ->Tag(kNormLandmarksTag)
      .packets.push_back(
          MakePacket<mediapipe::NormalizedLandmarkList>(std::move(input))
              .At(Timestamp(1)));
  runner.MutableInputs()
      ->Tag(kNormRectTag)
      .packets.push_back(MakePacket<mediapipe::NormalizedRect>(std::move(rect))
                             .At(Timestamp(1)));

  MP_RETURN_IF_ERROR(runner.Run());
  const auto& output_packets = runner.Outputs().Tag(kNormLandmarksTag).packets;
  RET_CHECK_EQ(output_packets.size(), 1);
  return output_packets[0].Get<mediapipe::NormalizedLandmarkList>();
}

TEST(LandmarkProjectionCalculatorTest, ProjectingWithDefaultRect) {
  mediapipe::NormalizedLandmarkList landmarks =
      ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 10, y: 20, z: -0.5 }
      )pb");
  mediapipe::NormalizedRect rect =
      ParseTextProtoOrDie<mediapipe::NormalizedRect>(
          R"pb(
            x_center: 0.5,
            y_center: 0.5,
            width: 1.0,
            height: 1.0,
            rotation: 0.0
          )pb");

  auto status_or_result = RunCalculator(std::move(landmarks), std::move(rect));
  MP_ASSERT_OK(status_or_result);

  EXPECT_THAT(
      status_or_result.value(),
      EqualsProto(ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 10, y: 20, z: -0.5 }
      )pb")));
}

mediapipe::NormalizedRect GetCroppedRect() {
  return ParseTextProtoOrDie<mediapipe::NormalizedRect>(
      R"pb(
        x_center: 0.5, y_center: 0.5, width: 0.5, height: 2, rotation: 0.0
      )pb");
}

mediapipe::NormalizedLandmarkList GetCroppedRectTestInput() {
  return ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
    landmark { x: 1.0, y: 1.0, z: -0.5 }
  )pb");
}

mediapipe::NormalizedLandmarkList GetCroppedRectTestExpectedResult() {
  return ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
    landmark { x: 0.75, y: 1.5, z: -0.25 }
  )pb");
}

TEST(LandmarkProjectionCalculatorTest, ProjectingWithCroppedRect) {
  auto status_or_result =
      RunCalculator(GetCroppedRectTestInput(), GetCroppedRect());
  MP_ASSERT_OK(status_or_result);

  EXPECT_THAT(status_or_result.value(),
              EqualsProto(GetCroppedRectTestExpectedResult()));
}

absl::StatusOr<mediapipe::NormalizedLandmarkList> RunCalculator(
    mediapipe::NormalizedLandmarkList input, std::array<float, 16> matrix) {
  mediapipe::CalculatorRunner runner(
      ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(R"pb(
        calculator: "LandmarkProjectionCalculator"
        input_stream: "NORM_LANDMARKS:landmarks"
        input_stream: "PROJECTION_MATRIX:matrix"
        output_stream: "NORM_LANDMARKS:projected_landmarks"
      )pb"));
  runner.MutableInputs()
      ->Tag(kNormLandmarksTag)
      .packets.push_back(
          MakePacket<mediapipe::NormalizedLandmarkList>(std::move(input))
              .At(Timestamp(1)));
  runner.MutableInputs()
      ->Tag(kProjectionMatrixTag)
      .packets.push_back(MakePacket<std::array<float, 16>>(std::move(matrix))
                             .At(Timestamp(1)));

  MP_RETURN_IF_ERROR(runner.Run());
  const auto& output_packets = runner.Outputs().Tag(kNormLandmarksTag).packets;
  RET_CHECK_EQ(output_packets.size(), 1);
  return output_packets[0].Get<mediapipe::NormalizedLandmarkList>();
}

TEST(LandmarkProjectionCalculatorTest, ProjectingWithIdentityMatrix) {
  mediapipe::NormalizedLandmarkList landmarks =
      ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 10, y: 20, z: -0.5 }
      )pb");
  // clang-format off
  std::array<float, 16> matrix = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
  };
  // clang-format on

  auto status_or_result =
      RunCalculator(std::move(landmarks), std::move(matrix));
  MP_ASSERT_OK(status_or_result);

  EXPECT_THAT(
      status_or_result.value(),
      EqualsProto(ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 10, y: 20, z: -0.5 }
      )pb")));
}

TEST(LandmarkProjectionCalculatorTest, ProjectingWithCroppedRectMatrix) {
  constexpr int kRectWidth = 1280;
  constexpr int kRectHeight = 720;
  auto roi = GetRoi(kRectWidth, kRectHeight, GetCroppedRect());
  std::array<float, 16> matrix;
  GetRotatedSubRectToRectTransformMatrix(roi, kRectWidth, kRectHeight,
                                         /*flip_horizontaly=*/false, &matrix);
  auto status_or_result = RunCalculator(GetCroppedRectTestInput(), matrix);
  MP_ASSERT_OK(status_or_result);

  EXPECT_THAT(status_or_result.value(),
              EqualsProto(GetCroppedRectTestExpectedResult()));
}

TEST(LandmarkProjectionCalculatorTest, ProjectingWithScaleMatrix) {
  mediapipe::NormalizedLandmarkList landmarks =
      ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 10, y: 20, z: -0.5 }
        landmark { x: 5, y: 6, z: 7 }
      )pb");
  // clang-format off
  std::array<float, 16> matrix = {
    10.0f, 0.0f,   0.0f, 0.0f,
    0.0f,  100.0f, 0.0f, 0.0f,
    0.0f,  0.0f,   1.0f, 0.0f,
    0.0f,  0.0f,   0.0f, 1.0f,
  };
  // clang-format on

  auto status_or_result =
      RunCalculator(std::move(landmarks), std::move(matrix));
  MP_ASSERT_OK(status_or_result);

  EXPECT_THAT(
      status_or_result.value(),
      EqualsProto(ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 100, y: 2000, z: -5 }
        landmark { x: 50, y: 600, z: 70 }
      )pb")));
}

TEST(LandmarkProjectionCalculatorTest, ProjectingWithTranslateMatrix) {
  mediapipe::NormalizedLandmarkList landmarks =
      ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 10, y: 20, z: -0.5 }
      )pb");
  // clang-format off
  std::array<float, 16> matrix = {
    1.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 1.0f, 0.0f, 2.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
  };
  // clang-format on

  auto status_or_result =
      RunCalculator(std::move(landmarks), std::move(matrix));
  MP_ASSERT_OK(status_or_result);

  EXPECT_THAT(
      status_or_result.value(),
      EqualsProto(ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 11, y: 22, z: -0.5 }
      )pb")));
}

TEST(LandmarkProjectionCalculatorTest, ProjectingWithRotationMatrix) {
  mediapipe::NormalizedLandmarkList landmarks =
      ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 4, y: 0, z: -0.5 }
      )pb");
  // clang-format off
  // 90 degrees rotation matrix
  std::array<float, 16> matrix = {
    0.0f, -1.0f, 0.0f, 0.0f,
    1.0f,  0.0f, 0.0f, 0.0f,
    0.0f,  0.0f, 1.0f, 0.0f,
    0.0f,  0.0f, 0.0f, 1.0f,
  };
  // clang-format on

  auto status_or_result =
      RunCalculator(std::move(landmarks), std::move(matrix));
  MP_ASSERT_OK(status_or_result);

  EXPECT_THAT(
      status_or_result.value(),
      EqualsProto(ParseTextProtoOrDie<mediapipe::NormalizedLandmarkList>(R"pb(
        landmark { x: 0, y: 4, z: -0.5 }
      )pb")));
}

}  // namespace
}  // namespace mediapipe
