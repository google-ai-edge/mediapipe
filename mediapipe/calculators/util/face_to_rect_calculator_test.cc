
#include <algorithm>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/util/face_to_rect_calculator.pb.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

constexpr int kImageWidth = 1280;
constexpr int kImageHeight = 720;
constexpr float kEps = 1e-5f;

Detection DetectionWithKeyPoints(
    const std::vector<std::pair<float, float>>& key_points) {
  Detection detection;
  LocationData* location_data = detection.mutable_location_data();
  std::for_each(key_points.begin(), key_points.end(),
                [location_data](std::pair<float, float> kp) {
                  auto* new_kp = location_data->add_relative_keypoints();
                  new_kp->set_x(kp.first);
                  new_kp->set_y(kp.second);
                });
  return detection;
}

// Creates a FaceToRectCalculatorOptions from landmarks sizes of eye, nose, and
// mouth.
FaceToRectCalculatorOptions CreateOptions(int eye, int nose, int mouth) {
  FaceToRectCalculatorOptions face_options;
  face_options.set_eye_landmark_size(eye);
  face_options.set_nose_landmark_size(nose);
  face_options.set_mouth_landmark_size(mouth);
  return face_options;
}

absl::StatusOr<NormalizedRect> RunCalculator(
    Detection detection, FaceToRectCalculatorOptions options) {
  CalculatorGraphConfig::Node node =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "FaceToRectCalculator"
        input_stream: "DETECTION:detection"
        input_stream: "IMAGE_SIZE:frame_size"
        output_stream: "NORM_RECT:rect"
      )pb");
  node.mutable_node_options()->Add()->PackFrom(options);
  CalculatorRunner runner(node);

  runner.MutableInputs()
      ->Tag("DETECTION")
      .packets.push_back(MakePacket<Detection>(std::move(detection))
                             .At(Timestamp::PostStream()));
  runner.MutableInputs()
      ->Tag("IMAGE_SIZE")
      .packets.push_back(
          MakePacket<std::pair<int, int>>(kImageWidth, kImageHeight)
              .At(Timestamp::PostStream()));

  MP_RETURN_IF_ERROR(runner.Run());
  const std::vector<Packet>& output = runner.Outputs().Tag("NORM_RECT").packets;
  RET_CHECK_EQ(output.size(), 1);
  return output[0].Get<NormalizedRect>();
}

}  // namespace

TEST(FaceToRectCalculator, WrongNumberOfKeyPoints) {
  auto status_or_value = RunCalculator(
      /*detection=*/DetectionWithKeyPoints({
          {0.3f, 0.5f},   // left eye
          {0.6f, 0.45f},  // right eye
          {1.0f, 1.0f},   // nose - not used
          {0.5f, 0.65f},  // mouth
      }),
      CreateOptions(1, 1, 2));
  EXPECT_FALSE(status_or_value.ok());
  EXPECT_EQ(status_or_value.status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(FaceToRectCalculator, DetectionToNormalizedRect) {
  auto status_or_value = RunCalculator(
      /*detection=*/DetectionWithKeyPoints({
          {0.3f, 0.5f},   // left eye
          {0.6f, 0.45f},  // right eye
          {1.0f, 1.0f},   // nose - not used
          {0.5f, 0.65f},  // mouth
      }),
      CreateOptions(1, 1, 1));
  MP_ASSERT_OK(status_or_value);
  const auto& rect = status_or_value.value();
  EXPECT_THAT(rect.x_center(), testing::FloatNear(0.454688f, kEps));
  EXPECT_THAT(rect.y_center(), testing::FloatNear(0.493056f, kEps));
  EXPECT_THAT(rect.height(), testing::FloatNear(2.14306f, kEps));
  EXPECT_THAT(rect.width(), testing::FloatNear(1.20547f, kEps));
  EXPECT_THAT(rect.rotation(), testing::FloatNear(-0.193622f, kEps));
}

TEST(FaceToRectCalculator, LandmarksToNormalizedRect) {
  auto status_or_value = RunCalculator(
      /*detection=*/DetectionWithKeyPoints({
          {0.3f, 0.5f},   // left eye
          {0.3f, 0.5f},   // left eye
          {0.6f, 0.45f},  // right eye
          {0.6f, 0.45f},  // right eye
          {0.5f, 0.65f},  // mouth
          {0.5f, 0.65f},  // mouth
      }),
      CreateOptions(2, 0, 2));
  MP_ASSERT_OK(status_or_value);
  const auto& rect = status_or_value.value();
  EXPECT_THAT(rect.x_center(), testing::FloatNear(0.454688f, kEps));
  EXPECT_THAT(rect.y_center(), testing::FloatNear(0.493056f, kEps));
  EXPECT_THAT(rect.height(), testing::FloatNear(2.14306f, kEps));
  EXPECT_THAT(rect.width(), testing::FloatNear(1.20547f, kEps));
  EXPECT_THAT(rect.rotation(), testing::FloatNear(-0.193622f, kEps));
}

TEST(FaceToRectCalculator, LandmarksToNormalizedRectObtuseAngle) {
  auto status_or_value = RunCalculator(
      /*detection=*/DetectionWithKeyPoints({
          {0.6f, 0.8f},  // left eye
          {0.6f, 0.8f},  // left eye
          {0.2f, 0.4f},  // right eye
          {0.2f, 0.4f},  // right eye
          {0.8f, 0.2f},  // mouth
          {0.8f, 0.2f},  // mouth
      }),
      CreateOptions(2, 0, 2));
  MP_ASSERT_OK(status_or_value);
  const auto& rect = status_or_value.value();
  EXPECT_THAT(rect.x_center(), testing::FloatNear(0.439844f, kEps));
  EXPECT_THAT(rect.y_center(), testing::FloatNear(0.559722f, kEps));
  EXPECT_THAT(rect.height(), testing::FloatNear(3.26389f, kEps));
  EXPECT_THAT(rect.width(), testing::FloatNear(1.83594f, kEps));
  EXPECT_THAT(rect.rotation(), testing::FloatNear(-2.35619f, kEps));
}

}  // namespace mediapipe
