// Copyright 2019 The MediaPipe Authors.
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

#include <array>
#include <cmath>
#include <vector>

#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/point2.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

constexpr char kProjectionMatrixTag[] = "PROJECTION_MATRIX";
constexpr char kDetectionsTag[] = "DETECTIONS";

using ::testing::ElementsAre;
using ::testing::FloatNear;

constexpr float kMaxError = 1e-4;

MATCHER_P2(PointEq, x, y, "") {
  bool result = testing::Value(arg.x(), FloatNear(x, kMaxError)) &&
                testing::Value(arg.y(), FloatNear(y, kMaxError));
  if (!result) {
    *result_listener << "actual: {" << arg.x() << ", " << arg.y()
                     << "}, expected: {" << x << ", " << y << "}";
  }
  return result;
}

MATCHER_P4(BoundingBoxEq, xmin, ymin, width, height, "") {
  return testing::Value(arg.xmin(), FloatNear(xmin, kMaxError)) &&
         testing::Value(arg.ymin(), FloatNear(ymin, kMaxError)) &&
         testing::Value(arg.width(), FloatNear(width, kMaxError)) &&
         testing::Value(arg.height(), FloatNear(height, kMaxError));
}

std::vector<Point2_f> GetPoints(const Detection& detection) {
  std::vector<Point2_f> points;
  const auto& location_data = detection.location_data();
  for (int i = 0; i < location_data.relative_keypoints_size(); ++i) {
    const auto& kp = location_data.relative_keypoints(i);
    points.push_back({kp.x(), kp.y()});
  }
  return points;
}

// Test helper function to run "DetectionProjectionCalculator".
absl::StatusOr<Detection> RunProjectionCalculator(
    Detection detection, std::array<float, 16> project_mat) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "DetectionProjectionCalculator"
    input_stream: "DETECTIONS:detections"
    input_stream: "PROJECTION_MATRIX:matrix"
    output_stream: "DETECTIONS:projected_detections"
  )pb"));

  runner.MutableInputs()
      ->Tag(kDetectionsTag)
      .packets.push_back(MakePacket<std::vector<Detection>>(
                             std::vector<Detection>({std::move(detection)}))
                             .At(Timestamp::PostStream()));
  runner.MutableInputs()
      ->Tag(kProjectionMatrixTag)
      .packets.push_back(
          MakePacket<std::array<float, 16>>(std::move(project_mat))
              .At(Timestamp::PostStream()));

  MP_RETURN_IF_ERROR(runner.Run());
  const std::vector<Packet>& output =
      runner.Outputs().Tag(kDetectionsTag).packets;
  RET_CHECK_EQ(output.size(), 1);
  const auto& output_detections = output[0].Get<std::vector<Detection>>();

  RET_CHECK_EQ(output_detections.size(), 1);
  return output_detections[0];
}

TEST(DetectionProjectionCalculatorTest, ProjectionFullRoiNoOp) {
  Detection detection;
  auto* location_data = detection.mutable_location_data();
  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  location_data->mutable_relative_bounding_box()->set_xmin(0.0f);
  location_data->mutable_relative_bounding_box()->set_ymin(0.0f);
  location_data->mutable_relative_bounding_box()->set_width(0.5f);
  location_data->mutable_relative_bounding_box()->set_height(0.5f);

  auto* kp = location_data->add_relative_keypoints();
  kp->set_x(0.25f);
  kp->set_y(0.25f);

  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.0f);
  roi.set_height(1.0f);
  roi.set_rotation(0.0f);

  constexpr int kImageWidth = 100;
  constexpr int kImageHeight = 100;

  RotatedRect rect;
  rect.center_x = roi.x_center() * kImageWidth;
  rect.center_y = roi.y_center() * kImageHeight;
  rect.width = roi.width() * kImageWidth;
  rect.height = roi.height() * kImageHeight;
  rect.rotation = roi.rotation();

  std::array<float, 16> projection_matrix;
  GetRotatedSubRectToRectTransformMatrix(rect, kImageWidth, kImageHeight,
                                         /*flip_horizontaly=*/false,
                                         &projection_matrix);

  auto status_or_result = RunProjectionCalculator(std::move(detection),
                                                  std::move(projection_matrix));
  MP_ASSERT_OK(status_or_result);
  const auto& result = status_or_result.value();
  ASSERT_EQ(result.location_data().format(),
            LocationData::RELATIVE_BOUNDING_BOX);
  EXPECT_THAT(result.location_data().relative_bounding_box(),
              BoundingBoxEq(0.0f, 0.0f, 0.5f, 0.5f));
  EXPECT_THAT(GetPoints(result), testing::ElementsAre(PointEq(0.25f, 0.25f)));
}

TEST(DetectionProjectionCalculatorTest, ProjectionFullRoi90Rotation) {
  Detection detection;
  auto* location_data = detection.mutable_location_data();
  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  location_data->mutable_relative_bounding_box()->set_xmin(0.0f);
  location_data->mutable_relative_bounding_box()->set_ymin(0.0f);
  location_data->mutable_relative_bounding_box()->set_width(0.5f);
  location_data->mutable_relative_bounding_box()->set_height(0.5f);

  auto* kp = location_data->add_relative_keypoints();
  kp->set_x(0.25f);
  kp->set_y(0.25f);

  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.5f);
  roi.set_y_center(0.5f);
  roi.set_width(1.0f);
  roi.set_height(1.0f);
  roi.set_rotation(90 * M_PI / 180.0f);

  constexpr int kImageWidth = 100;
  constexpr int kImageHeight = 100;

  RotatedRect rect;
  rect.center_x = roi.x_center() * kImageWidth;
  rect.center_y = roi.y_center() * kImageHeight;
  rect.width = roi.width() * kImageWidth;
  rect.height = roi.height() * kImageHeight;
  rect.rotation = roi.rotation();

  std::array<float, 16> projection_matrix;
  GetRotatedSubRectToRectTransformMatrix(rect, kImageWidth, kImageHeight,
                                         /*flip_horizontaly=*/false,
                                         &projection_matrix);

  auto status_or_result = RunProjectionCalculator(std::move(detection),
                                                  std::move(projection_matrix));
  MP_ASSERT_OK(status_or_result);
  const auto& result = status_or_result.value();
  ASSERT_EQ(result.location_data().format(),
            LocationData::RELATIVE_BOUNDING_BOX);
  EXPECT_THAT(result.location_data().relative_bounding_box(),
              BoundingBoxEq(0.5f, 0.0f, 0.5f, 0.5f));
  EXPECT_THAT(GetPoints(result), ElementsAre(PointEq(0.75f, 0.25f)));
}

TEST(DetectionProjectionCalculatorTest, ProjectionSmallerRoi) {
  Detection detection;
  auto* location_data = detection.mutable_location_data();
  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  location_data->mutable_relative_bounding_box()->set_xmin(0.5f);
  location_data->mutable_relative_bounding_box()->set_ymin(0.0f);
  location_data->mutable_relative_bounding_box()->set_width(0.5f);
  location_data->mutable_relative_bounding_box()->set_height(0.5f);

  auto* kp = location_data->add_relative_keypoints();
  kp->set_x(0.5f);
  kp->set_y(0.5f);

  mediapipe::NormalizedRect roi;
  roi.set_x_center(0.75f);
  roi.set_y_center(0.75f);
  roi.set_width(0.5f);
  roi.set_height(0.5f);
  roi.set_rotation(0.0f);

  constexpr int kImageWidth = 100;
  constexpr int kImageHeight = 100;

  RotatedRect rect;
  rect.center_x = roi.x_center() * kImageWidth;
  rect.center_y = roi.y_center() * kImageHeight;
  rect.width = roi.width() * kImageWidth;
  rect.height = roi.height() * kImageHeight;
  rect.rotation = roi.rotation();

  std::array<float, 16> projection_matrix;
  GetRotatedSubRectToRectTransformMatrix(rect, kImageWidth, kImageHeight,
                                         /*flip_horizontaly=*/false,
                                         &projection_matrix);

  auto status_or_result = RunProjectionCalculator(std::move(detection),
                                                  std::move(projection_matrix));
  MP_ASSERT_OK(status_or_result);
  const auto& result = status_or_result.value();
  ASSERT_EQ(result.location_data().format(),
            LocationData::RELATIVE_BOUNDING_BOX);
  EXPECT_THAT(result.location_data().relative_bounding_box(),
              BoundingBoxEq(0.75f, 0.5f, 0.25f, 0.25f));
  EXPECT_THAT(GetPoints(result), ElementsAre(PointEq(0.75f, 0.75f)));
}

TEST(DetectionProjectionCalculatorTest, ProjectionSmallerRoi30Rotation) {
  constexpr float kImageWidth = 80;
  constexpr float kImageHeight = 120;
  constexpr float kRectWidth = 50;
  constexpr float kRectHeight = 30;
  constexpr float kRectXCenter = 65;
  constexpr float kRectYCenter = 85;
  constexpr float kRectRotation = 30 * M_PI / 180.0f;

  Detection detection;
  auto* location_data = detection.mutable_location_data();
  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  location_data->mutable_relative_bounding_box()->set_xmin(0.0f);
  location_data->mutable_relative_bounding_box()->set_ymin(0.0f);
  location_data->mutable_relative_bounding_box()->set_width(1.0f);
  location_data->mutable_relative_bounding_box()->set_height(1.0f);
  // Expected box values were calculated manually from image.
  constexpr float kExpectedBoxXMin = 35.849f / kImageWidth;
  constexpr float kExpectedBoxYMin = 59.510f / kImageHeight;
  constexpr float kExpectedBoxWidth = 58.301f / kImageWidth;
  constexpr float kExpectedBoxHeight = 50.981f / kImageHeight;

  auto* kp1 = location_data->add_relative_keypoints();
  kp1->set_x(0.0f);
  kp1->set_y(0.0f);
  auto* kp2 = location_data->add_relative_keypoints();
  kp2->set_x(0.5f);
  kp2->set_y(0.5f);
  auto* kp3 = location_data->add_relative_keypoints();
  kp3->set_x(1.0f);
  kp3->set_y(0.0f);
  // Expected key points were calculated manually from image.
  constexpr float kExpectedPoint1X = 50.85f / kImageWidth;
  constexpr float kExpectedPoint1Y = 59.52f / kImageHeight;
  constexpr float kExpectedPoint2X = kRectXCenter / kImageWidth;
  constexpr float kExpectedPoint2Y = kRectYCenter / kImageHeight;
  constexpr float kExpectedPoint3X = 94.15f / kImageWidth;
  constexpr float kExpectedPoint3Y = 84.51f / kImageHeight;

  mediapipe::NormalizedRect roi;
  roi.set_x_center(kRectXCenter / kImageWidth);
  roi.set_y_center(kRectYCenter / kImageHeight);
  roi.set_width(kRectWidth / kImageWidth);
  roi.set_height(kRectHeight / kImageHeight);
  roi.set_rotation(kRectRotation);

  RotatedRect rect;
  rect.center_x = roi.x_center() * kImageWidth;
  rect.center_y = roi.y_center() * kImageHeight;
  rect.width = roi.width() * kImageWidth;
  rect.height = roi.height() * kImageHeight;
  rect.rotation = roi.rotation();

  std::array<float, 16> projection_matrix;
  GetRotatedSubRectToRectTransformMatrix(rect, kImageWidth, kImageHeight,
                                         /*flip_horizontaly=*/false,
                                         &projection_matrix);

  auto status_or_result = RunProjectionCalculator(std::move(detection),
                                                  std::move(projection_matrix));
  MP_ASSERT_OK(status_or_result);
  const auto& result = status_or_result.value();
  ASSERT_EQ(result.location_data().format(),
            LocationData::RELATIVE_BOUNDING_BOX);
  EXPECT_THAT(result.location_data().relative_bounding_box(),
              BoundingBoxEq(kExpectedBoxXMin, kExpectedBoxYMin,
                            kExpectedBoxWidth, kExpectedBoxHeight));
  EXPECT_THAT(GetPoints(result),
              ElementsAre(PointEq(kExpectedPoint1X, kExpectedPoint1Y),
                          PointEq(kExpectedPoint2X, kExpectedPoint2Y),
                          PointEq(kExpectedPoint3X, kExpectedPoint3Y)));
}

}  // namespace
}  // namespace mediapipe
