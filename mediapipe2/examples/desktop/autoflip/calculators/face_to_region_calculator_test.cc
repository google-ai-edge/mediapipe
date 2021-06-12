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

#include "absl/strings/string_view.h"
#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/examples/desktop/autoflip/calculators/face_to_region_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"

using mediapipe::Detection;

namespace mediapipe {
namespace autoflip {
namespace {

const char kConfig[] = R"(
    calculator: "FaceToRegionCalculator"
    input_stream: "VIDEO:frames"
    input_stream: "FACES:faces"
    output_stream: "REGIONS:regions"
    )";

const char kConfigNoVideo[] = R"(
    calculator: "FaceToRegionCalculator"
    input_stream: "FACES:faces"
    output_stream: "REGIONS:regions"
    )";

const char kFace1[] = R"(location_data {
           format: RELATIVE_BOUNDING_BOX
           relative_bounding_box {
             xmin: -0.00375
             ymin: 0.003333
             width: 0.125
             height: 0.33333
           }
           relative_keypoints { x: 0.03125 y: 0.05 }
           relative_keypoints { x: 0.0875 y: 0.0666666 }
           relative_keypoints { x: 0.03125 y: 0.05 }
           relative_keypoints { x: 0.0875 y: 0.0666666 }
           relative_keypoints { x: 0.0250 y: 0.0666666 }
           relative_keypoints { x: 0.0950 y: 0.0666666 }
         })";

const char kFace2[] = R"(location_data {
           format: RELATIVE_BOUNDING_BOX
           relative_bounding_box {
             xmin: 0.0025
             ymin: 0.005
             width: 0.25
             height: 0.5
           }
           relative_keypoints { x: 0 y: 0 }
           relative_keypoints { x: 0 y: 0 }
           relative_keypoints { x: 0 y: 0 }
           relative_keypoints { x: 0 y: 0 }
           relative_keypoints { x: 0 y: 0 }
           relative_keypoints { x: 0 y: 0 }
         })";

const char kFace3[] = R"(location_data {
           format: RELATIVE_BOUNDING_BOX
           relative_bounding_box {
             xmin: 0.0
             ymin: 0.0
             width: 0.5
             height: 0.5
           }
           relative_keypoints { x: 0 y: 0 }
           relative_keypoints { x: 0 y: 0 }
           relative_keypoints { x: 0 y: 0 }
           relative_keypoints { x: 0 y: 0 }
           relative_keypoints { x: 0 y: 0 }
           relative_keypoints { x: 0 y: 0 }
         })";

void SetInputs(const std::vector<std::string>& faces, const bool include_video,
               CalculatorRunner* runner) {
  // Setup an input video frame.
  if (include_video) {
    auto input_frame =
        ::absl::make_unique<ImageFrame>(ImageFormat::SRGB, 800, 600);
    runner->MutableInputs()->Tag("VIDEO").packets.push_back(
        Adopt(input_frame.release()).At(Timestamp::PostStream()));
  }
  // Setup two faces as input.
  auto input_faces = ::absl::make_unique<std::vector<Detection>>();
  // A face with landmarks.
  for (const auto& face : faces) {
    input_faces->push_back(ParseTextProtoOrDie<Detection>(face));
  }
  runner->MutableInputs()->Tag("FACES").packets.push_back(
      Adopt(input_faces.release()).At(Timestamp::PostStream()));
}

CalculatorGraphConfig::Node MakeConfig(std::string base_config, bool whole_face,
                                       bool landmarks, bool bb_from_landmarks,
                                       bool visual_scoring) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(base_config);
  config.mutable_options()
      ->MutableExtension(FaceToRegionCalculatorOptions::ext)
      ->set_export_whole_face(whole_face);
  config.mutable_options()
      ->MutableExtension(FaceToRegionCalculatorOptions::ext)
      ->set_export_individual_face_landmarks(landmarks);
  config.mutable_options()
      ->MutableExtension(FaceToRegionCalculatorOptions::ext)
      ->set_export_bbox_from_landmarks(bb_from_landmarks);
  config.mutable_options()
      ->MutableExtension(FaceToRegionCalculatorOptions::ext)
      ->set_use_visual_scorer(visual_scoring);

  return config;
}

TEST(FaceToRegionCalculatorTest, FaceFullTypeSize) {
  // Setup test
  auto runner = ::absl::make_unique<CalculatorRunner>(
      MakeConfig(kConfig, true, false, false, true));
  SetInputs({kFace1, kFace2}, true, runner.get());

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  // Check the output regions.
  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag("REGIONS").packets;
  ASSERT_EQ(1, output_packets.size());

  const auto& regions = output_packets[0].Get<DetectionSet>();
  ASSERT_EQ(2, regions.detections().size());
  auto face_1 = regions.detections(0);
  EXPECT_EQ(face_1.signal_type().standard(), SignalType::FACE_FULL);
  EXPECT_FLOAT_EQ(face_1.location_normalized().x(), 0);
  EXPECT_FLOAT_EQ(face_1.location_normalized().y(), 0.003333);
  EXPECT_FLOAT_EQ(face_1.location_normalized().width(), 0.12125);
  EXPECT_FLOAT_EQ(face_1.location_normalized().height(), 0.33333);
  EXPECT_FLOAT_EQ(face_1.score(), 0.040214583);

  auto face_2 = regions.detections(1);
  EXPECT_EQ(face_2.signal_type().standard(), SignalType::FACE_FULL);
  EXPECT_FLOAT_EQ(face_2.location_normalized().x(), 0.0025);
  EXPECT_FLOAT_EQ(face_2.location_normalized().y(), 0.005);
  EXPECT_FLOAT_EQ(face_2.location_normalized().width(), 0.25);
  EXPECT_FLOAT_EQ(face_2.location_normalized().height(), 0.5);
  EXPECT_FLOAT_EQ(face_2.score(), 0.125);
}

TEST(FaceToRegionCalculatorTest, FaceLandmarksTypeSize) {
  // Setup test
  auto runner = ::absl::make_unique<CalculatorRunner>(
      MakeConfig(kConfig, false, true, false, true));
  SetInputs({kFace1}, true, runner.get());

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  // Check the output regions.
  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag("REGIONS").packets;
  ASSERT_EQ(1, output_packets.size());

  const auto& regions = output_packets[0].Get<DetectionSet>();
  ASSERT_EQ(6, regions.detections().size());
  auto landmark_1 = regions.detections(0);
  EXPECT_EQ(landmark_1.signal_type().standard(), SignalType::FACE_LANDMARK);
  EXPECT_FLOAT_EQ(landmark_1.location_normalized().x(), 0.03125);
  EXPECT_FLOAT_EQ(landmark_1.location_normalized().y(), 0.05);
  EXPECT_FLOAT_EQ(landmark_1.location_normalized().width(), 0.00125);
  EXPECT_FLOAT_EQ(landmark_1.location_normalized().height(), 0.0016666667);

  auto landmark_2 = regions.detections(1);
  EXPECT_EQ(landmark_2.signal_type().standard(), SignalType::FACE_LANDMARK);
  EXPECT_FLOAT_EQ(landmark_2.location_normalized().x(), 0.0875);
  EXPECT_FLOAT_EQ(landmark_2.location_normalized().y(), 0.0666666);
  EXPECT_FLOAT_EQ(landmark_2.location_normalized().width(), 0.00125);
  EXPECT_FLOAT_EQ(landmark_2.location_normalized().height(), 0.0016666667);
}

TEST(FaceToRegionCalculatorTest, FaceLandmarksBox) {
  // Setup test
  auto runner = ::absl::make_unique<CalculatorRunner>(
      MakeConfig(kConfig, false, false, true, true));
  SetInputs({kFace1}, true, runner.get());

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  // Check the output regions.
  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag("REGIONS").packets;
  ASSERT_EQ(1, output_packets.size());

  const auto& regions = output_packets[0].Get<DetectionSet>();
  ASSERT_EQ(2, regions.detections().size());
  auto landmark_1 = regions.detections(0);
  EXPECT_EQ(landmark_1.signal_type().standard(),
            SignalType::FACE_CORE_LANDMARKS);
  EXPECT_FLOAT_EQ(landmark_1.location_normalized().x(), 0.03125);
  EXPECT_FLOAT_EQ(landmark_1.location_normalized().y(), 0.05);
  EXPECT_FLOAT_EQ(landmark_1.location_normalized().width(), 0.056249999);
  EXPECT_FLOAT_EQ(landmark_1.location_normalized().height(), 0.016666602);
  EXPECT_FLOAT_EQ(landmark_1.score(), 0.00084375002);

  auto landmark_2 = regions.detections(1);
  EXPECT_EQ(landmark_2.signal_type().standard(),
            SignalType::FACE_ALL_LANDMARKS);
  EXPECT_FLOAT_EQ(landmark_2.location_normalized().x(), 0.025);
  EXPECT_FLOAT_EQ(landmark_2.location_normalized().y(), 0.050000001);
  EXPECT_FLOAT_EQ(landmark_2.location_normalized().width(), 0.07);
  EXPECT_FLOAT_EQ(landmark_2.location_normalized().height(), 0.016666602);
  EXPECT_FLOAT_EQ(landmark_2.score(), 0.00105);
}

TEST(FaceToRegionCalculatorTest, FaceScore) {
  // Setup test
  auto runner = ::absl::make_unique<CalculatorRunner>(
      MakeConfig(kConfig, true, false, false, true));
  SetInputs({kFace3}, true, runner.get());

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  // Check the output regions.
  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag("REGIONS").packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& regions = output_packets[0].Get<DetectionSet>();
  ASSERT_EQ(1, regions.detections().size());
  auto landmark_1 = regions.detections(0);
  EXPECT_FLOAT_EQ(landmark_1.score(), 0.25);
}

TEST(FaceToRegionCalculatorTest, FaceNoVideoVisualScoreFail) {
  // Setup test
  auto runner = ::absl::make_unique<CalculatorRunner>(
      MakeConfig(kConfigNoVideo, true, false, false, true));
  SetInputs({kFace3}, false, runner.get());

  // Run the calculator.
  ASSERT_FALSE(runner->Run().ok());
}

TEST(FaceToRegionCalculatorTest, FaceNoVideoLandmarksFail) {
  // Setup test
  auto runner = ::absl::make_unique<CalculatorRunner>(
      MakeConfig(kConfigNoVideo, false, true, false, false));
  SetInputs({kFace3}, false, runner.get());

  // Run the calculator.
  ASSERT_FALSE(runner->Run().ok());
}

TEST(FaceToRegionCalculatorTest, FaceNoVideoBBLandmarksFail) {
  // Setup test
  auto runner = ::absl::make_unique<CalculatorRunner>(
      MakeConfig(kConfigNoVideo, false, false, true, false));
  SetInputs({kFace3}, false, runner.get());

  // Run the calculator.
  ASSERT_FALSE(runner->Run().ok());
}

TEST(FaceToRegionCalculatorTest, FaceNoVideoPass) {
  // Setup test
  auto runner = ::absl::make_unique<CalculatorRunner>(
      MakeConfig(kConfigNoVideo, true, false, false, false));
  SetInputs({kFace1, kFace2}, false, runner.get());

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  // Check the output regions.
  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag("REGIONS").packets;
  ASSERT_EQ(1, output_packets.size());

  const auto& regions = output_packets[0].Get<DetectionSet>();
  ASSERT_EQ(2, regions.detections().size());
  auto face_1 = regions.detections(0);
  EXPECT_EQ(face_1.signal_type().standard(), SignalType::FACE_FULL);
  EXPECT_FLOAT_EQ(face_1.location_normalized().x(), 0);
  EXPECT_FLOAT_EQ(face_1.location_normalized().y(), 0.003333);
  EXPECT_FLOAT_EQ(face_1.location_normalized().width(), 0.12125);
  EXPECT_FLOAT_EQ(face_1.location_normalized().height(), 0.33333);
  EXPECT_FLOAT_EQ(face_1.score(), 1);

  auto face_2 = regions.detections(1);
  EXPECT_EQ(face_2.signal_type().standard(), SignalType::FACE_FULL);
  EXPECT_FLOAT_EQ(face_2.location_normalized().x(), 0.0025);
  EXPECT_FLOAT_EQ(face_2.location_normalized().y(), 0.005);
  EXPECT_FLOAT_EQ(face_2.location_normalized().width(), 0.25);
  EXPECT_FLOAT_EQ(face_2.location_normalized().height(), 0.5);
  EXPECT_FLOAT_EQ(face_2.score(), 1);
}

}  // namespace
}  // namespace autoflip
}  // namespace mediapipe
