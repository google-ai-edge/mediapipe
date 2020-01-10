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
#include "mediapipe/examples/desktop/autoflip/calculators/signal_fusing_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"

using mediapipe::autoflip::DetectionSet;

namespace mediapipe {
namespace autoflip {
namespace {

const char kConfigA[] = R"(
    calculator: "SignalFusingCalculator"
    input_stream: "scene_change"
    input_stream: "detection_set_a"
    input_stream: "detection_set_b"
    output_stream: "salient_region"
    options:{
    [mediapipe.autoflip.SignalFusingCalculatorOptions.ext]:{
      signal_settings{
        type: {standard: FACE_FULL}
        min_score: 0.5
        max_score: 0.6
      }
      signal_settings{
        type: {standard: TEXT}
        min_score: 0.9
        max_score: 1.0
      }
    }
    })";

const char kConfigB[] = R"(
    calculator: "SignalFusingCalculator"
    input_stream: "scene_change"
    input_stream: "detection_set_a"
    input_stream: "detection_set_b"
    input_stream: "detection_set_c"
    output_stream: "salient_region"
    options:{
    [mediapipe.autoflip.SignalFusingCalculatorOptions.ext]:{
      signal_settings{
        type: {standard: FACE_FULL}
        min_score: 0.5
        max_score: 0.6
      }
      signal_settings{
        type: {custom: "text"}
        min_score: 0.9
        max_score: 1.0
      }
      signal_settings{
        type: {standard: LOGO}
        min_score: 0.1
        max_score: 0.3
      }
    }
    })";

TEST(SignalFusingCalculatorTest, TwoInputNoTracking) {
  auto runner = absl::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigA));

  auto input_border = absl::make_unique<bool>(false);
  runner->MutableInputs()->Index(0).packets.push_back(
      Adopt(input_border.release()).At(Timestamp(0)));

  auto input_face =
      absl::make_unique<DetectionSet>(ParseTextProtoOrDie<DetectionSet>(
          R"(
            detections {
              score: 0.5
              signal_type: { standard: FACE_FULL }
            }
            detections {
              score: 0.3
              signal_type: { standard: FACE_FULL }
            }
          )"));

  runner->MutableInputs()->Index(1).packets.push_back(
      Adopt(input_face.release()).At(Timestamp(0)));

  auto input_ocr =
      absl::make_unique<DetectionSet>(ParseTextProtoOrDie<DetectionSet>(
          R"(
            detections {
              score: 0.3
              signal_type: { standard: TEXT }
            }
            detections {
              score: 0.9
              signal_type: { standard: TEXT }
            }
          )"));

  runner->MutableInputs()->Index(2).packets.push_back(
      Adopt(input_ocr.release()).At(Timestamp(0)));

  MP_ASSERT_OK(runner->Run());

  const std::vector<Packet>& output_packets =
      runner->Outputs().Index(0).packets;
  const auto& detection_set = output_packets[0].Get<DetectionSet>();

  ASSERT_EQ(detection_set.detections().size(), 4);
  EXPECT_FLOAT_EQ(detection_set.detections(0).score(), .55);
  EXPECT_FLOAT_EQ(detection_set.detections(1).score(), .53);
  EXPECT_FLOAT_EQ(detection_set.detections(2).score(), .93);
  EXPECT_FLOAT_EQ(detection_set.detections(3).score(), .99);
}

TEST(SignalFusingCalculatorTest, ThreeInputTracking) {
  auto runner = absl::make_unique<CalculatorRunner>(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfigB));

  auto input_border_0 = absl::make_unique<bool>(false);
  runner->MutableInputs()->Index(0).packets.push_back(
      Adopt(input_border_0.release()).At(Timestamp(0)));

  // Time zero.
  auto input_face_0 =
      absl::make_unique<DetectionSet>(ParseTextProtoOrDie<DetectionSet>(
          R"(
            detections {
              score: 0.2
              signal_type: { standard: FACE_FULL }
              tracking_id: 0
            }
            detections {
              score: 0.0
              signal_type: { standard: FACE_FULL }
              tracking_id: 1
            }
            detections {
              score: 0.1
              signal_type: { standard: FACE_FULL }
            }
          )"));

  runner->MutableInputs()->Index(1).packets.push_back(
      Adopt(input_face_0.release()).At(Timestamp(0)));

  auto input_ocr_0 =
      absl::make_unique<DetectionSet>(ParseTextProtoOrDie<DetectionSet>(
          R"(
            detections {
              score: 0.2
              signal_type: { custom: "text" }
            }
          )"));

  runner->MutableInputs()->Index(2).packets.push_back(
      Adopt(input_ocr_0.release()).At(Timestamp(0)));

  auto input_agn_0 =
      absl::make_unique<DetectionSet>(ParseTextProtoOrDie<DetectionSet>(
          R"(
            detections {
              score: 0.3
              signal_type: { standard: LOGO }
              tracking_id: 0
            }
          )"));

  runner->MutableInputs()->Index(3).packets.push_back(
      Adopt(input_agn_0.release()).At(Timestamp(0)));

  // Time one
  auto input_border_1 = absl::make_unique<bool>(false);
  runner->MutableInputs()->Index(0).packets.push_back(
      Adopt(input_border_1.release()).At(Timestamp(1)));

  auto input_face_1 =
      absl::make_unique<DetectionSet>(ParseTextProtoOrDie<DetectionSet>(
          R"(
            detections {
              score: 0.7
              signal_type: { standard: FACE_FULL }
              tracking_id: 0
            }
            detections {
              score: 0.9
              signal_type: { standard: FACE_FULL }
              tracking_id: 1
            }
            detections {
              score: 0.2
              signal_type: { standard: FACE_FULL }
            }
          )"));

  runner->MutableInputs()->Index(1).packets.push_back(
      Adopt(input_face_1.release()).At(Timestamp(1)));

  auto input_ocr_1 =
      absl::make_unique<DetectionSet>(ParseTextProtoOrDie<DetectionSet>(
          R"(
            detections {
              score: 0.3
              signal_type: { custom: "text" }
            }
          )"));

  runner->MutableInputs()->Index(2).packets.push_back(
      Adopt(input_ocr_1.release()).At(Timestamp(1)));

  auto input_agn_1 =
      absl::make_unique<DetectionSet>(ParseTextProtoOrDie<DetectionSet>(
          R"(
            detections {
              score: 0.3
              signal_type: { standard: LOGO }
              tracking_id: 0
            }
          )"));

  runner->MutableInputs()->Index(3).packets.push_back(
      Adopt(input_agn_1.release()).At(Timestamp(1)));

  // Time two
  auto input_border_2 = absl::make_unique<bool>(false);
  runner->MutableInputs()->Index(0).packets.push_back(
      Adopt(input_border_2.release()).At(Timestamp(2)));

  auto input_face_2 =
      absl::make_unique<DetectionSet>(ParseTextProtoOrDie<DetectionSet>(
          R"(
            detections {
              score: 0.8
              signal_type: { standard: FACE_FULL }
              tracking_id: 0
            }
            detections {
              score: 0.9
              signal_type: { standard: FACE_FULL }
              tracking_id: 1
            }
            detections {
              score: 0.3
              signal_type: { standard: FACE_FULL }
            }
          )"));

  runner->MutableInputs()->Index(1).packets.push_back(
      Adopt(input_face_2.release()).At(Timestamp(2)));

  auto input_ocr_2 =
      absl::make_unique<DetectionSet>(ParseTextProtoOrDie<DetectionSet>(
          R"(
            detections {
              score: 0.3
              signal_type: { custom: "text" }
            }
          )"));

  runner->MutableInputs()->Index(2).packets.push_back(
      Adopt(input_ocr_2.release()).At(Timestamp(2)));

  auto input_agn_2 =
      absl::make_unique<DetectionSet>(ParseTextProtoOrDie<DetectionSet>(
          R"(
            detections {
              score: 0.9
              signal_type: { standard: LOGO }
              tracking_id: 0
            }
          )"));

  runner->MutableInputs()->Index(3).packets.push_back(
      Adopt(input_agn_2.release()).At(Timestamp(2)));

  // Time three (new scene)
  auto input_border_3 = absl::make_unique<bool>(true);
  runner->MutableInputs()->Index(0).packets.push_back(
      Adopt(input_border_3.release()).At(Timestamp(3)));

  auto input_face_3 =
      absl::make_unique<DetectionSet>(ParseTextProtoOrDie<DetectionSet>(
          R"(
            detections {
              score: 0.2
              signal_type: { standard: FACE_FULL }
              tracking_id: 0
            }
            detections {
              score: 0.3
              signal_type: { standard: FACE_FULL }
              tracking_id: 1
            }
            detections {
              score: 0.4
              signal_type: { standard: FACE_FULL }
            }
          )"));

  runner->MutableInputs()->Index(1).packets.push_back(
      Adopt(input_face_3.release()).At(Timestamp(3)));

  auto input_ocr_3 =
      absl::make_unique<DetectionSet>(ParseTextProtoOrDie<DetectionSet>(
          R"(
            detections {
              score: 0.5
              signal_type: { custom: "text" }
            }
          )"));

  runner->MutableInputs()->Index(2).packets.push_back(
      Adopt(input_ocr_3.release()).At(Timestamp(3)));

  auto input_agn_3 =
      absl::make_unique<DetectionSet>(ParseTextProtoOrDie<DetectionSet>(
          R"(
            detections {
              score: 0.6
              signal_type: { standard: LOGO }
              tracking_id: 0
            }
          )"));

  runner->MutableInputs()->Index(3).packets.push_back(
      Adopt(input_agn_3.release()).At(Timestamp(3)));

  MP_ASSERT_OK(runner->Run());

  // Check time 0
  std::vector<Packet> output_packets = runner->Outputs().Index(0).packets;
  DetectionSet detection_set = output_packets[0].Get<DetectionSet>();

  float face_id_0 = (.2 + .7 + .8) / 3;
  face_id_0 = face_id_0 * .1 + .5;
  float face_id_1 = (0.0 + .9 + .9) / 3;
  face_id_1 = face_id_1 * .1 + .5;
  float face_3 = 0.1;
  face_3 = face_3 * .1 + .5;
  float ocr_1 = 0.2;
  ocr_1 = ocr_1 * .1 + .9;
  float agn_1 = (.3 + .3 + .9) / 3;
  agn_1 = agn_1 * .2 + .1;

  ASSERT_EQ(detection_set.detections().size(), 5);
  EXPECT_FLOAT_EQ(detection_set.detections(0).score(), face_id_0);
  EXPECT_FLOAT_EQ(detection_set.detections(1).score(), face_id_1);
  EXPECT_FLOAT_EQ(detection_set.detections(2).score(), face_3);
  EXPECT_FLOAT_EQ(detection_set.detections(3).score(), ocr_1);
  EXPECT_FLOAT_EQ(detection_set.detections(4).score(), agn_1);

  // Check time 1
  detection_set = output_packets[1].Get<DetectionSet>();

  face_id_0 = (.2 + .7 + .8) / 3;
  face_id_0 = face_id_0 * .1 + .5;
  face_id_1 = (0.0 + .9 + .9) / 3;
  face_id_1 = face_id_1 * .1 + .5;
  face_3 = 0.2;
  face_3 = face_3 * .1 + .5;
  ocr_1 = 0.3;
  ocr_1 = ocr_1 * .1 + .9;
  agn_1 = (.3 + .3 + .9) / 3;
  agn_1 = agn_1 * .2 + .1;

  ASSERT_EQ(detection_set.detections().size(), 5);
  EXPECT_FLOAT_EQ(detection_set.detections(0).score(), face_id_0);
  EXPECT_FLOAT_EQ(detection_set.detections(1).score(), face_id_1);
  EXPECT_FLOAT_EQ(detection_set.detections(2).score(), face_3);
  EXPECT_FLOAT_EQ(detection_set.detections(3).score(), ocr_1);
  EXPECT_FLOAT_EQ(detection_set.detections(4).score(), agn_1);

  // Check time 2
  detection_set = output_packets[2].Get<DetectionSet>();

  face_id_0 = (.2 + .7 + .8) / 3;
  face_id_0 = face_id_0 * .1 + .5;
  face_id_1 = (0.0 + .9 + .9) / 3;
  face_id_1 = face_id_1 * .1 + .5;
  face_3 = 0.3;
  face_3 = face_3 * .1 + .5;
  ocr_1 = 0.3;
  ocr_1 = ocr_1 * .1 + .9;
  agn_1 = (.3 + .3 + .9) / 3;
  agn_1 = agn_1 * .2 + .1;

  ASSERT_EQ(detection_set.detections().size(), 5);
  EXPECT_FLOAT_EQ(detection_set.detections(0).score(), face_id_0);
  EXPECT_FLOAT_EQ(detection_set.detections(1).score(), face_id_1);
  EXPECT_FLOAT_EQ(detection_set.detections(2).score(), face_3);
  EXPECT_FLOAT_EQ(detection_set.detections(3).score(), ocr_1);
  EXPECT_FLOAT_EQ(detection_set.detections(4).score(), agn_1);

  // Check time 3 (new scene)
  detection_set = output_packets[3].Get<DetectionSet>();

  face_id_0 = 0.2;
  face_id_0 = face_id_0 * .1 + .5;
  face_id_1 = 0.3;
  face_id_1 = face_id_1 * .1 + .5;
  face_3 = 0.4;
  face_3 = face_3 * .1 + .5;
  ocr_1 = 0.5;
  ocr_1 = ocr_1 * .1 + .9;
  agn_1 = .6;
  agn_1 = agn_1 * .2 + .1;

  ASSERT_EQ(detection_set.detections().size(), 5);
  EXPECT_FLOAT_EQ(detection_set.detections(0).score(), face_id_0);
  EXPECT_FLOAT_EQ(detection_set.detections(1).score(), face_id_1);
  EXPECT_FLOAT_EQ(detection_set.detections(2).score(), face_3);
  EXPECT_FLOAT_EQ(detection_set.detections(3).score(), ocr_1);
  EXPECT_FLOAT_EQ(detection_set.detections(4).score(), agn_1);
}

}  // namespace
}  // namespace autoflip
}  // namespace mediapipe
