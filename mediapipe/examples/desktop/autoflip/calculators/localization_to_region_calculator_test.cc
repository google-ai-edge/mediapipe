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
#include "mediapipe/examples/desktop/autoflip/calculators/localization_to_region_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
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
    calculator: "LocalizationToRegionCalculator"
    input_stream: "DETECTIONS:detections"
    output_stream: "REGIONS:regions"
    )";

const char kCar[] = R"(
           label: "car"
           location_data {
           format: RELATIVE_BOUNDING_BOX
           relative_bounding_box {
             xmin: -0.00375
             ymin: 0.003333
             width: 0.125
             height: 0.33333
           }
         })";

const char kDog[] = R"(
          label: "dog"
           location_data {
           format: RELATIVE_BOUNDING_BOX
           relative_bounding_box {
             xmin: 0.0025
             ymin: 0.005
             width: 0.25
             height: 0.5
           }
         })";

const char kZebra[] = R"(
          label: "zebra"
          location_data {
           format: RELATIVE_BOUNDING_BOX
           relative_bounding_box {
             xmin: 0.0
             ymin: 0.0
             width: 0.5
             height: 0.5
           }
         })";

void SetInputs(CalculatorRunner* runner,
               const std::vector<std::string>& detections) {
  auto inputs = ::absl::make_unique<std::vector<Detection>>();
  // A face with landmarks.
  for (const auto& detection : detections) {
    inputs->push_back(ParseTextProtoOrDie<Detection>(detection));
  }
  runner->MutableInputs()
      ->Tag("DETECTIONS")
      .packets.push_back(Adopt(inputs.release()).At(Timestamp::PostStream()));
}

CalculatorGraphConfig::Node MakeConfig(bool output_standard, bool output_all) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(kConfig);

  config.mutable_options()
      ->MutableExtension(LocalizationToRegionCalculatorOptions::ext)
      ->set_output_standard_signals(output_standard);

  config.mutable_options()
      ->MutableExtension(LocalizationToRegionCalculatorOptions::ext)
      ->set_output_all_signals(output_all);

  return config;
}

TEST(LocalizationToRegionCalculatorTest, StandardTypes) {
  // Setup test
  auto runner = ::absl::make_unique<CalculatorRunner>(MakeConfig(true, false));
  SetInputs(runner.get(), {kCar, kDog, kZebra});

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  // Check the output regions.
  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag("REGIONS").packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& regions = output_packets[0].Get<DetectionSet>();
  ASSERT_EQ(2, regions.detections().size());
  const auto& detection = regions.detections(0);
  EXPECT_EQ(detection.signal_type().standard(), SignalType::CAR);
  EXPECT_FLOAT_EQ(detection.location_normalized().x(), -0.00375);
  EXPECT_FLOAT_EQ(detection.location_normalized().y(), 0.003333);
  EXPECT_FLOAT_EQ(detection.location_normalized().width(), 0.125);
  EXPECT_FLOAT_EQ(detection.location_normalized().height(), 0.33333);
  const auto& detection_1 = regions.detections(1);
  EXPECT_EQ(detection_1.signal_type().standard(), SignalType::PET);
  EXPECT_FLOAT_EQ(detection_1.location_normalized().x(), 0.0025);
  EXPECT_FLOAT_EQ(detection_1.location_normalized().y(), 0.005);
  EXPECT_FLOAT_EQ(detection_1.location_normalized().width(), 0.25);
  EXPECT_FLOAT_EQ(detection_1.location_normalized().height(), 0.5);
}

TEST(LocalizationToRegionCalculatorTest, AllTypes) {
  // Setup test
  auto runner = ::absl::make_unique<CalculatorRunner>(MakeConfig(false, true));
  SetInputs(runner.get(), {kCar, kDog, kZebra});

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  // Check the output regions.
  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag("REGIONS").packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& regions = output_packets[0].Get<DetectionSet>();
  ASSERT_EQ(3, regions.detections().size());
}

TEST(LocalizationToRegionCalculatorTest, BothTypes) {
  // Setup test
  auto runner = ::absl::make_unique<CalculatorRunner>(MakeConfig(true, true));
  SetInputs(runner.get(), {kCar, kDog, kZebra});

  // Run the calculator.
  MP_ASSERT_OK(runner->Run());

  // Check the output regions.
  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag("REGIONS").packets;
  ASSERT_EQ(1, output_packets.size());
  const auto& regions = output_packets[0].Get<DetectionSet>();
  ASSERT_EQ(5, regions.detections().size());
}

}  // namespace
}  // namespace autoflip
}  // namespace mediapipe
