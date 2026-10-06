// Copyright 2026 The MediaPipe Authors.
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

#include <memory>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using ::mediapipe::ParseTextProtoOrDie;

TEST(AddAudioHeaderCalculatorTest, AddsHeaderWithOptions) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "AddAudioHeaderCalculator"
        input_stream: "audio_in"
        output_stream: "audio_out"
        node_options: {
          [type.googleapis.com/mediapipe.AddAudioHeaderCalculatorOptions] {
            sample_rate: 16000.0
            num_channels: 2
          }
        }
      )pb");
  CalculatorRunner runner(node_config);

  auto input_matrix = std::make_unique<Matrix>(2, 160);
  input_matrix->setRandom();
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(input_matrix.release()).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());

  const auto& output_packets = runner.Outputs().Index(0).packets;
  ASSERT_EQ(output_packets.size(), 1);
  EXPECT_EQ(output_packets[0].Get<Matrix>().rows(), 2);
  EXPECT_EQ(output_packets[0].Get<Matrix>().cols(), 160);

  ASSERT_TRUE(runner.Outputs()
                  .Index(0)
                  .header.ValidateAsType<mediapipe::TimeSeriesHeader>()
                  .ok());
  const auto& header =
      runner.Outputs().Index(0).header.Get<mediapipe::TimeSeriesHeader>();
  EXPECT_EQ(header.sample_rate(), 16000.0);
  EXPECT_EQ(header.num_channels(), 2);
}

TEST(AddAudioHeaderCalculatorTest, AddsHeaderWithDefaults) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "AddAudioHeaderCalculator"
        input_stream: "audio_in"
        output_stream: "audio_out"
      )pb");
  CalculatorRunner runner(node_config);

  auto input_matrix = std::make_unique<Matrix>(1, 480);
  input_matrix->setRandom();
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(input_matrix.release()).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());

  const auto& output_packets = runner.Outputs().Index(0).packets;
  ASSERT_EQ(output_packets.size(), 1);
  EXPECT_EQ(output_packets[0].Get<Matrix>().rows(), 1);
  EXPECT_EQ(output_packets[0].Get<Matrix>().cols(), 480);

  ASSERT_TRUE(runner.Outputs()
                  .Index(0)
                  .header.ValidateAsType<mediapipe::TimeSeriesHeader>()
                  .ok());
  const auto& header =
      runner.Outputs().Index(0).header.Get<mediapipe::TimeSeriesHeader>();
  EXPECT_EQ(header.sample_rate(), 48000.0);
  EXPECT_EQ(header.num_channels(), 1);
}

}  // namespace

}  // namespace mediapipe
