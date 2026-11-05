// Copyright 2022 The MediaPipe Authors.
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
#include <string>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h" // NOLINT

namespace mediapipe {

void AddInputVector(const std::vector<int> &input, int64 timestamp,
                    CalculatorRunner *runner) {
  runner->MutableInputs()->Index(0).packets.push_back(
      MakePacket<std::vector<int>>(input).At(Timestamp(timestamp)));
}

TEST(PacketRateCalculatorTest, EmptyVectorInput) {

  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "PacketRateCalculator"
        input_stream: "input_packet"
        output_stream: "packet_rate"
      )pb");

  CalculatorRunner runner(node_config);

  AddInputVector({0}, /*timestamp (microseconds)=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  EXPECT_EQ(0, runner.Outputs().Index(0).packets.size());

  AddInputVector({1}, /*timestamp=*/1001, &runner);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet> &outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, outputs.size());

  // * first packet came with timestamp 1us
  // * second packet has timestamp 1001us
  // So period is 1ms, which equal to rate 1000 packets per second.

  auto &v = outputs[0].Get<float>();
  EXPECT_FLOAT_EQ(1e+3, v);
}

} // namespace mediapipe