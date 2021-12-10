// Copyright 2019-2020 The MediaPipe Authors.
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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

namespace {

constexpr char kDisallowTag[] = "DISALLOW";
constexpr char kAllowTag[] = "ALLOW";

class GateCalculatorTest : public ::testing::Test {
 protected:
  // Helper to run a graph and return status.
  static absl::Status RunGraph(const std::string& proto) {
    auto runner = absl::make_unique<CalculatorRunner>(
        ParseTextProtoOrDie<CalculatorGraphConfig::Node>(proto));
    return runner->Run();
  }

  // Use this when ALLOW/DISALLOW input is provided as a side packet.
  void RunTimeStep(int64 timestamp, bool stream_payload) {
    runner_->MutableInputs()->Get("", 0).packets.push_back(
        MakePacket<bool>(stream_payload).At(Timestamp(timestamp)));
    MP_ASSERT_OK(runner_->Run()) << "Calculator execution failed.";
  }

  // Use this when ALLOW/DISALLOW input is provided as an input stream.
  void RunTimeStep(int64 timestamp, const std::string& control_tag,
                   bool control) {
    runner_->MutableInputs()->Get("", 0).packets.push_back(
        MakePacket<bool>(true).At(Timestamp(timestamp)));
    runner_->MutableInputs()
        ->Tag(control_tag)
        .packets.push_back(MakePacket<bool>(control).At(Timestamp(timestamp)));
    MP_ASSERT_OK(runner_->Run()) << "Calculator execution failed.";
  }

  void SetRunner(const std::string& proto) {
    runner_ = absl::make_unique<CalculatorRunner>(
        ParseTextProtoOrDie<CalculatorGraphConfig::Node>(proto));
  }

  CalculatorRunner* runner() { return runner_.get(); }

 private:
  std::unique_ptr<CalculatorRunner> runner_;
};

TEST_F(GateCalculatorTest, InvalidInputs) {
  EXPECT_TRUE(absl::IsInternal(GateCalculatorTest::RunGraph(R"(
        calculator: "GateCalculator"
        input_stream: "test_input"
        input_stream: "ALLOW:gating_stream"
        input_stream: "DISALLOW:gating_stream"
        output_stream: "test_output"
  )")));

  EXPECT_TRUE(absl::IsInternal(GateCalculatorTest::RunGraph(R"(
        calculator: "GateCalculator"
        input_stream: "test_input"
        input_side_packet: "ALLOW:gating_stream"
        input_side_packet: "DISALLOW:gating_stream"
        output_stream: "test_output"
  )")));

  EXPECT_TRUE(absl::IsInternal(GateCalculatorTest::RunGraph(R"(
        calculator: "GateCalculator"
        input_stream: "test_input"
        input_stream: "ALLOW:gating_stream"
        input_side_packet: "ALLOW:gating_stream"
        output_stream: "test_output"
  )")));

  EXPECT_TRUE(absl::IsInternal(GateCalculatorTest::RunGraph(R"(
        calculator: "GateCalculator"
        input_stream: "test_input"
        input_stream: "DISALLOW:gating_stream"
        input_side_packet: "DISALLOW:gating_stream"
        output_stream: "test_output"
  )")));

  EXPECT_TRUE(absl::IsInternal(GateCalculatorTest::RunGraph(R"(
        calculator: "GateCalculator"
        input_stream: "test_input"
        input_stream: "ALLOW:gating_stream"
        input_side_packet: "DISALLOW:gating_stream"
        output_stream: "test_output"
  )")));

  EXPECT_TRUE(absl::IsInternal(GateCalculatorTest::RunGraph(R"(
        calculator: "GateCalculator"
        input_stream: "test_input"
        input_stream: "DISALLOW:gating_stream"
        input_side_packet: "ALLOW:gating_stream"
        output_stream: "test_output"
  )")));
}

TEST_F(GateCalculatorTest, AllowByALLOWOptionToTrue) {
  SetRunner(R"(
        calculator: "GateCalculator"
        input_stream: "test_input"
        output_stream: "test_output"
        options: {
          [mediapipe.GateCalculatorOptions.ext] {
            allow: true
          }
        }
  )");

  constexpr int64 kTimestampValue0 = 42;
  RunTimeStep(kTimestampValue0, true);
  constexpr int64 kTimestampValue1 = 43;
  RunTimeStep(kTimestampValue1, false);

  const std::vector<Packet>& output = runner()->Outputs().Get("", 0).packets;
  ASSERT_EQ(2, output.size());
  EXPECT_EQ(kTimestampValue0, output[0].Timestamp().Value());
  EXPECT_EQ(kTimestampValue1, output[1].Timestamp().Value());
  EXPECT_EQ(true, output[0].Get<bool>());
  EXPECT_EQ(false, output[1].Get<bool>());
}

TEST_F(GateCalculatorTest, DisallowByALLOWOptionSetToFalse) {
  SetRunner(R"(
        calculator: "GateCalculator"
        input_stream: "test_input"
        output_stream: "test_output"
        options: {
          [mediapipe.GateCalculatorOptions.ext] {
            allow: false
          }
        }
  )");

  constexpr int64 kTimestampValue0 = 42;
  RunTimeStep(kTimestampValue0, true);
  constexpr int64 kTimestampValue1 = 43;
  RunTimeStep(kTimestampValue1, false);

  const std::vector<Packet>& output = runner()->Outputs().Get("", 0).packets;
  ASSERT_EQ(0, output.size());
}

TEST_F(GateCalculatorTest, DisallowByALLOWOptionNotSet) {
  SetRunner(R"(
        calculator: "GateCalculator"
        input_stream: "test_input"
        output_stream: "test_output"
  )");

  constexpr int64 kTimestampValue0 = 42;
  RunTimeStep(kTimestampValue0, true);
  constexpr int64 kTimestampValue1 = 43;
  RunTimeStep(kTimestampValue1, false);

  const std::vector<Packet>& output = runner()->Outputs().Get("", 0).packets;
  ASSERT_EQ(0, output.size());
}

TEST_F(GateCalculatorTest, AllowByALLOWSidePacketSetToTrue) {
  SetRunner(R"(
        calculator: "GateCalculator"
        input_side_packet: "ALLOW:gating_stream"
        input_stream: "test_input"
        output_stream: "test_output"
  )");
  runner()->MutableSidePackets()->Tag(kAllowTag) = Adopt(new bool(true));

  constexpr int64 kTimestampValue0 = 42;
  RunTimeStep(kTimestampValue0, true);
  constexpr int64 kTimestampValue1 = 43;
  RunTimeStep(kTimestampValue1, false);

  const std::vector<Packet>& output = runner()->Outputs().Get("", 0).packets;
  ASSERT_EQ(2, output.size());
  EXPECT_EQ(kTimestampValue0, output[0].Timestamp().Value());
  EXPECT_EQ(kTimestampValue1, output[1].Timestamp().Value());
  EXPECT_EQ(true, output[0].Get<bool>());
  EXPECT_EQ(false, output[1].Get<bool>());
}

TEST_F(GateCalculatorTest, AllowByDisallowSidePacketSetToFalse) {
  SetRunner(R"(
        calculator: "GateCalculator"
        input_side_packet: "DISALLOW:gating_stream"
        input_stream: "test_input"
        output_stream: "test_output"
  )");
  runner()->MutableSidePackets()->Tag(kDisallowTag) = Adopt(new bool(false));

  constexpr int64 kTimestampValue0 = 42;
  RunTimeStep(kTimestampValue0, true);
  constexpr int64 kTimestampValue1 = 43;
  RunTimeStep(kTimestampValue1, false);

  const std::vector<Packet>& output = runner()->Outputs().Get("", 0).packets;
  ASSERT_EQ(2, output.size());
  EXPECT_EQ(kTimestampValue0, output[0].Timestamp().Value());
  EXPECT_EQ(kTimestampValue1, output[1].Timestamp().Value());
  EXPECT_EQ(true, output[0].Get<bool>());
  EXPECT_EQ(false, output[1].Get<bool>());
}

TEST_F(GateCalculatorTest, DisallowByALLOWSidePacketSetToFalse) {
  SetRunner(R"(
        calculator: "GateCalculator"
        input_side_packet: "ALLOW:gating_stream"
        input_stream: "test_input"
        output_stream: "test_output"
  )");
  runner()->MutableSidePackets()->Tag(kAllowTag) = Adopt(new bool(false));

  constexpr int64 kTimestampValue0 = 42;
  RunTimeStep(kTimestampValue0, true);
  constexpr int64 kTimestampValue1 = 43;
  RunTimeStep(kTimestampValue1, false);

  const std::vector<Packet>& output = runner()->Outputs().Get("", 0).packets;
  ASSERT_EQ(0, output.size());
}

TEST_F(GateCalculatorTest, DisallowByDISALLOWSidePacketSetToTrue) {
  SetRunner(R"(
        calculator: "GateCalculator"
        input_side_packet: "DISALLOW:gating_stream"
        input_stream: "test_input"
        output_stream: "test_output"
  )");
  runner()->MutableSidePackets()->Tag(kDisallowTag) = Adopt(new bool(true));

  constexpr int64 kTimestampValue0 = 42;
  RunTimeStep(kTimestampValue0, true);
  constexpr int64 kTimestampValue1 = 43;
  RunTimeStep(kTimestampValue1, false);

  const std::vector<Packet>& output = runner()->Outputs().Get("", 0).packets;
  ASSERT_EQ(0, output.size());
}

TEST_F(GateCalculatorTest, Allow) {
  SetRunner(R"(
        calculator: "GateCalculator"
        input_stream: "test_input"
        input_stream: "ALLOW:gating_stream"
        output_stream: "test_output"
  )");

  constexpr int64 kTimestampValue0 = 42;
  RunTimeStep(kTimestampValue0, "ALLOW", true);
  constexpr int64 kTimestampValue1 = 43;
  RunTimeStep(kTimestampValue1, "ALLOW", false);
  constexpr int64 kTimestampValue2 = 44;
  RunTimeStep(kTimestampValue2, "ALLOW", true);
  constexpr int64 kTimestampValue3 = 45;
  RunTimeStep(kTimestampValue3, "ALLOW", false);

  const std::vector<Packet>& output = runner()->Outputs().Get("", 0).packets;
  ASSERT_EQ(2, output.size());
  EXPECT_EQ(kTimestampValue0, output[0].Timestamp().Value());
  EXPECT_EQ(kTimestampValue2, output[1].Timestamp().Value());
  EXPECT_EQ(true, output[0].Get<bool>());
  EXPECT_EQ(true, output[1].Get<bool>());
}

TEST_F(GateCalculatorTest, Disallow) {
  SetRunner(R"(
        calculator: "GateCalculator"
        input_stream: "test_input"
        input_stream: "DISALLOW:gating_stream"
        output_stream: "test_output"
  )");

  constexpr int64 kTimestampValue0 = 42;
  RunTimeStep(kTimestampValue0, "DISALLOW", true);
  constexpr int64 kTimestampValue1 = 43;
  RunTimeStep(kTimestampValue1, "DISALLOW", false);
  constexpr int64 kTimestampValue2 = 44;
  RunTimeStep(kTimestampValue2, "DISALLOW", true);
  constexpr int64 kTimestampValue3 = 45;
  RunTimeStep(kTimestampValue3, "DISALLOW", false);

  const std::vector<Packet>& output = runner()->Outputs().Get("", 0).packets;
  ASSERT_EQ(2, output.size());
  EXPECT_EQ(kTimestampValue1, output[0].Timestamp().Value());
  EXPECT_EQ(kTimestampValue3, output[1].Timestamp().Value());
  EXPECT_EQ(true, output[0].Get<bool>());
  EXPECT_EQ(true, output[1].Get<bool>());
}

TEST_F(GateCalculatorTest, AllowWithStateChange) {
  SetRunner(R"(
        calculator: "GateCalculator"
        input_stream: "test_input"
        input_stream: "ALLOW:gating_stream"
        output_stream: "test_output"
        output_stream: "STATE_CHANGE:state_changed"
  )");

  constexpr int64 kTimestampValue0 = 42;
  RunTimeStep(kTimestampValue0, "ALLOW", false);
  constexpr int64 kTimestampValue1 = 43;
  RunTimeStep(kTimestampValue1, "ALLOW", true);
  constexpr int64 kTimestampValue2 = 44;
  RunTimeStep(kTimestampValue2, "ALLOW", true);
  constexpr int64 kTimestampValue3 = 45;
  RunTimeStep(kTimestampValue3, "ALLOW", false);

  const std::vector<Packet>& output =
      runner()->Outputs().Get("STATE_CHANGE", 0).packets;
  ASSERT_EQ(2, output.size());
  EXPECT_EQ(kTimestampValue1, output[0].Timestamp().Value());
  EXPECT_EQ(kTimestampValue3, output[1].Timestamp().Value());
  EXPECT_EQ(true, output[0].Get<bool>());   // Allow.
  EXPECT_EQ(false, output[1].Get<bool>());  // Disallow.
}

TEST_F(GateCalculatorTest, DisallowWithStateChange) {
  SetRunner(R"(
        calculator: "GateCalculator"
        input_stream: "test_input"
        input_stream: "DISALLOW:gating_stream"
        output_stream: "test_output"
        output_stream: "STATE_CHANGE:state_changed"
  )");

  constexpr int64 kTimestampValue0 = 42;
  RunTimeStep(kTimestampValue0, "DISALLOW", true);
  constexpr int64 kTimestampValue1 = 43;
  RunTimeStep(kTimestampValue1, "DISALLOW", false);
  constexpr int64 kTimestampValue2 = 44;
  RunTimeStep(kTimestampValue2, "DISALLOW", false);
  constexpr int64 kTimestampValue3 = 45;
  RunTimeStep(kTimestampValue3, "DISALLOW", true);

  const std::vector<Packet>& output =
      runner()->Outputs().Get("STATE_CHANGE", 0).packets;
  ASSERT_EQ(2, output.size());
  EXPECT_EQ(kTimestampValue1, output[0].Timestamp().Value());
  EXPECT_EQ(kTimestampValue3, output[1].Timestamp().Value());
  EXPECT_EQ(true, output[0].Get<bool>());   // Allow.
  EXPECT_EQ(false, output[1].Get<bool>());  // Disallow.
}

// Must not detect disallow value for first timestamp as a state change.
TEST_F(GateCalculatorTest, DisallowInitialNoStateTransition) {
  SetRunner(R"(
        calculator: "GateCalculator"
        input_stream: "test_input"
        input_stream: "DISALLOW:gating_stream"
        output_stream: "test_output"
        output_stream: "STATE_CHANGE:state_changed"
  )");

  constexpr int64 kTimestampValue0 = 42;
  RunTimeStep(kTimestampValue0, "DISALLOW", false);

  const std::vector<Packet>& output =
      runner()->Outputs().Get("STATE_CHANGE", 0).packets;
  ASSERT_EQ(0, output.size());
}

// Must not detect allow value for first timestamp as a state change.
TEST_F(GateCalculatorTest, AllowInitialNoStateTransition) {
  SetRunner(R"(
        calculator: "GateCalculator"
        input_stream: "test_input"
        input_stream: "ALLOW:gating_stream"
        output_stream: "test_output"
        output_stream: "STATE_CHANGE:state_changed"
  )");

  constexpr int64 kTimestampValue0 = 42;
  RunTimeStep(kTimestampValue0, "ALLOW", true);

  const std::vector<Packet>& output =
      runner()->Outputs().Get("STATE_CHANGE", 0).packets;
  ASSERT_EQ(0, output.size());
}

}  // namespace
}  // namespace mediapipe
