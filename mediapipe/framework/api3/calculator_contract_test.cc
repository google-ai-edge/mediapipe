// Copyright 2025 The MediaPipe Authors.
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

#include "mediapipe/framework/api3/calculator_contract.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/internal/specializers.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/api3/port_test_nodes.h"
#include "mediapipe/framework/calculator_contract.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::api3 {
namespace {

TEST(PortContractTest, PortsCanUpdateContract) {
  mediapipe::CalculatorContract contract;
  MP_ASSERT_OK(
      contract.Initialize(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        input_stream: "INPUT:in"
        input_side_packet: "SIDE_INPUT:side_in"
        output_stream: "OUTPUT:out"
        output_side_packet: "SIDE_OUTPUT:side_out"
      )pb")));

  ASSERT_TRUE(contract.Inputs().HasTag("INPUT"));
  ASSERT_TRUE(contract.InputSidePackets().HasTag("SIDE_INPUT"));
  ASSERT_TRUE(contract.Outputs().HasTag("OUTPUT"));
  ASSERT_TRUE(contract.OutputSidePackets().HasTag("SIDE_OUTPUT"));

  mediapipe::PacketType input_output_expected_type;
  input_output_expected_type.Set<int>();
  mediapipe::PacketType side_input_output_expected_type;
  side_input_output_expected_type.Set<std::string>();

  // Verify that ports lack type information.
  EXPECT_FALSE(contract.Inputs().Tag("INPUT").IsConsistentWith(
      input_output_expected_type));
  EXPECT_FALSE(contract.InputSidePackets()
                   .Tag("SIDE_INPUT")
                   .IsConsistentWith(side_input_output_expected_type));
  EXPECT_FALSE(contract.Outputs().Tag("OUTPUT").IsConsistentWith(
      input_output_expected_type));
  EXPECT_FALSE(contract.OutputSidePackets()
                   .Tag("SIDE_OUTPUT")
                   .IsConsistentWith(side_input_output_expected_type));

  // Constructs calculator contract and adds type information.
  CalculatorContract<FooNode> foo(
      contract, [](absl::Status status) { ASSERT_TRUE(status.ok()); });

  EXPECT_TRUE(contract.Inputs().Tag("INPUT").IsConsistentWith(
      input_output_expected_type));
  EXPECT_TRUE(contract.InputSidePackets()
                  .Tag("SIDE_INPUT")
                  .IsConsistentWith(side_input_output_expected_type));
  EXPECT_TRUE(contract.Outputs().Tag("OUTPUT").IsConsistentWith(
      input_output_expected_type));
  EXPECT_TRUE(contract.OutputSidePackets()
                  .Tag("SIDE_OUTPUT")
                  .IsConsistentWith(side_input_output_expected_type));
}

TEST(PortContractTest, RepeatedPortsCanUpdateContract) {
  mediapipe::CalculatorContract contract;
  MP_ASSERT_OK(
      contract.Initialize(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        input_stream: "INPUT:in"
        input_side_packet: "SIDE_INPUT:0:side_in0"
        input_side_packet: "SIDE_INPUT:1:side_in1"
        output_stream: "OUTPUT:0:out0"
        output_stream: "OUTPUT:1:out1"
        output_stream: "OUTPUT:2:out2"
        output_stream: "OUTPUT:3:out3"
        output_side_packet: "SIDE_OUTPUT:0:side_out0"
        output_side_packet: "SIDE_OUTPUT:1:side_out1"
      )pb")));

  ASSERT_TRUE(contract.Inputs().HasTag("INPUT"));
  ASSERT_TRUE(contract.InputSidePackets().HasTag("SIDE_INPUT"));
  ASSERT_TRUE(contract.Outputs().HasTag("OUTPUT"));
  ASSERT_TRUE(contract.OutputSidePackets().HasTag("SIDE_OUTPUT"));

  mediapipe::PacketType input_output_expected_type;
  input_output_expected_type.Set<int>();
  mediapipe::PacketType side_input_output_expected_type;
  side_input_output_expected_type.Set<std::string>();

  // Verify that ports lack type information.
  EXPECT_FALSE(contract.Inputs()
                   .Get("INPUT", 0)
                   .IsConsistentWith(input_output_expected_type));
  EXPECT_FALSE(contract.InputSidePackets()
                   .Get("SIDE_INPUT", 1)
                   .IsConsistentWith(side_input_output_expected_type));
  EXPECT_FALSE(contract.Outputs()
                   .Get("OUTPUT", 3)
                   .IsConsistentWith(input_output_expected_type));
  EXPECT_FALSE(contract.OutputSidePackets()
                   .Get("SIDE_OUTPUT", 1)
                   .IsConsistentWith(side_input_output_expected_type));

  // Constructs calculator contract and adds type information.
  CalculatorContract<RepeatedFooNode> foo(
      contract, [](absl::Status status) { ASSERT_TRUE(status.ok()); });

  EXPECT_TRUE(contract.Inputs()
                  .Get("INPUT", 0)
                  .IsConsistentWith(input_output_expected_type));
  EXPECT_TRUE(contract.InputSidePackets()
                  .Get("SIDE_INPUT", 1)
                  .IsConsistentWith(side_input_output_expected_type));
  EXPECT_TRUE(contract.Outputs()
                  .Get("OUTPUT", 3)
                  .IsConsistentWith(input_output_expected_type));
  EXPECT_TRUE(contract.OutputSidePackets()
                  .Get("SIDE_OUTPUT", 1)
                  .IsConsistentWith(side_input_output_expected_type));
}

TEST(PortContractTest, CanCheckRepeatedPortEntriesCount) {
  mediapipe::CalculatorContract contract;
  MP_ASSERT_OK(
      contract.Initialize(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        input_stream: "INPUT:in"
        input_side_packet: "SIDE_INPUT:0:side_in0"
        input_side_packet: "SIDE_INPUT:1:side_in1"
        output_stream: "OUTPUT:0:out0"
        output_stream: "OUTPUT:1:out1"
        output_stream: "OUTPUT:2:out2"
        output_stream: "OUTPUT:3:out3"
        output_side_packet: "SIDE_OUTPUT:0:side_out0"
        output_side_packet: "SIDE_OUTPUT:1:side_out1"
      )pb")));

  CalculatorContract<RepeatedFooNode> foo(
      contract, [](absl::Status status) { ASSERT_TRUE(status.ok()); });

  EXPECT_EQ(foo.input.Count(), 1);
  EXPECT_EQ(foo.output.Count(), 4);
  EXPECT_EQ(foo.side_input.Count(), 2);
  EXPECT_EQ(foo.side_output.Count(), 2);
}

TEST(PortContractTest, CanAccessRepeatedPortEntriesAt) {
  mediapipe::CalculatorContract contract;
  MP_ASSERT_OK(
      contract.Initialize(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        input_stream: "INPUT:in"
        input_side_packet: "SIDE_INPUT:0:side_in0"
        input_side_packet: "SIDE_INPUT:1:side_in1"
        output_stream: "OUTPUT:0:out0"
        output_stream: "OUTPUT:1:out1"
        output_stream: "OUTPUT:2:out2"
        output_stream: "OUTPUT:3:out3"
        output_side_packet: "SIDE_OUTPUT:0:side_out0"
        output_side_packet: "SIDE_OUTPUT:1:side_out1"
      )pb")));

  CalculatorContract<RepeatedFooNode> foo(
      contract, [](absl::Status status) { ASSERT_TRUE(status.ok()); });

  EXPECT_EQ(foo.input.At(0).Index(), 0);
  EXPECT_EQ(foo.output.At(0).Index(), 0);
  EXPECT_EQ(foo.output.At(1).Index(), 1);
  EXPECT_EQ(foo.output.At(2).Index(), 2);
  EXPECT_EQ(foo.output.At(3).Index(), 3);
  EXPECT_EQ(foo.side_input.At(0).Index(), 0);
  EXPECT_EQ(foo.side_input.At(1).Index(), 1);
  EXPECT_EQ(foo.side_output.At(0).Index(), 0);
  EXPECT_EQ(foo.side_output.At(1).Index(), 1);
}

TEST(PortContractTest, CanAccessRepeatedPortEntriesSubscript) {
  mediapipe::CalculatorContract contract;
  MP_ASSERT_OK(
      contract.Initialize(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        input_stream: "INPUT:in"
        input_side_packet: "SIDE_INPUT:0:side_in0"
        input_side_packet: "SIDE_INPUT:1:side_in1"
        output_stream: "OUTPUT:0:out0"
        output_stream: "OUTPUT:1:out1"
        output_stream: "OUTPUT:2:out2"
        output_stream: "OUTPUT:3:out3"
        output_side_packet: "SIDE_OUTPUT:0:side_out0"
        output_side_packet: "SIDE_OUTPUT:1:side_out1"
      )pb")));

  CalculatorContract<RepeatedFooNode> foo(
      contract, [](absl::Status status) { ASSERT_TRUE(status.ok()); });

  EXPECT_EQ(foo.input.At(0).Index(), 0);
  EXPECT_EQ(foo.output.At(0).Index(), 0);
  EXPECT_EQ(foo.output.At(1).Index(), 1);
  EXPECT_EQ(foo.output.At(2).Index(), 2);
  EXPECT_EQ(foo.output.At(3).Index(), 3);
  EXPECT_EQ(foo.side_input.At(0).Index(), 0);
  EXPECT_EQ(foo.side_input.At(1).Index(), 1);
  EXPECT_EQ(foo.side_output.At(0).Index(), 0);
  EXPECT_EQ(foo.side_output.At(1).Index(), 1);
}

TEST(PortContractTest, CanAccessRepeatedPortEntriesIterator) {
  mediapipe::CalculatorContract contract;
  MP_ASSERT_OK(
      contract.Initialize(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        input_stream: "INPUT:in"
        input_side_packet: "SIDE_INPUT:0:side_in0"
        input_side_packet: "SIDE_INPUT:1:side_in1"
        output_stream: "OUTPUT:0:out0"
        output_stream: "OUTPUT:1:out1"
        output_stream: "OUTPUT:2:out2"
        output_stream: "OUTPUT:3:out3"
        output_side_packet: "SIDE_OUTPUT:0:side_out0"
        output_side_packet: "SIDE_OUTPUT:1:side_out1"
      )pb")));

  CalculatorContract<RepeatedFooNode> foo(
      contract, [](absl::Status status) { ASSERT_TRUE(status.ok()); });

  int count = 0;
  for (auto it = foo.input.begin(); it != foo.input.end(); ++it) {
    EXPECT_EQ((*it).Index(), count++);
  }
  EXPECT_EQ(count, 1);

  count = 0;
  for (auto it = foo.output.begin(); it != foo.output.end(); ++it) {
    EXPECT_EQ((*it).Index(), count++);
  }
  EXPECT_EQ(count, 4);

  count = 0;
  for (auto it = foo.side_input.begin(); it != foo.side_input.end(); ++it) {
    EXPECT_EQ((*it).Index(), count++);
  }
  EXPECT_EQ(count, 2);

  count = 0;
  for (auto it = foo.side_output.begin(); it != foo.side_output.end(); ++it) {
    EXPECT_EQ((*it).Index(), count++);
  }
  EXPECT_EQ(count, 2);
}

template <typename S>
struct OptionalFoo {
  Optional<Input<S, float>> input{"INPUT"};
  Optional<Output<S, float>> output{"OUTPUT"};
  Optional<SideInput<S, float>> side_input{"SIDE_INPUT"};
  Optional<SideOutput<S, float>> side_output{"SIDE_OUTPUT"};
};

/*inline*/ constexpr absl::string_view kOptionalFooName = "OptionalFoo";
struct OptionalFooNode : Node<kOptionalFooName> {
  template <typename S>
  using Contract = OptionalFoo<S>;
};

TEST(PortContractTest, CanGetOptionalPortTags) {
  OptionalFoo<ContractSpecializer> bar;
  EXPECT_EQ(bar.input.Tag(), "INPUT");
  EXPECT_EQ(bar.output.Tag(), "OUTPUT");
  EXPECT_EQ(bar.side_input.Tag(), "SIDE_INPUT");
  EXPECT_EQ(bar.side_output.Tag(), "SIDE_OUTPUT");
}

TEST(PortContractTest, CanCheckOptionalPortIsConnected) {
  mediapipe::CalculatorContract contract;
  MP_ASSERT_OK(
      contract.Initialize(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        input_stream: "INPUT:in"
        input_side_packet: "SIDE_INPUT:side_in"
        output_stream: "OUTPUT:out"
        output_side_packet: "SIDE_OUTPUT:side_out"
      )pb")));

  CalculatorContract<OptionalFooNode> foo(
      contract, [](absl::Status status) { ASSERT_TRUE(status.ok()); });

  EXPECT_TRUE(foo.input.IsConnected());
  EXPECT_TRUE(foo.side_input.IsConnected());
  EXPECT_TRUE(foo.output.IsConnected());
  EXPECT_TRUE(foo.side_output.IsConnected());
}

TEST(PortContractTest, CanCheckOptionalPortIsNotConnected) {
  mediapipe::CalculatorContract contract;
  MP_ASSERT_OK(
      contract.Initialize(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        input_stream: "INPUT_OTHER:in"
        input_side_packet: "SIDE_INPUT_OTHER:side_in"
        output_stream: "OUTPUT_OTHER:out"
        output_side_packet: "SIDE_OUTPUT_OTHER:side_out"
      )pb")));

  CalculatorContract<OptionalFooNode> foo(
      contract, [](absl::Status status) { ASSERT_TRUE(status.ok()); });

  EXPECT_FALSE(foo.input.IsConnected());
  EXPECT_FALSE(foo.output.IsConnected());
  EXPECT_FALSE(foo.side_input.IsConnected());
  EXPECT_FALSE(foo.side_output.IsConnected());
}

}  // namespace
}  // namespace mediapipe::api3
