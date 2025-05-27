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

#include "mediapipe/framework/api3/calculator_context.h"

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api3/port_test_nodes.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::api3 {
namespace {

class FooApi1Base : public CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc) {
    cc->Inputs().Tag("INPUT").Set<int>();
    cc->Outputs().Tag("OUTPUT").Set<int>();
    cc->InputSidePackets().Tag("SIDE_INPUT").Set<std::string>();
    cc->OutputSidePackets().Tag("SIDE_OUTPUT").Set<std::string>();
    return absl::OkStatus();
  }
};

class FooCheckPortsAccessCalculator : public FooApi1Base {
 public:
  using FooApi1Base::GetContract;

  absl::Status Process(mediapipe::CalculatorContext* cc) final {
    CalculatorContext<FooNode> foo(*cc);

    RET_CHECK(foo.input);
    RET_CHECK_EQ(foo.input.GetOrDie(), 21);
    RET_CHECK_EQ(foo.input.Packet().GetOrDie(), 21);

    RET_CHECK(foo.side_input);
    RET_CHECK_EQ(foo.side_input.GetOrDie(), "foo_input_side");
    RET_CHECK_EQ(foo.side_input.Packet().GetOrDie(), "foo_input_side");

    foo.output.Send(42);
    foo.side_output.Set("foo_output_side");

    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(FooCheckPortsAccessCalculator);

TEST(PortContextTest, CanReadInputWriteOutputPorts) {
  // Setup.
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "in"
        output_stream: "out"
        input_side_packet: "side_in"
        output_side_packet: "side_out"
        node {
          calculator: "FooCheckPortsAccessCalculator"
          input_stream: "INPUT:in"
          input_side_packet: "SIDE_INPUT:side_in"
          output_stream: "OUTPUT:out"
          output_side_packet: "SIDE_OUTPUT:side_out"
        }
      )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(std::move(config)));
  mediapipe::Packet out;
  MP_ASSERT_OK(
      graph.ObserveOutputStream("out", [&out](const mediapipe::Packet& p) {
        out = p;
        return absl::OkStatus();
      }));

  // Starting and sending inputs.
  MP_ASSERT_OK(graph.StartRun(
      {{"side_in", mediapipe::MakePacket<std::string>("foo_input_side")}}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "in", mediapipe::MakePacket<int>(21).At(Timestamp(0))));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Verifying outputs.
  MP_ASSERT_OK_AND_ASSIGN(mediapipe::Packet side_out,
                          graph.GetOutputSidePacket("side_out"));
  ASSERT_FALSE(side_out.IsEmpty());
  EXPECT_EQ(side_out.Get<std::string>(), "foo_output_side");

  ASSERT_FALSE(out.IsEmpty());
  EXPECT_EQ(out.Get<int>(), 42);

  // Cleanuper.
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

class RepeatedFooContractBase : public CalculatorBase {
 public:
  static absl::Status GetContract(mediapipe::CalculatorContract* cc) {
    for (int i = 0; i < cc->Inputs().NumEntries("INPUT"); ++i) {
      cc->Inputs().Get("INPUT", i).Set<int>();
    }
    for (int i = 0; i < cc->Outputs().NumEntries("OUTPUT"); ++i) {
      cc->Outputs().Get("OUTPUT", i).Set<int>();
    }
    for (int i = 0; i < cc->InputSidePackets().NumEntries("SIDE_INPUT"); ++i) {
      cc->InputSidePackets().Get("SIDE_INPUT", i).Set<std::string>();
    }
    for (int i = 0; i < cc->OutputSidePackets().NumEntries("SIDE_OUTPUT");
         ++i) {
      cc->OutputSidePackets().Get("SIDE_OUTPUT", i).Set<std::string>();
    }
    return absl::OkStatus();
  }
};

class RepeatedFooCheckPortsAccessCalculator : public RepeatedFooContractBase {
 public:
  using RepeatedFooContractBase::GetContract;

  absl::Status Process(mediapipe::CalculatorContext* cc) final {
    CalculatorContext<RepeatedFooNode> repeated_foo(*cc);

    RET_CHECK_EQ(repeated_foo.input.Count(), 1);
    RET_CHECK(repeated_foo.input[0]);
    RET_CHECK_EQ(repeated_foo.input[0].GetOrDie(), 21);
    RET_CHECK_EQ(repeated_foo.input[0].Packet().GetOrDie(), 21);

    RET_CHECK_EQ(repeated_foo.side_input.Count(), 1);
    RET_CHECK(repeated_foo.side_input[0]);
    RET_CHECK_EQ(repeated_foo.side_input[0].GetOrDie(), "foo_input_side");
    RET_CHECK_EQ(repeated_foo.side_input[0].Packet().GetOrDie(),
                 "foo_input_side");

    RET_CHECK_EQ(repeated_foo.output.Count(), 1);
    repeated_foo.output[0].Send(42);

    RET_CHECK_EQ(repeated_foo.side_output.Count(), 1);
    repeated_foo.side_output[0].Set("foo_output_side");

    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(RepeatedFooCheckPortsAccessCalculator);

TEST(PortContextTest, CanReadInputWriteOutputRepeatedPortsWhenOneSpecified) {
  // Setup.
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "in"
        output_stream: "out"
        input_side_packet: "side_in"
        output_side_packet: "side_out"
        node {
          calculator: "RepeatedFooCheckPortsAccessCalculator"
          input_stream: "INPUT:in"
          input_side_packet: "SIDE_INPUT:side_in"
          output_stream: "OUTPUT:out"
          output_side_packet: "SIDE_OUTPUT:side_out"
        }
      )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(std::move(config)));
  mediapipe::Packet out;
  MP_ASSERT_OK(
      graph.ObserveOutputStream("out", [&out](const mediapipe::Packet& p) {
        out = p;
        return absl::OkStatus();
      }));

  // Starting and sending inputs.
  MP_ASSERT_OK(graph.StartRun(
      {{"side_in", mediapipe::MakePacket<std::string>("foo_input_side")}}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "in", mediapipe::MakePacket<int>(21).At(Timestamp(0))));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Verifying outputs.
  MP_ASSERT_OK_AND_ASSIGN(mediapipe::Packet side_out,
                          graph.GetOutputSidePacket("side_out"));
  ASSERT_FALSE(side_out.IsEmpty());
  EXPECT_EQ(side_out.Get<std::string>(), "foo_output_side");

  ASSERT_FALSE(out.IsEmpty());
  EXPECT_EQ(out.Get<int>(), 42);

  // Cleanup.
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

class RepeatedFooCheckManyPortsAccessCalculator
    : public RepeatedFooContractBase {
 public:
  using RepeatedFooContractBase::GetContract;

  absl::Status Process(mediapipe::CalculatorContext* cc) final {
    CalculatorContext<RepeatedFooNode> repeated_foo(*cc);

    RET_CHECK_EQ(repeated_foo.input.Count(), 2);
    RET_CHECK_EQ(repeated_foo.input[0].GetOrDie(), 21);
    RET_CHECK_EQ(repeated_foo.input[0].Packet().GetOrDie(), 21);
    RET_CHECK_EQ(repeated_foo.input[1].GetOrDie(), 22);
    RET_CHECK_EQ(repeated_foo.input[1].Packet().GetOrDie(), 22);

    RET_CHECK_EQ(repeated_foo.side_input.Count(), 3);
    RET_CHECK(repeated_foo.side_input[0]);
    RET_CHECK_EQ(repeated_foo.side_input[0].GetOrDie(), "foo_input_side0");
    RET_CHECK_EQ(repeated_foo.side_input[0].Packet().GetOrDie(),
                 "foo_input_side0");
    RET_CHECK(repeated_foo.side_input[1]);
    RET_CHECK_EQ(repeated_foo.side_input[1].GetOrDie(), "foo_input_side1");
    RET_CHECK_EQ(repeated_foo.side_input[1].Packet().GetOrDie(),
                 "foo_input_side1");
    RET_CHECK(repeated_foo.side_input[2]);
    RET_CHECK_EQ(repeated_foo.side_input[2].GetOrDie(), "foo_input_side2");
    RET_CHECK_EQ(repeated_foo.side_input[2].Packet().GetOrDie(),
                 "foo_input_side2");

    RET_CHECK_EQ(repeated_foo.output.Count(), 3);
    repeated_foo.output[0].Send(42);
    repeated_foo.output[1].Send(43);
    repeated_foo.output[2].Send(44);

    RET_CHECK_EQ(repeated_foo.side_output.Count(), 4);
    repeated_foo.side_output[0].Set("foo_output_side0");
    repeated_foo.side_output[1].Set("foo_output_side1");
    repeated_foo.side_output[2].Set("foo_output_side2");
    repeated_foo.side_output[3].Set("foo_output_side3");

    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(RepeatedFooCheckManyPortsAccessCalculator);

TEST(PortContextTest, CanReadInputWriteOutputRepeatedPortsWhenManySpecified) {
  // Setup.
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "in0"
        input_stream: "in1"
        input_side_packet: "side_in0"
        input_side_packet: "side_in1"
        input_side_packet: "side_in3"
        node {
          calculator: "RepeatedFooCheckManyPortsAccessCalculator"
          input_stream: "INPUT:0:in0"
          input_stream: "INPUT:1:in1"
          input_side_packet: "SIDE_INPUT:0:side_in0"
          input_side_packet: "SIDE_INPUT:1:side_in1"
          input_side_packet: "SIDE_INPUT:2:side_in2"
          output_stream: "OUTPUT:0:out0"
          output_stream: "OUTPUT:1:out1"
          output_stream: "OUTPUT:2:out2"
          output_side_packet: "SIDE_OUTPUT:0:side_out0"
          output_side_packet: "SIDE_OUTPUT:1:side_out1"
          output_side_packet: "SIDE_OUTPUT:2:side_out2"
          output_side_packet: "SIDE_OUTPUT:3:side_out3"
        }
      )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(std::move(config)));
  mediapipe::Packet out0;
  MP_ASSERT_OK(
      graph.ObserveOutputStream("out0", [&out0](const mediapipe::Packet& p) {
        out0 = p;
        return absl::OkStatus();
      }));
  mediapipe::Packet out1;
  MP_ASSERT_OK(
      graph.ObserveOutputStream("out1", [&out1](const mediapipe::Packet& p) {
        out1 = p;
        return absl::OkStatus();
      }));
  mediapipe::Packet out2;
  MP_ASSERT_OK(
      graph.ObserveOutputStream("out2", [&out2](const mediapipe::Packet& p) {
        out2 = p;
        return absl::OkStatus();
      }));

  // Staring and sending inputs.
  MP_ASSERT_OK(graph.StartRun(
      {{"side_in0", mediapipe::MakePacket<std::string>("foo_input_side0")},
       {"side_in1", mediapipe::MakePacket<std::string>("foo_input_side1")},
       {"side_in2", mediapipe::MakePacket<std::string>("foo_input_side2")}}));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "in0", mediapipe::MakePacket<int>(21).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "in1", mediapipe::MakePacket<int>(22).At(Timestamp(0))));

  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Verifying outputs.
  mediapipe::Packet side_out;
  MP_ASSERT_OK_AND_ASSIGN(side_out, graph.GetOutputSidePacket("side_out0"));
  ASSERT_FALSE(side_out.IsEmpty());
  EXPECT_EQ(side_out.Get<std::string>(), "foo_output_side0");
  MP_ASSERT_OK_AND_ASSIGN(side_out, graph.GetOutputSidePacket("side_out1"));
  ASSERT_FALSE(side_out.IsEmpty());
  EXPECT_EQ(side_out.Get<std::string>(), "foo_output_side1");
  MP_ASSERT_OK_AND_ASSIGN(side_out, graph.GetOutputSidePacket("side_out2"));
  ASSERT_FALSE(side_out.IsEmpty());
  EXPECT_EQ(side_out.Get<std::string>(), "foo_output_side2");

  ASSERT_FALSE(out0.IsEmpty());
  EXPECT_EQ(out0.Get<int>(), 42);
  ASSERT_FALSE(out1.IsEmpty());
  EXPECT_EQ(out1.Get<int>(), 43);
  ASSERT_FALSE(out2.IsEmpty());
  EXPECT_EQ(out2.Get<int>(), 44);

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

}  // namespace
}  // namespace mediapipe::api3
