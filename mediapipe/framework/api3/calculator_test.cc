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

#include "mediapipe/framework/api3/calculator.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/api3/calculator_contract.h"
#include "mediapipe/framework/api3/calculator_test.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/graph_service.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/tool/options_map.h"

namespace mediapipe::api3 {

inline constexpr GraphService<int> kTestIntService(
    "kTestIntService", GraphServiceBase::kDisallowDefaultInitialization);

absl::Status PassThroughNodeImpl::UpdateContract(
    CalculatorContract<PassThroughNode>& cc) {
  cc.SetProcessTimestampBounds(true);
  RET_CHECK_EQ(cc.foo_options.Get().a(), 1);
  RET_CHECK_EQ(cc.foo_options.Get().b(), "1");
  RET_CHECK_EQ(cc.bar_options.Get().a(), 2);
  RET_CHECK_EQ(cc.bar_options.Get().b(), "2");
  return absl::OkStatus();
}

absl::Status PassThroughNodeImpl::Open(CalculatorContext<PassThroughNode>& cc) {
  RET_CHECK(cc.Service(kTestStringService).IsAvailable());
  RET_CHECK_EQ(cc.Service(kTestStringService).GetObject(), "test_service");
  RET_CHECK_EQ(cc.foo_options.Get().a(), 1);
  RET_CHECK_EQ(cc.foo_options.Get().b(), "1");
  RET_CHECK_EQ(cc.bar_options.Get().a(), 2);
  RET_CHECK_EQ(cc.bar_options.Get().b(), "2");
  cc.side_out.Set(cc.side_in.GetOrDie());
  return absl::OkStatus();
}

absl::Status PassThroughNodeImpl::Process(
    CalculatorContext<PassThroughNode>& cc) {
  RET_CHECK(cc.Service(kTestStringService).IsAvailable());
  RET_CHECK_EQ(cc.Service(kTestStringService).GetObject(), "test_service");
  cc.out.Send(cc.in.GetOrDie());
  return absl::OkStatus();
}

absl::Status PassThroughNodeImpl::Close(
    CalculatorContext<PassThroughNode>& cc) {
  RET_CHECK(cc.Service(kTestStringService).IsAvailable());
  RET_CHECK_EQ(cc.Service(kTestStringService).GetObject(), "test_service");
  cc.out.Close();
  return absl::OkStatus();
}

TEST(CalculatorTest, CanReadWritePortsAndUseServices) {
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "in"
        output_stream: "out"
        input_side_packet: "side_in"
        output_side_packet: "side_out"
        node {
          calculator: "PassThrough"
          input_stream: "IN:in"
          input_side_packet: "SIDE_IN:side_in"
          output_stream: "OUT:out"
          output_side_packet: "SIDE_OUT:side_out"
          node_options: {
            [type.googleapis.com/mediapipe.FooOptions] { a: 1 b: "1" }
          }
          node_options: {
            [type.googleapis.com/mediapipe.BarOptions] { a: 2 b: "2" }
          }
        }
      )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(std::move(config)));
  MP_ASSERT_OK(graph.SetServiceObject(
      kTestStringService, std::make_shared<std::string>("test_service")));
  mediapipe::Packet out;
  MP_ASSERT_OK(
      graph.ObserveOutputStream("out", [&out](const mediapipe::Packet& p) {
        out = p;
        return absl::OkStatus();
      }));

  // Starting and sending inputs.
  MP_ASSERT_OK(graph.StartRun(
      {{"side_in", mediapipe::MakePacket<std::string>("side")}}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "in", mediapipe::MakePacket<int>(42).At(Timestamp(0))));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Verifying outputs.
  MP_ASSERT_OK_AND_ASSIGN(mediapipe::Packet side_out,
                          graph.GetOutputSidePacket("side_out"));
  ASSERT_FALSE(side_out.IsEmpty());
  EXPECT_EQ(side_out.Get<std::string>(), "side");

  ASSERT_FALSE(out.IsEmpty());
  EXPECT_EQ(out.Get<int>(), 42);

  // Cleanup.
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST(CalculatorTest, FailsForIncorrectNodeConfiguration) {
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "in"
        output_stream: "out"
        input_side_packet: "side_in"
        output_side_packet: "side_out"
        node {
          calculator: "PassThrough"
          input_stream: "IN:in"
          input_side_packet: "SIDE_IN:side_in"
          output_side_packet: "SIDE_OUT:side_out"
        }
      )pb");

  CalculatorGraph graph;
  EXPECT_FALSE(graph.Initialize(std::move(config)).ok());
}

class SharedPassThroughNodeAImpl
    : public Calculator<SharedPassThroughANode, SharedPassThroughNodeAImpl> {
 public:
  static absl::Status UpdateContract(
      CalculatorContract<SharedPassThroughANode>& cc) {
    cc.UseService(kTestIntService);
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext<SharedPassThroughANode>& cc) final {
    RET_CHECK(cc.Service(kTestStringService).IsAvailable());
    RET_CHECK_EQ(cc.Service(kTestStringService).GetObject(), "test_service");
    RET_CHECK(cc.Service(kTestIntService).IsAvailable());
    RET_CHECK_EQ(cc.Service(kTestIntService).GetObject(), 42);
    cc.side_out.Set(cc.side_in.GetOrDie());
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext<SharedPassThroughANode>& cc) final {
    RET_CHECK(cc.Service(kTestStringService).IsAvailable());
    RET_CHECK_EQ(cc.Service(kTestStringService).GetObject(), "test_service");
    RET_CHECK(cc.Service(kTestIntService).IsAvailable());
    RET_CHECK_EQ(cc.Service(kTestIntService).GetObject(), 42);
    cc.out.Send(cc.in.GetOrDie());
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext<SharedPassThroughANode>& cc) final {
    RET_CHECK(cc.Service(kTestStringService).IsAvailable());
    RET_CHECK_EQ(cc.Service(kTestStringService).GetObject(), "test_service");
    RET_CHECK(cc.Service(kTestIntService).IsAvailable());
    RET_CHECK_EQ(cc.Service(kTestIntService).GetObject(), 42);
    cc.out.Close();
    return absl::OkStatus();
  }
};

class SharedPassThroughNodeBImpl
    : public Calculator<SharedPassThroughBNode, SharedPassThroughNodeBImpl> {
 public:
  static absl::Status UpdateContract(
      CalculatorContract<SharedPassThroughBNode>& cc) {
    cc.UseService(kTestIntService);
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext<SharedPassThroughBNode>& cc) final {
    RET_CHECK(cc.Service(kTestStringService).IsAvailable());
    RET_CHECK_EQ(cc.Service(kTestStringService).GetObject(), "test_service");
    RET_CHECK(cc.Service(kTestIntService).IsAvailable());
    RET_CHECK_EQ(cc.Service(kTestIntService).GetObject(), 42);
    cc.side_out.Set(cc.side_in.GetOrDie());
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext<SharedPassThroughBNode>& cc) final {
    RET_CHECK(cc.Service(kTestStringService).IsAvailable());
    RET_CHECK_EQ(cc.Service(kTestStringService).GetObject(), "test_service");
    RET_CHECK(cc.Service(kTestIntService).IsAvailable());
    RET_CHECK_EQ(cc.Service(kTestIntService).GetObject(), 42);
    cc.out.Send(cc.in.GetOrDie());
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext<SharedPassThroughBNode>& cc) final {
    RET_CHECK(cc.Service(kTestStringService).IsAvailable());
    RET_CHECK_EQ(cc.Service(kTestStringService).GetObject(), "test_service");
    RET_CHECK(cc.Service(kTestIntService).IsAvailable());
    RET_CHECK_EQ(cc.Service(kTestIntService).GetObject(), 42);
    cc.out.Close();
    return absl::OkStatus();
  }
};

TEST(CalculatorTest, CanUseSharedContract) {
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "in"
        output_stream: "out"
        input_side_packet: "side_in"
        output_side_packet: "side_out"
        node {
          calculator: "SharedPassThroughA"
          input_stream: "IN:in"
          input_side_packet: "SIDE_IN:side_in"
          output_stream: "OUT:out_a"
          output_side_packet: "SIDE_OUT:side_out_a"
        }
        node {
          calculator: "SharedPassThroughB"
          input_stream: "IN:in"
          input_side_packet: "SIDE_IN:side_in"
          output_stream: "OUT:out_b"
          output_side_packet: "SIDE_OUT:side_out_b"
        }
      )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(std::move(config)));
  MP_ASSERT_OK(graph.SetServiceObject(
      kTestStringService, std::make_shared<std::string>("test_service")));
  MP_ASSERT_OK(
      graph.SetServiceObject(kTestIntService, std::make_shared<int>(42)));
  mediapipe::Packet out_a;
  MP_ASSERT_OK(
      graph.ObserveOutputStream("out_a", [&out_a](const mediapipe::Packet& p) {
        out_a = p;
        return absl::OkStatus();
      }));
  mediapipe::Packet out_b;
  MP_ASSERT_OK(
      graph.ObserveOutputStream("out_b", [&out_b](const mediapipe::Packet& p) {
        out_b = p;
        return absl::OkStatus();
      }));

  // Starting and sending inputs.
  MP_ASSERT_OK(graph.StartRun(
      {{"side_in", mediapipe::MakePacket<std::string>("side")}}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "in", mediapipe::MakePacket<int>(42).At(Timestamp(0))));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Verifying outputs.
  MP_ASSERT_OK_AND_ASSIGN(mediapipe::Packet side_out_a,
                          graph.GetOutputSidePacket("side_out_a"));
  ASSERT_FALSE(side_out_a.IsEmpty());
  EXPECT_EQ(side_out_a.Get<std::string>(), "side");

  MP_ASSERT_OK_AND_ASSIGN(mediapipe::Packet side_out_b,
                          graph.GetOutputSidePacket("side_out_b"));
  ASSERT_FALSE(side_out_a.IsEmpty());
  EXPECT_EQ(side_out_a.Get<std::string>(), "side");

  ASSERT_FALSE(out_a.IsEmpty());
  EXPECT_EQ(out_a.Get<int>(), 42);

  ASSERT_FALSE(out_b.IsEmpty());
  EXPECT_EQ(out_b.Get<int>(), 42);

  // Cleanup.
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

inline constexpr absl::string_view kNoOpNodeName = "NoOpNode";
struct NoOpNode : Node<kNoOpNodeName> {
  template <typename S>
  struct Contract {
    Input<S, int> input{"IN"};
    Output<S, int> output{"OUT"};
  };
};

class NoOpNodeImpl : public Calculator<NoOpNode, NoOpNodeImpl> {
 public:
  absl::Status Process(CalculatorContext<NoOpNode>& cc) override {
    // Not outputting anything should result in timestamp bound update by
    // default.
    return absl::OkStatus();
  }
};

TEST(CalculatorTest, TimestampOffsetZeroIsTheDefault) {
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "in"
        output_stream: "out"
        node {
          calculator: "NoOpNode"
          input_stream: "IN:in"
          output_stream: "OUT:out"
        }
      )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(std::move(config)));
  std::vector<mediapipe::Packet> output_packets;
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "out",
      [&output_packets](const mediapipe::Packet& p) {
        output_packets.push_back(p);
        return absl::OkStatus();
      },
      /*observe_timestamp_bounds=*/true));
  MP_ASSERT_OK(graph.StartRun({}));

  // Send inputs.
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "in", mediapipe::MakePacket<int>(42).At(Timestamp(0))));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "in", mediapipe::MakePacket<int>(43).At(Timestamp(1))));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Verify outputs.
  ASSERT_EQ(output_packets.size(), 2);
  for (const auto& packet : output_packets) {
    EXPECT_TRUE(packet.IsEmpty());
  }

  // Cleanup.
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

inline constexpr absl::string_view kNoOpNodeUnsetOffsetName =
    "NoOpNodeUnsetOffset";
struct NoOpNodeUnsetOffset : Node<kNoOpNodeUnsetOffsetName> {
  template <typename S>
  struct Contract {
    Input<S, int> input{"IN"};
    Output<S, int> output{"OUT"};

    static absl::Status UpdateContract(
        CalculatorContract<NoOpNodeUnsetOffset>& cc) {
      cc.SetTimestampOffset(TimestampDiff::Unset());
      return absl::OkStatus();
    }
  };
};

class NoOpNodeUnsetOffsetImpl
    : public Calculator<NoOpNodeUnsetOffset, NoOpNodeUnsetOffsetImpl> {
 public:
  absl::Status Process(CalculatorContext<NoOpNodeUnsetOffset>& cc) override {
    // Not outputting anything should result in timestamp bound update by
    // default.
    return absl::OkStatus();
  }
};

TEST(CalculatorTest, DefaultTimestampOffsetCanBeUnset) {
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "in"
        output_stream: "out"
        node {
          calculator: "NoOpNodeUnsetOffset"
          input_stream: "IN:in"
          output_stream: "OUT:out"
        }
      )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(std::move(config)));
  std::vector<mediapipe::Packet> output_packets;
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "out",
      [&output_packets](const mediapipe::Packet& p) {
        output_packets.push_back(p);
        return absl::OkStatus();
      },
      /*observe_timestamp_bounds=*/true));
  MP_ASSERT_OK(graph.StartRun({}));

  // Send inputs.
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "in", mediapipe::MakePacket<int>(42).At(Timestamp(0))));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "in", mediapipe::MakePacket<int>(43).At(Timestamp(1))));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Verify outputs.
  EXPECT_TRUE(output_packets.empty());

  // Cleanup.
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

inline constexpr absl::string_view kGeneratorNodeName = "GeneratorNode";
struct GeneratorNode : Node<kGeneratorNodeName> {
  template <typename S>
  struct Contract {
    SideOutput<S, int> side_output{"INT"};
  };
};

class GeneratorNodeImpl : public Calculator<GeneratorNode, GeneratorNodeImpl> {
 public:
  absl::Status Open(CalculatorContext<GeneratorNode>& cc) final {
    cc.side_output.Set(42);
    return absl::OkStatus();
  }
};

TEST(CalculatorTest, CanRunGeneratorCalculator) {
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        output_side_packet: "value"
        node { calculator: "GeneratorNode" output_side_packet: "INT:value" }
      )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(std::move(config)));

  // Starting and sending inputs.
  MP_ASSERT_OK(graph.Run());

  // Verifying outputs.
  MP_ASSERT_OK_AND_ASSIGN(mediapipe::Packet value,
                          graph.GetOutputSidePacket("value"));
  ASSERT_FALSE(value.IsEmpty());
  EXPECT_EQ(value.Get<int>(), 42);
}

inline constexpr absl::string_view kInvalidGeneratorNodeName =
    "InvalidGeneratorNode";
struct InvalidGeneratorNode : Node<kInvalidGeneratorNodeName> {
  template <typename S>
  struct Contract {
    SideOutput<S, int> side_output{"INT"};
    Output<S, int> output{"INT_STREAM"};
  };
};

class InvalidGeneratorNodeImpl
    : public Calculator<InvalidGeneratorNode, InvalidGeneratorNodeImpl> {
 public:
  absl::Status Open(CalculatorContext<InvalidGeneratorNode>& cc) final {
    cc.side_output.Set(42);
    return absl::OkStatus();
  }
};

TEST(CalculatorTest, FailsProperlyForInvalidGeneratorCalculator) {
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        output_side_packet: "value"
        node {
          calculator: "InvalidGeneratorNode"
          output_side_packet: "INT:value"
          output_stream: "INT_STREAM:value_stream"
        }
      )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(std::move(config)));

  EXPECT_THAT(graph.Run(),
              StatusIs(absl::StatusCode::kUnimplemented,
                       testing::HasSubstr("`Process` must be implemented")));
}

TEST(CalculatorTest, FailsOnMaxInFlightConfigForSimultaneousRuns) {
  CalculatorGraphConfig config = ParseTextProtoOrDie<CalculatorGraphConfig>(
      R"pb(
        input_stream: "IN:in"
        node {
          calculator: "PassThrough"
          input_stream: "IN:in"
          output_stream: "OUT:out"
          max_in_flight: 20
        }
      )pb");

  CalculatorGraph graph;
  EXPECT_THAT(graph.Initialize(std::move(config)),
              StatusIs(absl::StatusCode::kInternal,
                       testing::HasSubstr("single invocation")));
}

}  // namespace mediapipe::api3
