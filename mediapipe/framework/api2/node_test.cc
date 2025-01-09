#include "mediapipe/framework/api2/node.h"

#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/contract.h"
#include "mediapipe/framework/api2/node_test.pb.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/api2/test_contracts.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace api2 {
namespace test {

// Returns the packet values for a vector of Packets.
template <typename T>
std::vector<T> PacketValues(const std::vector<mediapipe::Packet>& packets) {
  std::vector<T> result;
  for (const auto& packet : packets) {
    result.push_back(packet.Get<T>());
  }
  return result;
}

class FooImpl : public NodeImpl<Foo, FooImpl> {
 public:
  absl::Status Process(CalculatorContext* cc) override {
    float bias = kBias(cc).GetOr(0.0);
    float scale = kScale(cc).GetOr(1.0);
    kOut(cc).Send(*kBase(cc) * scale + bias);
    return {};
  }
};

class Foo3 : public FunctionNode<Foo3> {
 public:
  static constexpr Input<int> kBase{"BASE"};
  static constexpr Input<float>::Optional kScale{"SCALE"};
  static constexpr Output<float> kOut{"OUT"};
  static constexpr SideInput<float>::Optional kBias{"BIAS"};

  static float foo(int base, Packet<float> bias, Packet<float> scale) {
    return base * scale.GetOr(1.0) + bias.GetOr(0.0);
  }

  // TODO: add support for methods.
  MEDIAPIPE_NODE_INTERFACE(Foo3, ProcessFn(&foo, kBase, kBias, kScale, kOut));
};

class Foo4 : public FunctionNode<Foo4> {
 public:
  static float foo(int base, Packet<float> bias, Packet<float> scale) {
    return base * scale.GetOr(1.0) + bias.GetOr(0.0);
  }

  MEDIAPIPE_NODE_INTERFACE(Foo4, ProcessFn(&foo, Input<int>{"BASE"},
                                           SideInput<float>::Optional{"BIAS"},
                                           Input<float>::Optional{"SCALE"},
                                           Output<float>{"OUT"}));
};

class Foo5 : public FunctionNode<Foo5> {
 public:
  MEDIAPIPE_NODE_INTERFACE(
      Foo5, ProcessFn(
                [](int base, Packet<float> bias, Packet<float> scale) {
                  return base * scale.GetOr(1.0) + bias.GetOr(0.0);
                },
                Input<int>{"BASE"}, SideInput<float>::Optional{"BIAS"},
                Input<float>::Optional{"SCALE"}, Output<float>{"OUT"}));
};

class Foo2Impl : public NodeImpl<Foo2, Foo2Impl> {
 public:
  absl::Status Process(CalculatorContext* cc) override {
    float bias = SideIn(MPP_TAG("BIAS"), cc).GetOr(0.0);
    float scale = In(MPP_TAG("SCALE"), cc).GetOr(1.0);
    Out(MPP_TAG("OUT"), cc).Send(*In(MPP_TAG("BASE"), cc) * scale + bias);
    return {};
  }
};

class BarImpl : public NodeImpl<Bar, BarImpl> {
 public:
  absl::Status Process(CalculatorContext* cc) override {
    Packet p = kIn(cc);
    kOut(cc).Send(p);
    return {};
  }
};

class BazImpl : public NodeImpl<Baz> {
 public:
  static absl::Status UpdateContract(CalculatorContract* cc) { return {}; }

  absl::Status Process(CalculatorContext* cc) override {
    for (int i = 0; i < kData(cc).Count(); ++i) {
      kDataOut(cc)[i].Send(kData(cc)[i]);
    }
    return {};
  }
};
MEDIAPIPE_NODE_IMPLEMENTATION(BazImpl);

class IntForwarderImpl : public NodeImpl<IntForwarder, IntForwarderImpl> {
 public:
  absl::Status Process(CalculatorContext* cc) override {
    kOut(cc).Send(*kIn(cc));
    return {};
  }
};

class ToFloatImpl : public NodeImpl<ToFloat, ToFloatImpl> {
 public:
  absl::Status Process(CalculatorContext* cc) override {
    kIn(cc).Visit([cc](auto x) { kOut(cc).Send(x); });
    return {};
  }
};

TEST(NodeTest, GetContract) {
  // In the old API, contracts are defined "backwards"; first you fill it in
  // with what you have in the graph, then you let the calculator fill it in
  // with what it expects, and then you see if they match.
  const CalculatorGraphConfig::Node node_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "Foo"
        input_stream: "BASE:base"
        input_stream: "SCALE:scale"
        output_stream: "OUT:out"
      )pb");
  mediapipe::CalculatorContract contract;
  MP_EXPECT_OK(contract.Initialize(node_config));
  MP_EXPECT_OK(Foo::Contract::GetContract(&contract));
  MP_EXPECT_OK(ValidatePacketTypeSet(contract.Inputs()));
  MP_EXPECT_OK(ValidatePacketTypeSet(contract.Outputs()));
}

TEST(NodeTest, GetContractMulti) {
  const CalculatorGraphConfig::Node node_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "Baz"
        input_stream: "DATA:0:b"
        input_stream: "DATA:1:c"
        output_stream: "DATA:0:d"
        output_stream: "DATA:1:e"
      )pb");
  mediapipe::CalculatorContract contract;
  MP_EXPECT_OK(contract.Initialize(node_config));
  MP_EXPECT_OK(Baz::Contract::GetContract(&contract));
  MP_EXPECT_OK(ValidatePacketTypeSet(contract.Inputs()));
  MP_EXPECT_OK(ValidatePacketTypeSet(contract.Outputs()));
}

TEST(NodeTest, CreateByName) {
  MP_EXPECT_OK(CalculatorBaseRegistry::CreateByName("Foo"));
}

void RunFooCalculatorInGraph(const std::string& foo_name) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          absl::Substitute(R"(
        input_stream: "base"
        input_stream: "scale"
        output_stream: "out"
        node {
          calculator: "$0"
          input_stream: "BASE:base"
          input_stream: "SCALE:scale"
          output_stream: "OUT:out"
        }
      )",
                           foo_name));
  std::vector<mediapipe::Packet> out_packets;
  tool::AddVectorSink("out", &config, &out_packets);
  mediapipe::CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(config, {}));
  MP_EXPECT_OK(graph.StartRun({}));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "base", mediapipe::MakePacket<int>(10).At(Timestamp(1))));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "scale", mediapipe::MakePacket<float>(2.0).At(Timestamp(1))));
  MP_EXPECT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());
  EXPECT_THAT(PacketValues<float>(out_packets), testing::ElementsAre(20.0));
}

TEST(NodeTest, RunInGraph) { RunFooCalculatorInGraph("Foo"); }

TEST(NodeTest, RunInGraph3) { RunFooCalculatorInGraph("Foo3"); }

TEST(NodeTest, RunInGraph4) { RunFooCalculatorInGraph("Foo4"); }

TEST(NodeTest, RunInGraph5) { RunFooCalculatorInGraph("Foo5"); }

TEST(NodeTest, OptionalStream) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "base"
        input_side_packet: "bias"
        output_stream: "out"
        node {
          calculator: "Foo"
          input_stream: "BASE:base"
          input_side_packet: "BIAS:bias"
          output_stream: "OUT:out"
        }
      )pb");
  std::vector<mediapipe::Packet> out_packets;
  tool::AddVectorSink("out", &config, &out_packets);
  mediapipe::CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(config, {}));
  MP_EXPECT_OK(graph.StartRun({{"bias", mediapipe::MakePacket<float>(30.0)}}));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "base", mediapipe::MakePacket<int>(10).At(Timestamp(1))));
  MP_EXPECT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());
  EXPECT_THAT(PacketValues<float>(out_packets), testing::ElementsAre(40.0));
}

TEST(NodeTest, DynamicTypes) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "in"
        output_stream: "out"
        node {
          calculator: "Bar"
          input_stream: "IN:in"
          output_stream: "OUT:bar"
        }
        node {
          calculator: "IntForwarder"
          input_stream: "IN:bar"
          output_stream: "OUT:out"
        }
      )pb");
  std::vector<mediapipe::Packet> out_packets;
  tool::AddVectorSink("out", &config, &out_packets);
  mediapipe::CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(config, {}));
  MP_EXPECT_OK(graph.StartRun({}));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "in", mediapipe::MakePacket<int>(10).At(Timestamp(1))));
  MP_EXPECT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());
  EXPECT_THAT(PacketValues<int>(out_packets), testing::ElementsAre(10));
}

TEST(NodeTest, MultiPort) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "in0"
        input_stream: "in1"
        output_stream: "out0"
        output_stream: "out1"
        node {
          calculator: "Baz"
          input_stream: "DATA:0:in0"
          input_stream: "DATA:1:in1"
          output_stream: "DATA:0:baz0"
          output_stream: "DATA:1:baz1"
        }
        node {
          calculator: "IntForwarder"
          input_stream: "IN:baz0"
          output_stream: "OUT:out0"
        }
        node {
          calculator: "IntForwarder"
          input_stream: "IN:baz1"
          output_stream: "OUT:out1"
        }
      )pb");
  std::vector<mediapipe::Packet> out0_packets;
  std::vector<mediapipe::Packet> out1_packets;
  tool::AddVectorSink("out0", &config, &out0_packets);
  tool::AddVectorSink("out1", &config, &out1_packets);
  mediapipe::CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(config, {}));
  MP_EXPECT_OK(graph.StartRun({}));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "in0", mediapipe::MakePacket<int>(10).At(Timestamp(1))));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "in1", mediapipe::MakePacket<int>(5).At(Timestamp(1))));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "in0", mediapipe::MakePacket<int>(15).At(Timestamp(2))));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "in1", mediapipe::MakePacket<int>(7).At(Timestamp(2))));
  MP_EXPECT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());
  std::vector<int> out0_values;
  std::vector<int> out1_values;
  for (auto& packet : out0_packets) {
    out0_values.push_back(packet.Get<int>());
  }
  for (auto& packet : out1_packets) {
    out1_values.push_back(packet.Get<int>());
  }
  EXPECT_EQ(out0_values, (std::vector<int>{10, 15}));
  EXPECT_EQ(out1_values, (std::vector<int>{5, 7}));
}

struct SideFallback : public Node {
  static constexpr Input<int> kIn{"IN"};
  static constexpr Input<int>::SideFallback kFactor{"FACTOR"};
  static constexpr Output<int> kOut{"OUT"};

  MEDIAPIPE_NODE_CONTRACT(kIn, kFactor, kOut);

  absl::Status Process(CalculatorContext* cc) override {
    kOut(cc).Send(kIn(cc).Get() * kFactor(cc).Get());
    return {};
  }
};
MEDIAPIPE_REGISTER_NODE(SideFallback);

TEST(NodeTest, SideFallbackWithStream) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "in"
        input_stream: "factor"
        output_stream: "out"
        node {
          calculator: "SideFallback"
          input_stream: "IN:in"
          input_stream: "FACTOR:factor"
          output_stream: "OUT:out"
        }
      )pb");
  std::vector<int> outputs;
  mediapipe::CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(config, {}));
  MP_EXPECT_OK(
      graph.ObserveOutputStream("out", [&outputs](const mediapipe::Packet& p) {
        outputs.push_back(p.Get<int>());
        return absl::OkStatus();
      }));
  MP_EXPECT_OK(graph.StartRun({}));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "in", mediapipe::MakePacket<int>(10).At(Timestamp(0))));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "factor", mediapipe::MakePacket<int>(2).At(Timestamp(0))));
  MP_EXPECT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());
  EXPECT_EQ(outputs, std::vector<int>{20});
}

TEST(NodeTest, SideFallbackWithSide) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "in"
        input_side_packet: "factor"
        output_stream: "out"
        node {
          calculator: "SideFallback"
          input_stream: "IN:in"
          input_side_packet: "FACTOR:factor"
          output_stream: "OUT:out"
        }
      )pb");
  std::vector<int> outputs;
  mediapipe::CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(config, {}));
  MP_EXPECT_OK(
      graph.ObserveOutputStream("out", [&outputs](const mediapipe::Packet& p) {
        outputs.push_back(p.Get<int>());
        return absl::OkStatus();
      }));
  MP_EXPECT_OK(graph.StartRun({{"factor", mediapipe::MakePacket<int>(2)}}));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "in", mediapipe::MakePacket<int>(10).At(Timestamp(0))));
  MP_EXPECT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());
  EXPECT_EQ(outputs, std::vector<int>{20});
}

TEST(NodeTest, SideFallbackWithNone) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "in"
        output_stream: "out"
        node {
          calculator: "SideFallback"
          input_stream: "IN:in"
          output_stream: "OUT:out"
        }
      )pb");
  std::vector<int> outputs;
  mediapipe::CalculatorGraph graph;
  auto status = graph.Initialize(config, {});
  EXPECT_THAT(status.message(), testing::HasSubstr("must be connected"));
}

TEST(NodeTest, SideFallbackWithBoth) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "in"
        input_stream: "factor"
        input_side_packet: "factor_side"
        output_stream: "out"
        node {
          calculator: "SideFallback"
          input_stream: "IN:in"
          input_stream: "FACTOR:factor"
          input_side_packet: "FACTOR:factor_side"
          output_stream: "OUT:out"
        }
      )pb");
  std::vector<int> outputs;
  mediapipe::CalculatorGraph graph;
  auto status = graph.Initialize(config, {});
  EXPECT_THAT(status.message(), testing::HasSubstr("not both"));
}

TEST(NodeTest, OneOf) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "in"
        output_stream: "out"
        node {
          calculator: "ToFloat"
          input_stream: "IN:in"
          output_stream: "OUT:out"
        }
      )pb");
  std::vector<mediapipe::Packet> out_packets;
  tool::AddVectorSink("out", &config, &out_packets);
  mediapipe::CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(config, {}));
  MP_EXPECT_OK(graph.StartRun({}));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "in", mediapipe::MakePacket<int>(10).At(Timestamp(1))));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "in", mediapipe::MakePacket<float>(5.0).At(Timestamp(2))));
  MP_EXPECT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());
  EXPECT_THAT(PacketValues<float>(out_packets), testing::ElementsAre(10, 5.0));
}

struct DropEvenTimestamps : public Node {
  static constexpr Input<AnyType> kIn{"IN"};
  static constexpr Output<SameType<kIn>> kOut{"OUT"};

  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  absl::Status Process(CalculatorContext* cc) override {
    if (cc->InputTimestamp().Value() % 2) {
      kOut(cc).Send(kIn(cc));
    }
    return {};
  }
};
MEDIAPIPE_REGISTER_NODE(DropEvenTimestamps);

struct ListIntPackets : public Node {
  static constexpr Input<int>::Multiple kIn{"INT"};
  static constexpr Output<std::string> kOut{"STR"};

  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  absl::Status Process(CalculatorContext* cc) override {
    std::string result = absl::StrCat(cc->InputTimestamp().DebugString(), ":");
    for (int i = 0; i < kIn(cc).Count(); ++i) {
      if (kIn(cc)[i].IsEmpty()) {
        absl::StrAppend(&result, " empty");
      } else {
        absl::StrAppend(&result, " ", *kIn(cc)[i]);
      }
    }
    kOut(cc).Send(std::move(result));
    return {};
  }
};
MEDIAPIPE_REGISTER_NODE(ListIntPackets);

TEST(NodeTest, DefaultTimestampChange0) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "a"
        input_stream: "b"
        output_stream: "out"
        node {
          calculator: "DropEvenTimestamps"
          input_stream: "IN:a"
          output_stream: "OUT:a2"
        }
        node {
          calculator: "IntForwarder"
          input_stream: "IN:a2"
          output_stream: "OUT:a3"
        }
        node {
          calculator: "ListIntPackets"
          input_stream: "INT:0:a3"
          input_stream: "INT:1:b"
          output_stream: "STR:out"
        }
      )pb");
  std::vector<mediapipe::Packet> out_packets;
  tool::AddVectorSink("out", &config, &out_packets);
  mediapipe::CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(config, {}));
  MP_EXPECT_OK(graph.StartRun({}));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "a", mediapipe::MakePacket<int>(10).At(Timestamp(2))));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "b", mediapipe::MakePacket<int>(10).At(Timestamp(2))));
  MP_EXPECT_OK(graph.WaitUntilIdle());
  // The packet sent to a should have been dropped, but the timestamp bound
  // should be forwarded by IntForwarder, and ListIntPackets should have run.
  EXPECT_THAT(PacketValues<std::string>(out_packets),
              testing::ElementsAre("2: empty 10"));
  MP_EXPECT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());
}

struct ConsumerNode : public Node {
  static constexpr Input<int> kInt{"INT"};
  static constexpr Input<AnyType> kGeneric{"ANY"};
  static constexpr Input<OneOf<int, float>> kOneOf{"NUM"};

  MEDIAPIPE_NODE_CONTRACT(kInt, kGeneric, kOneOf);

  absl::Status Process(CalculatorContext* cc) override {
    MP_ASSIGN_OR_RETURN(auto maybe_int, kInt(cc).Consume());
    MP_ASSIGN_OR_RETURN(auto maybe_float, kGeneric(cc).Consume<float>());
    MP_ASSIGN_OR_RETURN(auto maybe_int2, kOneOf(cc).Consume<int>());
    return {};
  }
};
MEDIAPIPE_REGISTER_NODE(ConsumerNode);

TEST(NodeTest, ConsumeInputs) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "int"
        input_stream: "any"
        input_stream: "num"
        node {
          calculator: "ConsumerNode"
          input_stream: "INT:int"
          input_stream: "ANY:any"
          input_stream: "NUM:num"
        }
      )pb");
  mediapipe::CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(config, {}));
  MP_EXPECT_OK(graph.StartRun({}));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "int", mediapipe::MakePacket<int>(10).At(Timestamp(0))));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "any", mediapipe::MakePacket<float>(10).At(Timestamp(0))));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "num", mediapipe::MakePacket<int>(10).At(Timestamp(0))));
  MP_EXPECT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());
}

// Just to test that single-port contracts work.
struct LogSinkNode : public Node {
  static constexpr Input<int> kIn{"IN"};

  MEDIAPIPE_NODE_CONTRACT(kIn);

  absl::Status Process(CalculatorContext* cc) override {
    ABSL_LOG(INFO) << "LogSinkNode received: " << kIn(cc).Get();
    return {};
  }
};
MEDIAPIPE_REGISTER_NODE(LogSinkNode);

/******************************************************************************/
// Test different combinations of side packet connections.
class SidePacketConnectionTestSuite
    : public testing::TestWithParam<std::tuple<
          bool,  // Wether graph provides side input,
          bool,  // Whether graph connects side output.
          bool,  // Whether producer node internally connects to consumer node.
          bool   // Whether graph-level side packet connects to the first port.
          >> {};

struct SidePacktProducerNode : public Node {
  static constexpr SideOutput<int>::Optional kSideOut1{"SIDE_OUT_1"};
  static constexpr SideOutput<int>::Optional kSideOut2{"SIDE_OUT_2"};

  MEDIAPIPE_NODE_CONTRACT(kSideOut1, kSideOut2);

  absl::Status Open(CalculatorContext* cc) override {
    const auto& options = cc->Options<SidePacketConnectionTestOptions>();
    if (options.graph_connect_first_port()) {
      RET_CHECK_EQ(options.graph_has_side_output(),
                   kSideOut1(cc).IsConnected());
      RET_CHECK_EQ(options.internal_side_connection(),
                   kSideOut2(cc).IsConnected());
    } else {
      RET_CHECK_EQ(options.graph_has_side_output(),
                   kSideOut2(cc).IsConnected());
      RET_CHECK_EQ(options.internal_side_connection(),
                   kSideOut1(cc).IsConnected());
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const auto& options = cc->Options<SidePacketConnectionTestOptions>();
    if (options.graph_connect_first_port()) {
      RET_CHECK_EQ(options.graph_has_side_output(),
                   kSideOut1(cc).IsConnected());
      RET_CHECK_EQ(options.internal_side_connection(),
                   kSideOut2(cc).IsConnected());
    } else {
      RET_CHECK_EQ(options.graph_has_side_output(),
                   kSideOut2(cc).IsConnected());
      RET_CHECK_EQ(options.internal_side_connection(),
                   kSideOut1(cc).IsConnected());
    }
    return absl::OkStatus();
  }
};
MEDIAPIPE_REGISTER_NODE(SidePacktProducerNode);

struct SidePacktConsumerNode : public Node {
  static constexpr SideInput<int>::Optional kSideIn1{"SIDE_IN_1"};
  static constexpr SideInput<int>::Optional kSideIn2{"SIDE_IN_2"};

  MEDIAPIPE_NODE_CONTRACT(kSideIn1, kSideIn2);

  absl::Status Open(CalculatorContext* cc) override {
    const auto& options = cc->Options<SidePacketConnectionTestOptions>();
    if (options.graph_connect_first_port()) {
      RET_CHECK_EQ(options.graph_has_side_input(), kSideIn1(cc).IsConnected());
      RET_CHECK_EQ(options.internal_side_connection(),
                   kSideIn2(cc).IsConnected());
    } else {
      RET_CHECK_EQ(options.graph_has_side_input(), kSideIn2(cc).IsConnected());
      RET_CHECK_EQ(options.internal_side_connection(),
                   kSideIn1(cc).IsConnected());
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const auto& options = cc->Options<SidePacketConnectionTestOptions>();
    if (options.graph_connect_first_port()) {
      RET_CHECK_EQ(options.graph_has_side_input(), kSideIn1(cc).IsConnected());
      RET_CHECK_EQ(options.internal_side_connection(),
                   kSideIn2(cc).IsConnected());
    } else {
      RET_CHECK_EQ(options.graph_has_side_input(), kSideIn2(cc).IsConnected());
      RET_CHECK_EQ(options.internal_side_connection(),
                   kSideIn1(cc).IsConnected());
    }
    return absl::OkStatus();
  }
};
MEDIAPIPE_REGISTER_NODE(SidePacktConsumerNode);

INSTANTIATE_TEST_SUITE_P(
    SidePacketConnectionTestGroup, SidePacketConnectionTestSuite,
    testing::Combine(testing::Values(true, false), testing::Values(true, false),
                     testing::Values(true, false),
                     testing::Values(true, false)),
    [](const testing::TestParamInfo<SidePacketConnectionTestSuite::ParamType>&
           info) {
      std::string name =
          absl::StrCat(std::get<0>(info.param) ? "GraphSideIn" : "",
                       std::get<1>(info.param) ? "GraphSideOut" : "",
                       std::get<2>(info.param) ? "InternalSideConnection" : "",
                       std::get<3>(info.param) ? "GraphConnectFirstPort"
                                               : "GraphConnectSecondPort");
      return name;
    });

TEST_P(SidePacketConnectionTestSuite, SidePacketConnectionTest) {
  const bool graph_side_in = std::get<0>(GetParam());
  const bool graph_side_out = std::get<1>(GetParam());
  const bool internal_side_connection = std::get<2>(GetParam());
  const bool graph_connect_first_port = std::get<3>(GetParam());
  builder::Graph builder;
  SidePacketConnectionTestOptions options;
  options.set_graph_has_side_input(graph_side_in);
  options.set_graph_has_side_output(graph_side_out);
  options.set_internal_side_connection(internal_side_connection);
  options.set_graph_connect_first_port(graph_connect_first_port);

  auto& producer = builder.AddNode("SidePacktProducerNode");
  producer.GetOptions<SidePacketConnectionTestOptions>() = options;
  auto& consumer = builder.AddNode("SidePacktConsumerNode");
  consumer.GetOptions<SidePacketConnectionTestOptions>() = options;
  std::map<std::string, mediapipe::Packet> side_packets;
  if (graph_side_in) {
    builder.SideIn("").SetName("side_in") >>
        consumer.SideIn(graph_connect_first_port ? "SIDE_IN_1" : "SIDE_IN_2");
    side_packets["side_in"] = MakePacket<int>(10);
  }
  if (graph_side_out) {
    producer.SideOut(graph_connect_first_port ? "SIDE_OUT_1" : "SIDE_OUT_2") >>
        builder.SideOut("SIDE_OUT");
  }
  if (internal_side_connection) {
    if (graph_connect_first_port) {
      producer.SideOut("SIDE_OUT_2") >> consumer.SideIn("SIDE_IN_2");
    } else {
      producer.SideOut("SIDE_OUT_1") >> consumer.SideIn("SIDE_IN_1");
    }
  }
  CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(builder.GetConfig(), {}));
  MP_EXPECT_OK(graph.StartRun(std::move(side_packets)));
  MP_EXPECT_OK(graph.WaitUntilDone());
}

/******************************************************************************/

}  // namespace test
}  // namespace api2
}  // namespace mediapipe
