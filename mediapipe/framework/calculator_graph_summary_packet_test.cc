#include "absl/status/status.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Node;
using ::mediapipe::api2::Output;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Value;

namespace {

MATCHER_P2(IntPacket, value, timestamp, "") {
  *result_listener << "where object is (value: " << arg.template Get<int>()
                   << ", timestamp: " << arg.Timestamp() << ")";
  return Value(arg.template Get<int>(), Eq(value)) &&
         Value(arg.Timestamp(), Eq(timestamp));
}

// Calculates and produces sum of all passed inputs when no more packets can be
// expected on the input stream.
class SummaryPacketCalculator : public Node {
 public:
  static constexpr Input<int> kIn{"IN"};
  static constexpr Output<int> kOut{"SUMMARY"};

  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  static absl::Status UpdateContract(CalculatorContract* cc) {
    // Makes sure there are no automatic timestamp bound updates when Process
    // is called.
    cc->SetTimestampOffset(TimestampDiff::Unset());
    // Currently, only ImmediateInputStreamHandler supports "done" timestamp
    // bound update. (ImmediateInputStreamhandler handles multiple input
    // streams differently, so, in that case, calculator adjustments may be
    // required.)
    // TODO: update all input stream handlers to support "done"
    // timestamp bound update.
    cc->SetInputStreamHandler("ImmediateInputStreamHandler");
    // Enables processing timestamp bound updates. For this use case we are
    // specifically interested in "done" timestamp bound update. (E.g. when
    // all input packet sources are closed.)
    cc->SetProcessTimestampBounds(true);
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    if (!kIn(cc).IsEmpty()) {
      value_ += kIn(cc).Get();
      value_set_ = true;
    }

    if (kOut(cc).IsClosed()) {
      // This can happen:
      // 1. If, during previous invocation, kIn(cc).IsDone() == true (e.g.
      //    source calculator finished generating packets sent to kIn) and
      //    HasNextAllowedInStream() == true (which is an often case).
      // 2. For Timestamp::PreStream, ImmediateInputStreamHandler will still
      //    invoke Process() with Timestamp::Max to indicate "Done" timestamp
      //    bound update.
      return absl::OkStatus();
    }

    // TODO: input stream holding a packet with timestamp that has
    // no next timestamp allowed in stream should always result in
    // InputStream::IsDone() == true.
    if (kIn(cc).IsDone() || !cc->InputTimestamp().HasNextAllowedInStream()) {
      // `Process` may or may not be invoked for "done" timestamp bound when
      // upstream calculator fails in `Close`. Hence, extra care is needed to
      // identify whether the calculator needs to send output.
      // TODO: remove when "done" timestamp bound flakiness fixed.
      if (value_set_) {
        // kOut(cc).Send(value_) can be used here as well, however in the case
        // of source calculator sending inputs into kIn the resulting timestamp
        // is not well defined (e.g. it can be the last packet timestamp or
        // Timestamp::Max())
        // TODO: last packet from source should always result in
        // InputStream::IsDone() == true.
        kOut(cc).Send(value_, Timestamp::Max());
      }
      kOut(cc).Close();
    }
    return absl::OkStatus();
  }

 private:
  int value_ = 0;
  bool value_set_ = false;
};
MEDIAPIPE_REGISTER_NODE(SummaryPacketCalculator);

TEST(SummaryPacketCalculatorUseCaseTest,
     ProducesSummaryPacketOnClosingAllPacketSources) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: 'input'
    node {
      calculator: "SummaryPacketCalculator"
      input_stream: 'IN:input'
      output_stream: 'SUMMARY:output'
    }
  )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(output_packets, IsEmpty());

  auto send_packet = [&graph](int value, Timestamp timestamp) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "input", MakePacket<int>(value).At(timestamp)));
  };

  send_packet(10, Timestamp(10));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(output_packets, IsEmpty());

  send_packet(20, Timestamp(11));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(output_packets, IsEmpty());

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
  EXPECT_THAT(output_packets, ElementsAre(IntPacket(30, Timestamp::Max())));
}

TEST(SummaryPacketCalculatorUseCaseTest, ProducesSummaryPacketOnMaxTimestamp) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: 'input'
    node {
      calculator: "SummaryPacketCalculator"
      input_stream: 'IN:input'
      output_stream: 'SUMMARY:output'
    }
  )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(output_packets, IsEmpty());

  auto send_packet = [&graph](int value, Timestamp timestamp) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "input", MakePacket<int>(value).At(timestamp)));
  };

  send_packet(10, Timestamp(10));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(output_packets, IsEmpty());

  send_packet(20, Timestamp::Max());
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(output_packets, ElementsAre(IntPacket(30, Timestamp::Max())));

  output_packets.clear();
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
  EXPECT_THAT(output_packets, IsEmpty());
}

TEST(SummaryPacketCalculatorUseCaseTest,
     ProducesSummaryPacketOnPreStreamTimestamp) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: 'input'
    node {
      calculator: "SummaryPacketCalculator"
      input_stream: 'IN:input'
      output_stream: 'SUMMARY:output'
    }
  )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(output_packets, IsEmpty());

  auto send_packet = [&graph](int value, Timestamp timestamp) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "input", MakePacket<int>(value).At(timestamp)));
  };

  send_packet(10, Timestamp::PreStream());
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(output_packets, ElementsAre(IntPacket(10, Timestamp::Max())));

  output_packets.clear();
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
  EXPECT_THAT(output_packets, IsEmpty());
}

TEST(SummaryPacketCalculatorUseCaseTest,
     ProducesSummaryPacketOnPostStreamTimestamp) {
  std::vector<Packet> output_packets;
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: 'input'
        node {
          calculator: "SummaryPacketCalculator"
          input_stream: 'IN:input'
          output_stream: 'SUMMARY:output'
        }
      )pb");
  tool::AddVectorSink("output", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(output_packets, IsEmpty());

  auto send_packet = [&graph](int value, Timestamp timestamp) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "input", MakePacket<int>(value).At(timestamp)));
  };

  send_packet(10, Timestamp::PostStream());
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(output_packets, ElementsAre(IntPacket(10, Timestamp::Max())));

  output_packets.clear();
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
  EXPECT_THAT(output_packets, IsEmpty());
}

class IntGeneratorCalculator : public Node {
 public:
  static constexpr Output<int> kOut{"INT"};

  MEDIAPIPE_NODE_CONTRACT(kOut);

  absl::Status Process(CalculatorContext* cc) final {
    kOut(cc).Send(20, Timestamp(0));
    kOut(cc).Send(10, Timestamp(1000));
    return tool::StatusStop();
  }
};
MEDIAPIPE_REGISTER_NODE(IntGeneratorCalculator);

TEST(SummaryPacketCalculatorUseCaseTest,
     ProducesSummaryPacketOnSourceCalculatorCompletion) {
  std::vector<Packet> output_packets;
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "IntGeneratorCalculator"
          output_stream: "INT:int_value"
        }
        node {
          calculator: "SummaryPacketCalculator"
          input_stream: "IN:int_value"
          output_stream: "SUMMARY:output"
        }
      )pb");
  tool::AddVectorSink("output", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_EXPECT_OK(graph.WaitUntilDone());
  EXPECT_THAT(output_packets, ElementsAre(IntPacket(30, Timestamp::Max())));
}

class EmitOnCloseCalculator : public Node {
 public:
  static constexpr Input<int> kIn{"IN"};
  static constexpr Output<int> kOut{"INT"};

  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  absl::Status Process(CalculatorContext* cc) final { return absl::OkStatus(); }

  absl::Status Close(CalculatorContext* cc) final {
    kOut(cc).Send(20, Timestamp(0));
    kOut(cc).Send(10, Timestamp(1000));
    return absl::OkStatus();
  }
};
MEDIAPIPE_REGISTER_NODE(EmitOnCloseCalculator);

TEST(SummaryPacketCalculatorUseCaseTest,
     ProducesSummaryPacketOnAnotherCalculatorClosure) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "input"
    node {
      calculator: "EmitOnCloseCalculator"
      input_stream: "IN:input"
      output_stream: "INT:int_value"
    }
    node {
      calculator: "SummaryPacketCalculator"
      input_stream: "IN:int_value"
      output_stream: "SUMMARY:output"
    }
  )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(output_packets, IsEmpty());

  MP_ASSERT_OK(graph.CloseInputStream("input"));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(output_packets, ElementsAre(IntPacket(30, Timestamp::Max())));

  output_packets.clear();
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
  EXPECT_THAT(output_packets, IsEmpty());
}

class FailureInCloseCalculator : public Node {
 public:
  static constexpr Input<int> kIn{"IN"};
  static constexpr Output<int> kOut{"INT"};

  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  absl::Status Process(CalculatorContext* cc) final { return absl::OkStatus(); }

  absl::Status Close(CalculatorContext* cc) final {
    return absl::InternalError("error");
  }
};
MEDIAPIPE_REGISTER_NODE(FailureInCloseCalculator);

TEST(SummaryPacketCalculatorUseCaseTest,
     DoesNotProduceSummaryPacketWhenUpstreamCalculatorFailsInClose) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "input"
    node {
      calculator: "FailureInCloseCalculator"
      input_stream: "IN:input"
      output_stream: "INT:int_value"
    }
    node {
      calculator: "SummaryPacketCalculator"
      input_stream: "IN:int_value"
      output_stream: "SUMMARY:output"
    }
  )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(output_packets, IsEmpty());

  MP_ASSERT_OK(graph.CloseInputStream("input"));
  EXPECT_THAT(graph.WaitUntilIdle(),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("error")));
  EXPECT_THAT(output_packets, IsEmpty());
}

class FailureInProcessCalculator : public Node {
 public:
  static constexpr Input<int> kIn{"IN"};
  static constexpr Output<int> kOut{"INT"};

  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  absl::Status Process(CalculatorContext* cc) final {
    return absl::InternalError("error");
  }
};
MEDIAPIPE_REGISTER_NODE(FailureInProcessCalculator);

TEST(SummaryPacketCalculatorUseCaseTest,
     DoesNotProduceSummaryPacketWhenUpstreamCalculatorFailsInProcess) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "input"
    node {
      calculator: "FailureInProcessCalculator"
      input_stream: "IN:input"
      output_stream: "INT:int_value"
    }
    node {
      calculator: "SummaryPacketCalculator"
      input_stream: "IN:int_value"
      output_stream: "SUMMARY:output"
    }
  )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_THAT(output_packets, IsEmpty());

  auto send_packet = [&graph](int value, Timestamp timestamp) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "input", MakePacket<int>(value).At(timestamp)));
  };

  send_packet(10, Timestamp::PostStream());
  EXPECT_THAT(graph.WaitUntilIdle(),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("error")));
  EXPECT_THAT(output_packets, IsEmpty());
}

}  // namespace
}  // namespace mediapipe
