// Copyright 2024 The MediaPipe Authors.
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

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_contract.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Node;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Stream;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::testing::Test;
using ::testing::Value;

MATCHER_P2(PacketOfIntEq, timestamp, value, "") {
  const Timestamp actual_timestamp = arg.Timestamp();
  const int actual_value = arg.template Get<int>();
  return Value(actual_timestamp, Eq(timestamp)) &&
         Value(actual_value, Eq(value));
}

class IncrementCalculator : public Node {
 public:
  static constexpr Input<int> kIn{""};
  static constexpr Output<int> kOut{""};

  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  absl::Status Process(CalculatorContext* cc) override {
    kOut(cc).Send(kIn(cc).Get() + 1);
    return absl::OkStatus();
  }
};

MEDIAPIPE_REGISTER_NODE(IncrementCalculator);

class GraphRunner {
 public:
  GraphRunner(int num_inputs, int num_outputs) {
    Graph builder;
    auto& begin_item_loop_calculator =
        builder.AddNode("BeginItemLoopCalculator");
    auto& increment_calculator = builder.AddNode("IncrementCalculator");
    auto& end_item_loop_calculator = builder.AddNode("EndItemLoopCalculator");

    for (int n = 0; n < num_inputs; ++n) {
      Stream<int> input_stream =
          builder.In(n).SetName(absl::StrCat("int", n)).Cast<int>();
      input_stream >> begin_item_loop_calculator.In("ITEM")[n];
    }

    begin_item_loop_calculator.Out("ITEM") >> increment_calculator.In("");
    begin_item_loop_calculator.Out("BATCH_END") >>
        end_item_loop_calculator.In("BATCH_END");
    increment_calculator.Out("") >> end_item_loop_calculator.In("ITEM");

    for (int n = 0; n < num_outputs; ++n) {
      end_item_loop_calculator.Out("ITEM")[n].SetName(
          absl::StrCat("ints_plus_one", n));
    }

    CalculatorGraphConfig graph_config = builder.GetConfig();

    output_packets_.resize(num_outputs);
    for (int n = 0; n < num_outputs; ++n) {
      tool::AddVectorSink(absl::StrCat("ints_plus_one", n), &graph_config,
                          &output_packets_[n]);
    }
    ABSL_QCHECK_OK(graph_.Initialize(graph_config));
    ABSL_QCHECK_OK(graph_.StartRun({}));
    ABSL_QCHECK_OK(graph_.WaitUntilIdle());
  }

  absl::Status Close() {
    MP_RETURN_IF_ERROR(graph_.CloseAllPacketSources());
    return graph_.WaitUntilDone();
  }

  absl::Status SendPacketsOfInts(Timestamp timestamp,
                                 std::vector<std::optional<int>> ints) {
    for (int n = 0; n < ints.size(); ++n) {
      const std::string name = absl::StrCat("int", n);
      if (ints[n]) {
        MP_RETURN_IF_ERROR(graph_.AddPacketToInputStream(
            name, MakePacket<int>(*ints[n]).At(timestamp)));
      } else {
        MP_RETURN_IF_ERROR(graph_.SetInputStreamTimestampBound(
            name, timestamp.NextAllowedInStream()));
      }
    }
    return graph_.WaitUntilIdle();
  }

  const std::vector<std::vector<Packet>>& output_packets() const {
    return output_packets_;
  }

 private:
  CalculatorGraph graph_;
  std::vector<std::vector<Packet>> output_packets_;
};

TEST(BeginEndLoopCalculatorGraphItemTest, NoItemPackets) {
  GraphRunner runner(/*num_inputs=*/2, /*num_outputs=*/2);

  EXPECT_THAT(runner.output_packets(), ElementsAre(SizeIs(0), SizeIs(0)));
  MP_EXPECT_OK(runner.Close());
}

TEST(BeginEndLoopCalculatorGraphItemTest, AllEmptyItemPackets) {
  GraphRunner runner(/*num_inputs=*/3, /*num_outputs=*/3);

  MP_EXPECT_OK(runner.SendPacketsOfInts(Timestamp(0), {{}, {}, {}}));

  // EndLoopCalc will forward the timestamp bound because there are no elements
  // in collection to output.
  EXPECT_THAT(runner.output_packets(),
              ElementsAre(SizeIs(0), SizeIs(0), SizeIs(0)));

  MP_EXPECT_OK(runner.Close());
}

TEST(BeginEndLoopCalculatorGraphItemTest, MultipleAllEmptyItemPackets) {
  GraphRunner runner(/*num_inputs=*/3, /*num_outputs=*/3);

  MP_EXPECT_OK(runner.SendPacketsOfInts(Timestamp(0), {{}, {}, {}}));
  MP_EXPECT_OK(runner.SendPacketsOfInts(Timestamp(1), {{}, {}, {}}));

  EXPECT_THAT(runner.output_packets(),
              ElementsAre(SizeIs(0), SizeIs(0), SizeIs(0)));

  MP_EXPECT_OK(runner.Close());
}

TEST(BeginEndLoopCalculatorGraphItemTest, NonEmptyItemPackets) {
  GraphRunner runner(/*num_inputs=*/3, /*num_outputs=*/3);

  Timestamp input_timestamp = Timestamp(0);
  MP_EXPECT_OK(runner.SendPacketsOfInts(input_timestamp, {0, 1, 2}));

  EXPECT_THAT(runner.output_packets(),
              ElementsAre(ElementsAre(PacketOfIntEq(input_timestamp, 1)),
                          ElementsAre(PacketOfIntEq(input_timestamp, 2)),
                          ElementsAre(PacketOfIntEq(input_timestamp, 3))));
  MP_EXPECT_OK(runner.Close());
}

TEST(BeginEndLoopCalculatorGraphItemTest, SomeEmptyItemPackets) {
  GraphRunner runner(/*num_inputs=*/3, /*num_outputs=*/3);

  Timestamp input_timestamp = Timestamp(0);
  EXPECT_THAT(runner.SendPacketsOfInts(input_timestamp, {0, 1, {}}),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("Cannot mix")));
}

TEST(BeginEndLoopCalculatorGraphItemTest, MoreOutputsThanInputs) {
  GraphRunner runner(/*num_inputs=*/2, /*num_outputs=*/3);

  Timestamp input_timestamp = Timestamp(0);

  EXPECT_THAT(runner.SendPacketsOfInts(input_timestamp, {3, 5}),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("input items must match")));
  EXPECT_THAT(runner.Close(), StatusIs(absl::StatusCode::kInternal));
}

TEST(BeginEndLoopCalculatorGraphItemTest, LessOutputsThanInputs) {
  GraphRunner runner(/*num_inputs=*/3, /*num_outputs=*/2);

  Timestamp input_timestamp = Timestamp(0);

  EXPECT_THAT(runner.SendPacketsOfInts(input_timestamp, {4, 6, 8}),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("input items must match")));
  EXPECT_THAT(runner.Close(), StatusIs(absl::StatusCode::kInternal));
}

TEST(BeginEndLoopCalculatorGraphItemTest, MultipleInputSets) {
  GraphRunner runner(/*num_inputs=*/2, /*num_outputs=*/2);

  Timestamp input_timestamp0 = Timestamp(0);
  MP_EXPECT_OK(runner.SendPacketsOfInts(input_timestamp0, {0, 1}));

  Timestamp input_timestamp1 = Timestamp(1);
  MP_EXPECT_OK(runner.SendPacketsOfInts(input_timestamp1, {2, 3}));

  Timestamp input_timestamp4 = Timestamp(4);
  MP_EXPECT_OK(runner.SendPacketsOfInts(input_timestamp4, {5, 6}));

  EXPECT_THAT(runner.output_packets(),
              ElementsAre(ElementsAre(PacketOfIntEq(input_timestamp0, 1),
                                      PacketOfIntEq(input_timestamp1, 3),
                                      PacketOfIntEq(input_timestamp4, 6)),
                          ElementsAre(PacketOfIntEq(input_timestamp0, 2),
                                      PacketOfIntEq(input_timestamp1, 4),
                                      PacketOfIntEq(input_timestamp4, 7))));
  MP_EXPECT_OK(runner.Close());
}

TEST(BeginEndLoopCalculatorGraphItemTest, AllowsArbitraryTimestampChange) {
  GraphRunner runner(/*num_inputs=*/1, /*num_outputs=*/1);

  Timestamp input_timestamp1 = Timestamp(1000);
  MP_EXPECT_OK(runner.SendPacketsOfInts(input_timestamp1, {{}}));

  Timestamp input_timestamp2 = Timestamp(1001);
  MP_EXPECT_OK(runner.SendPacketsOfInts(input_timestamp2, {{1}}));

  EXPECT_THAT(runner.output_packets(),
              ElementsAre(ElementsAre(PacketOfIntEq(input_timestamp2, 2))));
  MP_EXPECT_OK(runner.Close());
}

class MultiplyCalculator : public Node {
 public:
  static constexpr Input<int> kInA{"A"};
  static constexpr Input<int> kInB{"B"};
  static constexpr Output<int> kOut{""};

  MEDIAPIPE_NODE_CONTRACT(kInA, kInB, kOut);

  absl::Status Process(CalculatorContext* cc) override {
    kOut(cc).Send(kInA(cc).Get() * kInB(cc).Get());
    return absl::OkStatus();
  }
};

MEDIAPIPE_REGISTER_NODE(MultiplyCalculator);

class CloneGraphRunner {
 public:
  CloneGraphRunner() {
    auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(
        R"pb(
          num_threads: 4
          input_stream: "int0"
          input_stream: "int1"
          input_stream: "clone"
          node {
            calculator: "BeginItemLoopCalculator"
            input_stream: "ITEM:0:int0"
            input_stream: "ITEM:1:int1"
            input_stream: "CLONE:clone"
            output_stream: "ITEM:int_iter"
            output_stream: "CLONE:clone_iter"
            output_stream: "BATCH_END:timestamp"
          }
          node {
            calculator: "MultiplyCalculator"
            input_stream: "A:int_iter"
            input_stream: "B:clone_iter"
            output_stream: "int_times_clone_iter"
          }
          node {
            calculator: "EndItemLoopCalculator"
            input_stream: "ITEM:int_times_clone_iter"
            input_stream: "BATCH_END:timestamp"
            output_stream: "ITEM:0:int_times_clone0"
            output_stream: "ITEM:1:int_times_clone1"
          }
        )pb");
    output_packets_.resize(2);
    tool::AddVectorSink("int_times_clone0", &graph_config, &output_packets_[0]);
    tool::AddVectorSink("int_times_clone1", &graph_config, &output_packets_[1]);

    ABSL_QCHECK_OK(graph_.Initialize(graph_config));
    ABSL_QCHECK_OK(graph_.StartRun({}));
    ABSL_QCHECK_OK(graph_.WaitUntilIdle());
  }

  absl::Status Close() {
    MP_RETURN_IF_ERROR(graph_.CloseAllPacketSources());
    return graph_.WaitUntilDone();
  }

  absl::Status SendPacketsOfInts(Timestamp timestamp, std::optional<int> int0,
                                 std::optional<int> int1,
                                 std::optional<int> clone) {
    absl::flat_hash_map<std::string, std::optional<int>> inputs = {
        {"int0", int0}, {"int1", int1}, {"clone", clone}};
    for (const auto& [name, value] : inputs) {
      if (value) {
        MP_RETURN_IF_ERROR(graph_.AddPacketToInputStream(
            name, MakePacket<int>(*value).At(timestamp)));
      } else {
        MP_RETURN_IF_ERROR(graph_.SetInputStreamTimestampBound(
            name, timestamp.NextAllowedInStream()));
      }
    }
    return graph_.WaitUntilIdle();
  }

  const std::vector<std::vector<Packet>>& output_packets() const {
    return output_packets_;
  }

 private:
  CalculatorGraph graph_;
  std::vector<std::vector<Packet>> output_packets_;
};

TEST(BeginEndLoopCalculatorGraphItemTest, CloneWithNoItemPackets) {
  CloneGraphRunner runner;

  EXPECT_THAT(runner.output_packets(), ElementsAre(SizeIs(0), SizeIs(0)));
  MP_EXPECT_OK(runner.Close());
}

TEST(BeginEndLoopCalculatorGraphItemTest, CloneWithAllEmptyItemPackets) {
  CloneGraphRunner runner;

  MP_EXPECT_OK(runner.SendPacketsOfInts(Timestamp(0), /*int0=*/{}, /*int1=*/{},
                                        /*clone=*/{}));

  // EndLoopCalc will forward the timestamp bound because there are no elements
  // in collection to output.
  EXPECT_THAT(runner.output_packets(), ElementsAre(SizeIs(0), SizeIs(0)));
  MP_EXPECT_OK(runner.Close());
}

TEST(BeginEndLoopCalculatorGraphItemTest, CloneWithEmptyItemPackets) {
  CloneGraphRunner runner;

  MP_EXPECT_OK(runner.SendPacketsOfInts(Timestamp(0), /*int0=*/{}, /*int1=*/{},
                                        /*clone=*/42));

  // EndLoopCalc will forward the timestamp bound because there are no elements
  // in collection to output.
  EXPECT_THAT(runner.output_packets(), ElementsAre(SizeIs(0), SizeIs(0)));
  MP_EXPECT_OK(runner.Close());
}

TEST(BeginEndLoopCalculatorGraphItemTest, CloneWithNonEmptyItemPackets) {
  CloneGraphRunner runner;

  Timestamp input_timestamp = Timestamp(0);
  MP_EXPECT_OK(runner.SendPacketsOfInts(input_timestamp, /*int0=*/2, /*int1=*/3,
                                        /*clone=*/5));

  EXPECT_THAT(runner.output_packets(),
              ElementsAre(ElementsAre(PacketOfIntEq(input_timestamp, 10)),
                          ElementsAre(PacketOfIntEq(input_timestamp, 15))));
  MP_EXPECT_OK(runner.Close());
}

TEST(BeginEndLoopCalculatorGraphItemTest, CloneWithSomeEmptyItemPackets) {
  CloneGraphRunner runner;

  Timestamp input_timestamp = Timestamp(0);
  EXPECT_THAT(runner.SendPacketsOfInts(input_timestamp, /*int0=*/{}, /*int1=*/3,
                                       /*clone=*/5),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("Cannot mix")));
}

// TODO: b/335433439 - Fix issue with death tests and reenable test.

TEST(BeginEndLoopCalculatorGraphItemDeathTest,
     DISABLED_EmptyCloneWithNonEmptyItemPackets) {
  CloneGraphRunner runner;

  Timestamp input_timestamp = Timestamp(0);
  ASSERT_DEATH(runner
                   .SendPacketsOfInts(input_timestamp, /*int0=*/2, /*int1=*/3,
                                      /*clone=*/{})
                   .IgnoreError(),
               "Check failed");
}

TEST(BeginEndLoopCalculatorGraphItemTest, CloneWithMultipleInputSets) {
  CloneGraphRunner runner;

  Timestamp input_timestamp0 = Timestamp(0);
  MP_EXPECT_OK(runner.SendPacketsOfInts(input_timestamp0, /*int0=*/1,
                                        /*int1=*/2,
                                        /*clone=*/5));

  Timestamp input_timestamp1 = Timestamp(1);
  MP_EXPECT_OK(runner.SendPacketsOfInts(input_timestamp1, /*int0=*/{2},
                                        /*int1=*/{3}, /*clone=*/5));

  Timestamp input_timestamp3 = Timestamp(3);
  MP_EXPECT_OK(runner.SendPacketsOfInts(input_timestamp3, /*int0=*/5,
                                        /*int1=*/6,
                                        /*clone=*/5));

  EXPECT_THAT(runner.output_packets(),
              ElementsAre(ElementsAre(PacketOfIntEq(input_timestamp0, 5),
                                      PacketOfIntEq(input_timestamp1, 10),
                                      PacketOfIntEq(input_timestamp3, 25)),
                          ElementsAre(PacketOfIntEq(input_timestamp0, 10),
                                      PacketOfIntEq(input_timestamp1, 15),
                                      PacketOfIntEq(input_timestamp3, 30))));
  MP_EXPECT_OK(runner.Close());
}

}  // namespace
}  // namespace mediapipe
