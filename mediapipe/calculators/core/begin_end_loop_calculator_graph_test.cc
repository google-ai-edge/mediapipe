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

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "mediapipe/calculators/core/begin_loop_calculator.h"
#include "mediapipe/calculators/core/end_loop_calculator.h"
#include "mediapipe/framework/calculator_contract.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"  // NOLINT

namespace mediapipe {
namespace {

MATCHER_P2(PacketOfIntsEq, timestamp, value, "") {
  Timestamp actual_timestamp = arg.Timestamp();
  const auto& actual_value = arg.template Get<std::vector<int>>();
  return testing::Value(actual_timestamp, testing::Eq(timestamp)) &&
         testing::Value(actual_value, testing::ElementsAreArray(value));
}

typedef BeginLoopCalculator<std::vector<int>> BeginLoopIntegerCalculator;
REGISTER_CALCULATOR(BeginLoopIntegerCalculator);

class IncrementCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Outputs().Index(0).Set<int>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const int& input_int = cc->Inputs().Index(0).Get<int>();
    auto output_int = absl::make_unique<int>(input_int + 1);
    cc->Outputs().Index(0).Add(output_int.release(), cc->InputTimestamp());
    return absl::OkStatus();
  }
};

REGISTER_CALCULATOR(IncrementCalculator);

typedef EndLoopCalculator<std::vector<int>> EndLoopIntegersCalculator;
REGISTER_CALCULATOR(EndLoopIntegersCalculator);

class BeginEndLoopCalculatorGraphTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(
        R"pb(
          num_threads: 4
          input_stream: "ints"
          node {
            calculator: "BeginLoopIntegerCalculator"
            input_stream: "ITERABLE:ints"
            output_stream: "ITEM:int"
            output_stream: "BATCH_END:timestamp"
          }
          node {
            calculator: "IncrementCalculator"
            input_stream: "int"
            output_stream: "int_plus_one"
          }
          node {
            calculator: "EndLoopIntegersCalculator"
            input_stream: "ITEM:int_plus_one"
            input_stream: "BATCH_END:timestamp"
            output_stream: "ITERABLE:ints_plus_one"
          }
        )pb");
    tool::AddVectorSink("ints_plus_one", &graph_config, &output_packets_);
    MP_ASSERT_OK(graph_.Initialize(graph_config));
    MP_ASSERT_OK(graph_.StartRun({}));
  }

  void SendPacketOfInts(Timestamp timestamp, std::vector<int> ints) {
    MP_ASSERT_OK(graph_.AddPacketToInputStream(
        "ints", MakePacket<std::vector<int>>(std::move(ints)).At(timestamp)));
  }

  CalculatorGraph graph_;
  std::vector<Packet> output_packets_;
};

TEST_F(BeginEndLoopCalculatorGraphTest, InputStreamForIterableIsEmpty) {
  MP_ASSERT_OK(graph_.WaitUntilIdle());

  // EndLoopCalc will forward the timestamp bound because there are no packets
  // to process.
  ASSERT_EQ(0, output_packets_.size());

  MP_ASSERT_OK(graph_.CloseAllPacketSources());
  MP_ASSERT_OK(graph_.WaitUntilDone());
}

TEST_F(BeginEndLoopCalculatorGraphTest, SingleEmptyVector) {
  SendPacketOfInts(Timestamp(0), {});
  MP_ASSERT_OK(graph_.WaitUntilIdle());

  // EndLoopCalc will forward the timestamp bound because there are no elements
  // in collection to output.
  EXPECT_TRUE(output_packets_.empty());

  MP_ASSERT_OK(graph_.CloseAllPacketSources());
  MP_ASSERT_OK(graph_.WaitUntilDone());
}

TEST_F(BeginEndLoopCalculatorGraphTest, SingleNonEmptyVector) {
  Timestamp input_timestamp = Timestamp(0);
  SendPacketOfInts(input_timestamp, {0, 1, 2});
  MP_ASSERT_OK(graph_.WaitUntilIdle());

  EXPECT_THAT(output_packets_,
              testing::ElementsAre(
                  PacketOfIntsEq(input_timestamp, std::vector<int>{1, 2, 3})));

  MP_ASSERT_OK(graph_.CloseAllPacketSources());
  MP_ASSERT_OK(graph_.WaitUntilDone());
}

TEST_F(BeginEndLoopCalculatorGraphTest, MultipleVectors) {
  Timestamp input_timestamp0 = Timestamp(0);
  SendPacketOfInts(input_timestamp0, {0, 1});

  Timestamp input_timestamp1 = Timestamp(1);
  SendPacketOfInts(input_timestamp1, {});

  Timestamp input_timestamp2 = Timestamp(2);
  SendPacketOfInts(input_timestamp2, {2, 3});

  MP_ASSERT_OK(graph_.CloseAllPacketSources());
  MP_ASSERT_OK(graph_.WaitUntilDone());

  // At input_timestamp1, EndLoopCalc will forward timestamp bound as there are
  // no elements in vector to process.
  EXPECT_THAT(output_packets_,
              testing::ElementsAre(
                  PacketOfIntsEq(input_timestamp0, std::vector<int>{1, 2}),
                  PacketOfIntsEq(input_timestamp2, std::vector<int>{3, 4})));
}

TEST(BeginEndLoopCalculatorPossibleDataRaceTest,
     EndLoopForIntegersDoesNotRace) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(
      R"pb(
        num_threads: 4
        input_stream: "ints"
        node {
          calculator: "BeginLoopIntegerCalculator"
          input_stream: "ITERABLE:ints"
          output_stream: "ITEM:int"
          output_stream: "BATCH_END:timestamp"
        }
        node {
          calculator: "IncrementCalculator"
          input_stream: "int"
          output_stream: "int_plus_one"
        }
        # BEGIN: Data race possibility
        # EndLoop###Calculator and another calculator using the same input
        # may introduce race due to EndLoop###Calculator possibly consuming
        # packet.
        node {
          calculator: "EndLoopIntegersCalculator"
          input_stream: "ITEM:int_plus_one"
          input_stream: "BATCH_END:timestamp"
          output_stream: "ITERABLE:ints_plus_one"
        }
        node {
          calculator: "IncrementCalculator"
          input_stream: "int_plus_one"
          output_stream: "int_plus_two"
        }
        # END: Data race possibility
        node {
          calculator: "EndLoopIntegersCalculator"
          input_stream: "ITEM:int_plus_two"
          input_stream: "BATCH_END:timestamp"
          output_stream: "ITERABLE:ints_plus_two"
        }
      )pb");
  std::vector<Packet> int_plus_one_packets;
  tool::AddVectorSink("ints_plus_one", &graph_config, &int_plus_one_packets);
  std::vector<Packet> int_original_packets;
  tool::AddVectorSink("ints_plus_two", &graph_config, &int_original_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  for (int i = 0; i < 100; ++i) {
    std::vector<int> ints = {i, i + 1, i + 2};
    Timestamp ts = Timestamp(i);
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "ints", MakePacket<std::vector<int>>(std::move(ints)).At(ts)));
    MP_ASSERT_OK(graph.WaitUntilIdle());
    EXPECT_THAT(int_plus_one_packets,
                testing::ElementsAre(
                    PacketOfIntsEq(ts, std::vector<int>{i + 1, i + 2, i + 3})));
    EXPECT_THAT(int_original_packets,
                testing::ElementsAre(
                    PacketOfIntsEq(ts, std::vector<int>{i + 2, i + 3, i + 4})));

    int_plus_one_packets.clear();
    int_original_packets.clear();
  }

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// Passes non empty vector through or outputs empty vector in case of timestamp
// bound update.
class PassThroughOrEmptyVectorCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->SetProcessTimestampBounds(true);
    cc->Inputs().Index(0).Set<std::vector<int>>();
    cc->Outputs().Index(0).Set<std::vector<int>>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (!cc->Inputs().Index(0).IsEmpty()) {
      cc->Outputs().Index(0).AddPacket(cc->Inputs().Index(0).Value());
    } else {
      cc->Outputs().Index(0).AddPacket(
          MakePacket<std::vector<int>>(std::vector<int>())
              .At(cc->InputTimestamp()));
    }
    return absl::OkStatus();
  }
};

REGISTER_CALCULATOR(PassThroughOrEmptyVectorCalculator);

class BeginEndLoopCalculatorGraphProcessingEmptyPacketsTest
    : public ::testing::Test {
 protected:
  void SetUp() override {
    auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(
        R"pb(
          num_threads: 4
          input_stream: "ints"
          input_stream: "force_ints_to_be_timestamp_bound_update"
          node {
            calculator: "GateCalculator"
            input_stream: "ints"
            input_stream: "DISALLOW:force_ints_to_be_timestamp_bound_update"
            output_stream: "ints_passed_through"
          }
          node {
            calculator: "BeginLoopIntegerCalculator"
            input_stream: "ITERABLE:ints_passed_through"
            output_stream: "ITEM:int"
            output_stream: "BATCH_END:timestamp"
          }
          node {
            calculator: "IncrementCalculator"
            input_stream: "int"
            output_stream: "int_plus_one"
          }
          node {
            calculator: "EndLoopIntegersCalculator"
            input_stream: "ITEM:int_plus_one"
            input_stream: "BATCH_END:timestamp"
            output_stream: "ITERABLE:ints_plus_one"
          }
          node {
            calculator: "PassThroughOrEmptyVectorCalculator"
            input_stream: "ints_plus_one"
            output_stream: "ints_plus_one_passed_through"
          }
        )pb");
    tool::AddVectorSink("ints_plus_one_passed_through", &graph_config,
                        &output_packets_);
    MP_ASSERT_OK(graph_.Initialize(graph_config));
    MP_ASSERT_OK(graph_.StartRun({}));
  }

  void SendPacketOfIntsOrBound(Timestamp timestamp, std::vector<int> ints) {
    // All "ints" packets which are empty are forced to be just timestamp
    // bound updates for begin loop calculator.
    bool force_ints_to_be_timestamp_bound_update = ints.empty();
    MP_ASSERT_OK(graph_.AddPacketToInputStream(
        "force_ints_to_be_timestamp_bound_update",
        MakePacket<bool>(force_ints_to_be_timestamp_bound_update)
            .At(timestamp)));
    MP_ASSERT_OK(graph_.AddPacketToInputStream(
        "ints", MakePacket<std::vector<int>>(std::move(ints)).At(timestamp)));
  }

  CalculatorGraph graph_;
  std::vector<Packet> output_packets_;
};

TEST_F(BeginEndLoopCalculatorGraphProcessingEmptyPacketsTest,
       SingleEmptyVector) {
  SendPacketOfIntsOrBound(Timestamp(0), {});
  MP_ASSERT_OK(graph_.WaitUntilIdle());

  EXPECT_THAT(output_packets_, testing::ElementsAre(PacketOfIntsEq(
                                   Timestamp(0), std::vector<int>{})));

  MP_ASSERT_OK(graph_.CloseAllPacketSources());
  MP_ASSERT_OK(graph_.WaitUntilDone());
}

TEST_F(BeginEndLoopCalculatorGraphProcessingEmptyPacketsTest,
       SingleNonEmptyVector) {
  SendPacketOfIntsOrBound(Timestamp(0), {0, 1, 2});
  MP_ASSERT_OK(graph_.WaitUntilIdle());

  EXPECT_THAT(output_packets_, testing::ElementsAre(PacketOfIntsEq(
                                   Timestamp(0), std::vector<int>{1, 2, 3})));

  MP_ASSERT_OK(graph_.CloseAllPacketSources());
  MP_ASSERT_OK(graph_.WaitUntilDone());
}

TEST_F(BeginEndLoopCalculatorGraphProcessingEmptyPacketsTest, MultipleVectors) {
  SendPacketOfIntsOrBound(Timestamp(0), {});
  // Waiting until idle to guarantee all timestamp bound updates are processed
  // individually. (Timestamp bounds updates occur in the provide config only
  // if input is an empty vector.)
  MP_ASSERT_OK(graph_.WaitUntilIdle());

  SendPacketOfIntsOrBound(Timestamp(1), {0, 1});
  SendPacketOfIntsOrBound(Timestamp(2), {});
  // Waiting until idle to guarantee all timestamp bound updates are processed
  // individually. (Timestamp bounds updates occur in the provide config only
  // if input is an empty vector.)
  MP_ASSERT_OK(graph_.WaitUntilIdle());

  SendPacketOfIntsOrBound(Timestamp(3), {2, 3});
  SendPacketOfIntsOrBound(Timestamp(4), {});
  // Waiting until idle to guarantee all timestamp bound updates are processed
  // individually. (Timestamp bounds updates occur in the provide config only
  // if input is an empty vector.)
  MP_ASSERT_OK(graph_.WaitUntilIdle());

  MP_ASSERT_OK(graph_.CloseAllPacketSources());
  MP_ASSERT_OK(graph_.WaitUntilDone());

  EXPECT_THAT(
      output_packets_,
      testing::ElementsAre(PacketOfIntsEq(Timestamp(0), std::vector<int>{}),
                           PacketOfIntsEq(Timestamp(1), std::vector<int>{1, 2}),
                           PacketOfIntsEq(Timestamp(2), std::vector<int>{}),
                           PacketOfIntsEq(Timestamp(3), std::vector<int>{3, 4}),
                           PacketOfIntsEq(Timestamp(4), std::vector<int>{})));
}

class MultiplierCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Inputs().Index(1).Set<int>();
    cc->Outputs().Index(0).Set<int>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const int& input_int = cc->Inputs().Index(0).Get<int>();
    const int& multiplier_int = cc->Inputs().Index(1).Get<int>();
    auto output_int = absl::make_unique<int>(input_int * multiplier_int);
    cc->Outputs().Index(0).Add(output_int.release(), cc->InputTimestamp());
    return absl::OkStatus();
  }
};

REGISTER_CALCULATOR(MultiplierCalculator);

class BeginEndLoopCalculatorGraphWithClonedInputsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(
        R"pb(
          num_threads: 4
          input_stream: "ints"
          input_stream: "multiplier"
          node {
            calculator: "BeginLoopIntegerCalculator"
            input_stream: "ITERABLE:ints"
            input_stream: "CLONE:multiplier"
            output_stream: "ITEM:int_at_loop"
            output_stream: "CLONE:multiplier_cloned_at_loop"
            output_stream: "BATCH_END:timestamp"
          }
          node {
            calculator: "MultiplierCalculator"
            input_stream: "int_at_loop"
            input_stream: "multiplier_cloned_at_loop"
            output_stream: "multiplied_int_at_loop"
          }
          node {
            calculator: "EndLoopIntegersCalculator"
            input_stream: "ITEM:multiplied_int_at_loop"
            input_stream: "BATCH_END:timestamp"
            output_stream: "ITERABLE:multiplied_ints"
          }
        )pb");
    tool::AddVectorSink("multiplied_ints", &graph_config, &output_packets_);
    MP_ASSERT_OK(graph_.Initialize(graph_config));
    MP_ASSERT_OK(graph_.StartRun({}));
  }

  void SendPackets(Timestamp timestamp, int multiplier, std::vector<int> ints) {
    MP_ASSERT_OK(graph_.AddPacketToInputStream(
        "ints", MakePacket<std::vector<int>>(std::move(ints)).At(timestamp)));
    MP_ASSERT_OK(graph_.AddPacketToInputStream(
        "multiplier", MakePacket<int>(multiplier).At(timestamp)));
  }

  void SendMultiplier(Timestamp timestamp, int multiplier) {
    MP_ASSERT_OK(graph_.AddPacketToInputStream(
        "multiplier", MakePacket<int>(multiplier).At(timestamp)));
  }

  CalculatorGraph graph_;
  std::vector<Packet> output_packets_;
};

TEST_F(BeginEndLoopCalculatorGraphWithClonedInputsTest,
       InputStreamForIterableIsEmpty) {
  Timestamp input_timestamp = Timestamp(42);
  SendMultiplier(input_timestamp, /*multiplier=*/2);
  MP_ASSERT_OK(graph_.WaitUntilIdle());

  // EndLoopCalc will forward the timestamp bound because there are no packets
  // to process.
  ASSERT_EQ(0, output_packets_.size());

  MP_ASSERT_OK(graph_.CloseAllPacketSources());
  MP_ASSERT_OK(graph_.WaitUntilDone());
}

TEST_F(BeginEndLoopCalculatorGraphWithClonedInputsTest, SingleEmptyVector) {
  SendPackets(Timestamp(0), /*multiplier=*/2, /*ints=*/{});
  MP_ASSERT_OK(graph_.WaitUntilIdle());

  // EndLoopCalc will forward the timestamp bound because there are no elements
  // in collection to output.
  EXPECT_TRUE(output_packets_.empty());

  MP_ASSERT_OK(graph_.CloseAllPacketSources());
  MP_ASSERT_OK(graph_.WaitUntilDone());
}

TEST_F(BeginEndLoopCalculatorGraphWithClonedInputsTest, SingleNonEmptyVector) {
  Timestamp input_timestamp = Timestamp(42);
  SendPackets(input_timestamp, /*multiplier=*/2, /*ints=*/{0, 1, 2});
  MP_ASSERT_OK(graph_.WaitUntilIdle());

  EXPECT_THAT(output_packets_,
              testing::ElementsAre(
                  PacketOfIntsEq(input_timestamp, std::vector<int>{0, 2, 4})));

  MP_ASSERT_OK(graph_.CloseAllPacketSources());
  MP_ASSERT_OK(graph_.WaitUntilDone());
}

TEST_F(BeginEndLoopCalculatorGraphWithClonedInputsTest, MultipleVectors) {
  Timestamp input_timestamp0 = Timestamp(42);
  SendPackets(input_timestamp0, /*multiplier=*/2, /*ints=*/{0, 1});

  Timestamp input_timestamp1 = Timestamp(43);
  SendPackets(input_timestamp1, /*multiplier=*/2, /*ints=*/{});

  Timestamp input_timestamp2 = Timestamp(44);
  SendPackets(input_timestamp2, /*multiplier=*/3, /*ints=*/{2, 3});

  MP_ASSERT_OK(graph_.CloseAllPacketSources());
  MP_ASSERT_OK(graph_.WaitUntilDone());

  // At input_timestamp1, EndLoopCalc will forward timestamp bound as there are
  // no elements in vector to process.
  EXPECT_THAT(output_packets_,
              testing::ElementsAre(
                  PacketOfIntsEq(input_timestamp0, std::vector<int>{0, 2}),
                  PacketOfIntsEq(input_timestamp2, std::vector<int>{6, 9})));
}

class TestTensorCpuCopyCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<Tensor>();
    cc->Outputs().Index(0).Set<Tensor>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const Tensor& in_tensor = cc->Inputs().Index(0).Get<Tensor>();
    const Tensor::CpuReadView in_view = in_tensor.GetCpuReadView();
    const void* in_data = in_view.buffer<void>();

    Tensor out_tensor(in_tensor.element_type(), in_tensor.shape());
    auto out_view = out_tensor.GetCpuWriteView();
    void* out_data = out_view.buffer<void>();

    std::memcpy(out_data, in_data, in_tensor.bytes());

    cc->Outputs().Index(0).AddPacket(
        MakePacket<Tensor>(std::move(out_tensor)).At(cc->InputTimestamp()));
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(TestTensorCpuCopyCalculator);

absl::Status InitBeginEndTensorLoopTestGraph(
    CalculatorGraph& graph, std::vector<Packet>& output_packets) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(
      R"pb(
        num_threads: 4
        input_stream: "tensors"
        node {
          calculator: "BeginLoopTensorCalculator"
          input_stream: "ITERABLE:tensors"
          output_stream: "ITEM:tensor"
          output_stream: "BATCH_END:timestamp"
        }
        node {
          calculator: "TestTensorCpuCopyCalculator"
          input_stream: "tensor"
          output_stream: "copied_tensor"
        }
        node {
          calculator: "EndLoopTensorCalculator"
          input_stream: "ITEM:copied_tensor"
          input_stream: "BATCH_END:timestamp"
          output_stream: "ITERABLE:output_tensors"
        }
      )pb");
  tool::AddVectorSink("output_tensors", &graph_config, &output_packets);
  MP_RETURN_IF_ERROR(graph.Initialize(graph_config));
  return graph.StartRun({});
}

TEST(BeginEndTensorLoopCalculatorGraphTest, SingleNonEmptyVector) {
  // Initialize the graph.
  CalculatorGraph graph;
  std::vector<Packet> output_packets;
  MP_ASSERT_OK(InitBeginEndTensorLoopTestGraph(graph, output_packets));

  // Prepare the inputs and run.
  Timestamp input_timestamp = Timestamp(0);
  std::vector<mediapipe::Tensor> tensors;
  for (int i = 0; i < 4; i++) {
    tensors.emplace_back(Tensor::ElementType::kFloat32,
                         Tensor::Shape{4, 3, 2, 1});
    auto write_view = tensors.back().GetCpuWriteView();
    float* data = write_view.buffer<float>();

    // Populate with tensor index in the vector.
    std::fill(data, data + tensors.back().element_size(), i);
  }
  Packet vector_packet =
      MakePacket<std::vector<mediapipe::Tensor>>(std::move(tensors));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "tensors", std::move(vector_packet).At(input_timestamp)));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Verify the output packet.
  EXPECT_EQ(output_packets.size(), 1);
  const std::vector<Tensor>& output_tensors =
      output_packets[0].Get<std::vector<Tensor>>();
  EXPECT_EQ(output_tensors.size(), 4);
  for (int i = 0; i < output_tensors.size(); i++) {
    EXPECT_THAT(output_tensors[i].shape().dims,
                testing::ElementsAre(4, 3, 2, 1));
    const float* data = output_tensors[i].GetCpuReadView().buffer<float>();

    // Expect every element is equal to tensor index.
    EXPECT_THAT(absl::MakeSpan(data, output_tensors[i].element_size()),
                testing::Each(i));
  }

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

}  // namespace
}  // namespace mediapipe
