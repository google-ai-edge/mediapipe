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

#include "mediapipe/framework/tool/sink.h"

#include <memory>
#include <vector>

#include "absl/functional/bind_front.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/validate_type.h"

namespace mediapipe {

namespace {
class CountAndOutputSummarySidePacketInCloseCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->OutputSidePackets().Index(0).Set<int>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    ++count_;
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) final {
    cc->OutputSidePackets().Index(0).Set(
        MakePacket<int>(count_).At(Timestamp::Unset()));
    return absl::OkStatus();
  }

  int count_ = 0;
};
REGISTER_CALCULATOR(CountAndOutputSummarySidePacketInCloseCalculator);

TEST(CallbackFromGeneratorTest, TestAddVectorSink) {
  CalculatorGraphConfig graph_config;
  std::vector<Packet> dumped_data;
  tool::AddVectorSink("input_packets", &graph_config, &dumped_data);
  graph_config.add_input_stream("input_packets");
  CalculatorGraph graph(graph_config);
  MP_ASSERT_OK(graph.StartRun({}));
  for (int i = 0; i < 10; ++i) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "input_packets", MakePacket<int>(i).At(Timestamp(i))));
    MP_ASSERT_OK(graph.WaitUntilIdle());
  }
  MP_ASSERT_OK(graph.CloseInputStream("input_packets"));
  MP_ASSERT_OK(graph.WaitUntilDone());
  ASSERT_EQ(10, dumped_data.size());
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(Timestamp(i), dumped_data[i].Timestamp());
    EXPECT_EQ(i, dumped_data[i].Get<int>());
  }
}

TEST(CalculatorGraph, OutputSummarySidePacketInClose) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input_packets"
        node {
          calculator: "CountAndOutputSummarySidePacketInCloseCalculator"
          input_stream: "input_packets"
          output_side_packet: "num_of_packets"
        }
      )pb");

  Packet summary_packet;
  tool::AddSidePacketSink("num_of_packets", &config, &summary_packet);
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));

  // Run the graph twice.
  int max_count = 100;
  for (int run = 0; run < 1; ++run) {
    MP_ASSERT_OK(graph.StartRun({}));
    for (int i = 0; i < max_count; ++i) {
      MP_ASSERT_OK(graph.AddPacketToInputStream(
          "input_packets", MakePacket<int>(i).At(Timestamp(i))));
    }
    MP_ASSERT_OK(graph.CloseInputStream("input_packets"));
    MP_ASSERT_OK(graph.WaitUntilDone());
    EXPECT_EQ(max_count, summary_packet.Get<int>());
    EXPECT_EQ(Timestamp::PostStream(), summary_packet.Timestamp());
  }
}

TEST(CallbackTest, TestAddMultiStreamCallback) {
  CalculatorGraphConfig graph_config;
  graph_config.add_input_stream("foo");
  graph_config.add_input_stream("bar");

  std::vector<int> sums;

  std::pair<std::string, Packet> cb_packet;
  tool::AddMultiStreamCallback(
      {"foo", "bar"},
      [&sums](const std::vector<Packet>& packets) {
        Packet foo_p = packets[0];
        Packet bar_p = packets[1];
        int foo = foo_p.IsEmpty() ? 0 : foo_p.Get<int>();
        int bar = bar_p.IsEmpty() ? 0 : bar_p.Get<int>();
        sums.push_back(foo + bar);
      },
      &graph_config, &cb_packet);

  CalculatorGraph graph(graph_config);
  MP_ASSERT_OK(graph.StartRun({cb_packet}));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "foo", MakePacket<int>(10).At(Timestamp(1))));
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("bar", MakePacket<int>(5).At(Timestamp(1))));

  MP_ASSERT_OK(
      graph.AddPacketToInputStream("foo", MakePacket<int>(7).At(Timestamp(2))));
  // no bar input at 2

  MP_ASSERT_OK(
      graph.AddPacketToInputStream("foo", MakePacket<int>(4).At(Timestamp(3))));
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("bar", MakePacket<int>(5).At(Timestamp(3))));

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  EXPECT_THAT(sums, testing::ElementsAre(15, 7, 9));
}

class TimestampBoundTestCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Outputs().Index(0).Set<int>();
    cc->Outputs().Index(1).Set<int>();
    return absl::OkStatus();
  }
  absl::Status Open(CalculatorContext* cc) final { return absl::OkStatus(); }
  absl::Status Process(CalculatorContext* cc) final {
    if (count_ % 5 == 0) {
      cc->Outputs().Index(0).SetNextTimestampBound(Timestamp(count_ + 1));
      cc->Outputs().Index(1).SetNextTimestampBound(Timestamp(count_ + 1));
    }
    ++count_;
    if (count_ == 13) {
      return tool::StatusStop();
    }
    return absl::OkStatus();
  }

 private:
  int count_ = 1;
};
REGISTER_CALCULATOR(TimestampBoundTestCalculator);

#if 0  // test is flaky, try it with --runs_per_test=200
TEST(CallbackTest, TestAddMultiStreamCallbackWithTimestampNotification) {
  std::string config_str = R"(
            node {
              calculator: "TimestampBoundTestCalculator"
              output_stream: "foo"
              output_stream: "bar"
            }
          )";
  CalculatorGraphConfig graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(config_str);

  std::vector<int> sums;

  std::map<std::string, Packet> side_packets;
  tool::AddMultiStreamCallback(
      {"foo", "bar"},
      [&sums](const std::vector<Packet>& packets) {
        Packet foo_p = packets[0];
        Packet bar_p = packets[1];
        ASSERT_TRUE(foo_p.IsEmpty() && bar_p.IsEmpty());
        int foo = foo_p.Timestamp().Value();
        int bar = bar_p.Timestamp().Value();
        sums.push_back(foo + bar);
      },
      &graph_config, &side_packets, true);

  CalculatorGraph graph(graph_config);
  MP_ASSERT_OK(graph.StartRun(side_packets));
  MP_ASSERT_OK(graph.WaitUntilDone());

  EXPECT_THAT(sums, testing::ElementsAre(10, 20));
}
#endif

}  // namespace
}  // namespace mediapipe
