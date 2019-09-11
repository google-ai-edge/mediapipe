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
//
// Tests CalculatorRunner.

#include "mediapipe/framework/calculator_runner.h"

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_base.h"
#include "mediapipe/framework/calculator_registry.h"
#include "mediapipe/framework/input_stream.h"
#include "mediapipe/framework/output_stream.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace {

// Inputs: 2 streams with ints. Headers are strings.
// Input side packets: 1.
// Outputs: 3 streams with ints. #0 and #1 will contain the negated values from
// corresponding input streams, #2 will contain replicas of the input side
// packet
// at InputTimestamp. The headers are strings.
class CalculatorRunnerTestCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Inputs().Index(1).Set<int>();
    cc->Outputs().Index(0).Set<int>();
    cc->Outputs().Index(1).Set<int>();
    cc->Outputs().Index(2).SetSameAs(&cc->InputSidePackets().Index(0));
    cc->InputSidePackets().Index(0).SetAny();
    cc->OutputSidePackets()
        .Tag("SIDE_OUTPUT")
        .SetSameAs(&cc->InputSidePackets().Index(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    std::string input_header_string =
        absl::StrCat(cc->Inputs().Index(0).Header().Get<std::string>(),
                     cc->Inputs().Index(1).Header().Get<std::string>());
    for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
      // Set the header to the concatenation of the input headers and
      // the index of the output stream.
      cc->Outputs().Index(i).SetHeader(
          Adopt(new std::string(absl::StrCat(input_header_string, i))));
    }
    cc->OutputSidePackets()
        .Tag("SIDE_OUTPUT")
        .Set(cc->InputSidePackets().Index(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    for (int index = 0; index < 2; ++index) {
      cc->Outputs().Index(index).Add(
          new int(-cc->Inputs().Index(index).Get<int>()), cc->InputTimestamp());
    }
    cc->Outputs().Index(2).AddPacket(
        cc->InputSidePackets().Index(0).At(cc->InputTimestamp()));
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(CalculatorRunnerTestCalculator);

// Inputs: Any number of streams of integer, with any tags.
// Outputs: For each tag name (possibly including the empty tag), outputs a
//          a single stream with the sum of the integers belonging to streams
//          with the same tag name (and any index).
class CalculatorRunnerMultiTagTestCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    for (const std::string& tag : cc->Inputs().GetTags()) {
      for (CollectionItemId item_id = cc->Inputs().BeginId(tag);
           item_id < cc->Inputs().EndId(tag); ++item_id) {
        cc->Inputs().Get(item_id).Set<int>();
      }
      cc->Outputs().Get(tag, 0).Set<int>();
    }
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    for (const std::string& tag : cc->Inputs().GetTags()) {
      auto sum = absl::make_unique<int>(0);
      for (CollectionItemId item_id = cc->Inputs().BeginId(tag);
           item_id < cc->Inputs().EndId(tag); ++item_id) {
        if (!cc->Inputs().Get(item_id).IsEmpty()) {
          *sum += cc->Inputs().Get(item_id).Get<int>();
        }
      }
      cc->Outputs().Get(tag, 0).Add(sum.release(), cc->InputTimestamp());
    }
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(CalculatorRunnerMultiTagTestCalculator);

TEST(CalculatorRunner, RunsCalculator) {
  CalculatorRunner runner(R"(
      calculator: "CalculatorRunnerTestCalculator"
      input_stream: "input_0"
      input_stream: "input_1"
      output_stream: "output_0"
      output_stream: "output_1"
      output_stream: "output_2"
      input_side_packet: "input_side_packet_0"
      output_side_packet: "SIDE_OUTPUT:output_side_packet_0"
      options {
      }
  )");

  // Run CalculatorRunner::Run() several times, with different inputs. This
  // tests that a CalculatorRunner instance can be reused.
  for (int iter = 0; iter < 3; ++iter) {
    LOG(INFO) << "iter: " << iter;
    const int length = iter;
    // Generate the inputs at timestamps 0 ... length-1, at timestamp t having
    // values t and t*2 for the two streams, respectively.
    const std::string kHeaderPrefix = "header";
    for (int index = 0; index < 2; ++index) {
      runner.MutableInputs()->Index(index).packets.clear();
      for (int t = 0; t < length; ++t) {
        runner.MutableInputs()->Index(index).packets.push_back(
            Adopt(new int(t * (index + 1))).At(Timestamp(t)));
      }
      // Set the header to the concatenation of kHeaderPrefix and the index of
      // the input stream.
      runner.MutableInputs()->Index(index).header =
          Adopt(new std::string(absl::StrCat(kHeaderPrefix, index)));
    }
    const int input_side_packet_content = 10 + iter;
    runner.MutableSidePackets()->Index(0) =
        Adopt(new int(input_side_packet_content));
    MP_ASSERT_OK(runner.Run());
    EXPECT_EQ(input_side_packet_content,
              runner.OutputSidePackets().Tag("SIDE_OUTPUT").Get<int>());
    const auto& outputs = runner.Outputs();
    ASSERT_EQ(3, outputs.NumEntries());

    // Check the output headers and the number of Packets.
    for (int index = 0; index < outputs.NumEntries(); ++index) {
      // The header should be the concatenation of the input headers
      // and the index of the output stream.
      EXPECT_EQ(absl::StrCat(kHeaderPrefix, 0, kHeaderPrefix, 1, index),
                outputs.Index(index).header.Get<std::string>());
      // Check the packets.
      const std::vector<Packet>& packets = outputs.Index(index).packets;
      EXPECT_EQ(length, packets.size());
      for (int t = 0; t < length; ++t) {
        EXPECT_EQ(Timestamp(t), packets[t].Timestamp());
        // The first two output streams are negations of the inputs, the last
        // contains copies of the input side packet.
        if (index < 2) {
          EXPECT_EQ(-t * (index + 1), packets[t].Get<int>());
        } else {
          EXPECT_EQ(input_side_packet_content, packets[t].Get<int>());
        }
      }
    }
  }
}

TEST(CalculatorRunner, MultiTagTestCalculatorOk) {
  CalculatorRunner runner(R"(
      calculator: "CalculatorRunnerMultiTagTestCalculator"
      input_stream: "A:0:full_0"
      input_stream: "A:1:full_1"
      input_stream: "A:2:full_2"
      input_stream: "B:no_index_0"
      input_stream: "no_tag_or_index_0"
      input_stream: "no_tag_or_index_1"
      output_stream: "A:output_a"
      output_stream: "B:output_b"
      output_stream: "output_c"
  )");

  for (int ts = 0; ts < 5; ++ts) {
    for (int i = 0; i < 3; ++i) {
      runner.MutableInputs()->Get("A", i).packets.push_back(
          Adopt(new int(10 * ts + i)).At(Timestamp(ts)));
    }
    runner.MutableInputs()->Get("B", 0).packets.push_back(
        Adopt(new int(100)).At(Timestamp(ts)));
    runner.MutableInputs()
        ->Get("", ts % 2)
        .packets.push_back(Adopt(new int(ts)).At(Timestamp(ts)));
  }
  MP_ASSERT_OK(runner.Run());

  const auto& outputs = runner.Outputs();
  ASSERT_EQ(3, outputs.NumEntries());
  for (int ts = 0; ts < 5; ++ts) {
    const std::vector<Packet>& a_packets = outputs.Tag("A").packets;
    const std::vector<Packet>& b_packets = outputs.Tag("B").packets;
    const std::vector<Packet>& c_packets = outputs.Tag("").packets;
    EXPECT_EQ(Timestamp(ts), a_packets[ts].Timestamp());
    EXPECT_EQ(Timestamp(ts), b_packets[ts].Timestamp());
    EXPECT_EQ(Timestamp(ts), c_packets[ts].Timestamp());

    EXPECT_EQ(10 * 3 * ts + 3, a_packets[ts].Get<int>());
    EXPECT_EQ(100, b_packets[ts].Get<int>());
    EXPECT_EQ(ts, c_packets[ts].Get<int>());
  }
}

TEST(CalculatorRunner, MultiTagTestInvalidStreamTagCrashes) {
  const std::string graph_config = R"(
      calculator: "CalculatorRunnerMultiTagTestCalculator"
      input_stream: "A:0:a_0"
      input_stream: "A:a_1"
      input_stream: "A:2:a_2"
      output_stream: "A:output_a"
  )";
  EXPECT_DEATH(CalculatorRunner runner(graph_config),
               ".*tag \"A\" index 0 already had a name "
               "\"a_0\" but is being reassigned a name \"a_1\"");
}

}  // namespace
}  // namespace mediapipe
