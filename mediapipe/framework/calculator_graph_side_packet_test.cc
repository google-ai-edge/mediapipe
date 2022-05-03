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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

namespace {

// Takes an input stream packet and passes it (with timestamp removed) as an
// output side packet.
class OutputSidePacketInProcessCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->OutputSidePackets().Index(0).SetSameAs(&cc->Inputs().Index(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    cc->OutputSidePackets().Index(0).Set(
        cc->Inputs().Index(0).Value().At(Timestamp::Unset()));
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(OutputSidePacketInProcessCalculator);

// Takes an input side packet and passes it as an output side packet.
class OutputSidePacketInOpenCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->InputSidePackets().Index(0).SetAny();
    cc->OutputSidePackets().Index(0).SetSameAs(
        &cc->InputSidePackets().Index(0));
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    cc->OutputSidePackets().Index(0).Set(cc->InputSidePackets().Index(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final { return absl::OkStatus(); }
};
REGISTER_CALCULATOR(OutputSidePacketInOpenCalculator);

// Takes an input stream packet and counts the number of the packets it
// receives. Outputs the total number of packets as a side packet in Close.
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
    absl::SleepFor(absl::Milliseconds(300));  // For GetOutputSidePacket test.
    cc->OutputSidePackets().Index(0).Set(
        MakePacket<int>(count_).At(Timestamp::Unset()));
    return absl::OkStatus();
  }

  int count_ = 0;
};
REGISTER_CALCULATOR(CountAndOutputSummarySidePacketInCloseCalculator);

// Takes an input stream packet and passes it (with timestamp intact) as an
// output side packet. This triggers an error in the graph.
class OutputSidePacketWithTimestampCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->OutputSidePackets().Index(0).SetSameAs(&cc->Inputs().Index(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    cc->OutputSidePackets().Index(0).Set(cc->Inputs().Index(0).Value());
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(OutputSidePacketWithTimestampCalculator);

// Generates an output side packet containing the integer 1.
class IntegerOutputSidePacketCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->OutputSidePackets().Index(0).Set<int>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    cc->OutputSidePackets().Index(0).Set(MakePacket<int>(1));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    LOG(FATAL) << "Not reached.";
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(IntegerOutputSidePacketCalculator);

// Generates an output side packet containing the sum of the two integer input
// side packets.
class SidePacketAdderCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->InputSidePackets().Index(0).Set<int>();
    cc->InputSidePackets().Index(1).Set<int>();
    cc->OutputSidePackets().Index(0).Set<int>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    cc->OutputSidePackets().Index(0).Set(
        MakePacket<int>(cc->InputSidePackets().Index(1).Get<int>() +
                        cc->InputSidePackets().Index(0).Get<int>()));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    LOG(FATAL) << "Not reached.";
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(SidePacketAdderCalculator);

// Produces an output packet with the PostStream timestamp containing the
// input side packet.
class SidePacketToStreamPacketCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->InputSidePackets().Index(0).SetAny();
    cc->Outputs().Index(0).SetSameAs(&cc->InputSidePackets().Index(0));
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    cc->Outputs().Index(0).AddPacket(
        cc->InputSidePackets().Index(0).At(Timestamp::PostStream()));
    cc->Outputs().Index(0).Close();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    return mediapipe::tool::StatusStop();
  }
};
REGISTER_CALCULATOR(SidePacketToStreamPacketCalculator);

// Packet generator for an arbitrary unit64 packet.
class Uint64PacketGenerator : public PacketGenerator {
 public:
  static absl::Status FillExpectations(
      const PacketGeneratorOptions& extendable_options,
      PacketTypeSet* input_side_packets, PacketTypeSet* output_side_packets) {
    output_side_packets->Index(0).Set<uint64>();
    return absl::OkStatus();
  }

  static absl::Status Generate(const PacketGeneratorOptions& extendable_options,
                               const PacketSet& input_side_packets,
                               PacketSet* output_side_packets) {
    output_side_packets->Index(0) = Adopt(new uint64(15LL << 32 | 5));
    return absl::OkStatus();
  }
};
REGISTER_PACKET_GENERATOR(Uint64PacketGenerator);

TEST(CalculatorGraph, OutputSidePacketInProcess) {
  const int64 offset = 100;
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "offset"
        node {
          calculator: "OutputSidePacketInProcessCalculator"
          input_stream: "offset"
          output_side_packet: "offset"
        }
        node {
          calculator: "SidePacketToStreamPacketCalculator"
          output_stream: "output"
          input_side_packet: "offset"
        }
      )pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  std::vector<Packet> output_packets;
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "output", [&output_packets](const Packet& packet) {
        output_packets.push_back(packet);
        return absl::OkStatus();
      }));

  // Run the graph twice.
  for (int run = 0; run < 2; ++run) {
    output_packets.clear();
    MP_ASSERT_OK(graph.StartRun({}));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "offset", MakePacket<TimestampDiff>(offset).At(Timestamp(0))));
    MP_ASSERT_OK(graph.CloseInputStream("offset"));
    MP_ASSERT_OK(graph.WaitUntilDone());
    ASSERT_EQ(1, output_packets.size());
    EXPECT_EQ(offset, output_packets[0].Get<TimestampDiff>().Value());
  }
}

// A PacketGenerator that simply passes its input Packets through
// unchanged.  The inputs may be specified by tag or index.  The outputs
// must match the inputs exactly.  Any options may be specified and will
// also be ignored.
class PassThroughGenerator : public PacketGenerator {
 public:
  static absl::Status FillExpectations(
      const PacketGeneratorOptions& extendable_options, PacketTypeSet* inputs,
      PacketTypeSet* outputs) {
    if (!inputs->TagMap()->SameAs(*outputs->TagMap())) {
      return absl::InvalidArgumentError(
          "Input and outputs to PassThroughGenerator must use the same tags "
          "and indexes.");
    }
    for (CollectionItemId id = inputs->BeginId(); id < inputs->EndId(); ++id) {
      inputs->Get(id).SetAny();
      outputs->Get(id).SetSameAs(&inputs->Get(id));
    }
    return absl::OkStatus();
  }

  static absl::Status Generate(const PacketGeneratorOptions& extendable_options,
                               const PacketSet& input_side_packets,
                               PacketSet* output_side_packets) {
    for (CollectionItemId id = input_side_packets.BeginId();
         id < input_side_packets.EndId(); ++id) {
      output_side_packets->Get(id) = input_side_packets.Get(id);
    }
    return absl::OkStatus();
  }
};
REGISTER_PACKET_GENERATOR(PassThroughGenerator);

TEST(CalculatorGraph, SharePacketGeneratorGraph) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: 'CountingSourceCalculator'
          output_stream: 'count1'
          input_side_packet: 'MAX_COUNT:max_count1'
        }
        node {
          calculator: 'CountingSourceCalculator'
          output_stream: 'count2'
          input_side_packet: 'MAX_COUNT:max_count2'
        }
        node {
          calculator: 'CountingSourceCalculator'
          output_stream: 'count3'
          input_side_packet: 'MAX_COUNT:max_count3'
        }
        node {
          calculator: 'CountingSourceCalculator'
          output_stream: 'count4'
          input_side_packet: 'MAX_COUNT:max_count4'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_side_packet: 'MAX_COUNT:max_count5'
          output_side_packet: 'MAX_COUNT:max_count6'
        }
        node {
          calculator: 'CountingSourceCalculator'
          output_stream: 'count5'
          input_side_packet: 'MAX_COUNT:max_count6'
        }
        packet_generator {
          packet_generator: 'PassThroughGenerator'
          input_side_packet: 'max_count1'
          output_side_packet: 'max_count2'
        }
        packet_generator {
          packet_generator: 'PassThroughGenerator'
          input_side_packet: 'max_count4'
          output_side_packet: 'max_count5'
        }
      )pb");

  // At this point config is a standard config which specifies both
  // calculators and packet_factories/packet_genators.  The following
  // code is an example of reusing side packets across a number of
  // CalculatorGraphs.  It is particularly informative to note how each
  // side packet is created.
  //
  // max_count1 is set for all graphs by a PacketFactory in the config.
  // The side packet is created by generator_graph.InitializeGraph().
  //
  // max_count2 is set for all graphs by a PacketGenerator in the config.
  // The side packet is created by generator_graph.InitializeGraph()
  // because max_count1 is available at that time.
  //
  // max_count3 is set for all graphs by directly being specified as an
  // argument to generator_graph.InitializeGraph().
  //
  // max_count4 is set per graph because it is directly specified as an
  // argument to generator_graph.ProcessGraph().
  //
  // max_count5 is set per graph by a PacketGenerator which is run when
  // generator_graph.ProcessGraph() is run (because max_count4 isn't
  // available until then).

  // Before anything else, split the graph config into two parts, one
  // with the PacketFactory and PacketGenerator config and the other
  // with the Calculator config.
  CalculatorGraphConfig calculator_config = config;
  calculator_config.clear_packet_factory();
  calculator_config.clear_packet_generator();
  CalculatorGraphConfig generator_config = config;
  generator_config.clear_node();

  // Next, create a ValidatedGraphConfig for both configs.
  ValidatedGraphConfig validated_calculator_config;
  MP_ASSERT_OK(validated_calculator_config.Initialize(calculator_config));
  ValidatedGraphConfig validated_generator_config;
  MP_ASSERT_OK(validated_generator_config.Initialize(generator_config));

  // Create a PacketGeneratorGraph.  Side packets max_count1, max_count2,
  // and max_count3 are created upon initialization.
  // Note that validated_generator_config must outlive generator_graph.
  PacketGeneratorGraph generator_graph;
  MP_ASSERT_OK(
      generator_graph.Initialize(&validated_generator_config, nullptr,
                                 {{"max_count1", MakePacket<int>(10)},
                                  {"max_count3", MakePacket<int>(20)}}));
  ASSERT_THAT(generator_graph.BasePackets(),
              testing::ElementsAre(testing::Key("max_count1"),
                                   testing::Key("max_count2"),
                                   testing::Key("max_count3")));

  // Create a bunch of graphs.
  std::vector<std::unique_ptr<CalculatorGraph>> graphs;
  for (int i = 0; i < 100; ++i) {
    graphs.emplace_back(absl::make_unique<CalculatorGraph>());
    // Do not pass extra side packets here.
    // Note that validated_calculator_config must outlive the graph.
    MP_ASSERT_OK(graphs.back()->Initialize(calculator_config, {}));
  }
  // Run a bunch of graphs, reusing side packets max_count1, max_count2,
  // and max_count3.  The side packet max_count4 is added per run,
  // and triggers the execution of a packet generator which generates
  // max_count5.
  for (int i = 0; i < 100; ++i) {
    std::map<std::string, Packet> all_side_packets;
    // Creates max_count4 and max_count5.
    MP_ASSERT_OK(generator_graph.RunGraphSetup(
        {{"max_count4", MakePacket<int>(30 + i)}}, &all_side_packets));
    ASSERT_THAT(all_side_packets,
                testing::ElementsAre(
                    testing::Key("max_count1"), testing::Key("max_count2"),
                    testing::Key("max_count3"), testing::Key("max_count4"),
                    testing::Key("max_count5")));
    // Pass all the side packets prepared by generator_graph here.
    MP_ASSERT_OK(graphs[i]->Run(all_side_packets));
    // TODO Verify the actual output.
  }

  // Destroy all the graphs.
  graphs.clear();
}

TEST(CalculatorGraph, OutputSidePacketAlreadySet) {
  const int64 offset = 100;
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "offset"
        node {
          calculator: "OutputSidePacketInProcessCalculator"
          input_stream: "offset"
          output_side_packet: "offset"
        }
      )pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  // Send two input packets to cause OutputSidePacketInProcessCalculator to
  // set the output side packet twice.
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "offset", MakePacket<TimestampDiff>(offset).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "offset", MakePacket<TimestampDiff>(offset).At(Timestamp(1))));
  MP_ASSERT_OK(graph.CloseInputStream("offset"));

  absl::Status status = graph.WaitUntilDone();
  EXPECT_EQ(status.code(), absl::StatusCode::kAlreadyExists);
  EXPECT_THAT(status.message(), testing::HasSubstr("was already set."));
}

TEST(CalculatorGraph, OutputSidePacketWithTimestamp) {
  const int64 offset = 100;
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "offset"
        node {
          calculator: "OutputSidePacketWithTimestampCalculator"
          input_stream: "offset"
          output_side_packet: "offset"
        }
      )pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  // The OutputSidePacketWithTimestampCalculator neglects to clear the
  // timestamp in the input packet when it copies the input packet to the
  // output side packet. The timestamp value should appear in the error
  // message.
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "offset", MakePacket<TimestampDiff>(offset).At(Timestamp(237))));
  MP_ASSERT_OK(graph.CloseInputStream("offset"));
  absl::Status status = graph.WaitUntilDone();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::HasSubstr("has a timestamp 237."));
}

TEST(CalculatorGraph, OutputSidePacketConsumedBySourceNode) {
  const int max_count = 10;
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "max_count"
        node {
          calculator: "OutputSidePacketInProcessCalculator"
          input_stream: "max_count"
          output_side_packet: "max_count"
        }
        node {
          calculator: "CountingSourceCalculator"
          output_stream: "count"
          input_side_packet: "MAX_COUNT:max_count"
        }
        node {
          calculator: "PassThroughCalculator"
          input_stream: "count"
          output_stream: "output"
        }
      )pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  std::vector<Packet> output_packets;
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "output", [&output_packets](const Packet& packet) {
        output_packets.push_back(packet);
        return absl::OkStatus();
      }));
  MP_ASSERT_OK(graph.StartRun({}));
  // Wait until the graph is idle so that
  // Scheduler::TryToScheduleNextSourceLayer() gets called.
  // Scheduler::TryToScheduleNextSourceLayer() should not activate source
  // nodes that haven't been opened. We can't call graph.WaitUntilIdle()
  // because the graph has a source node.
  absl::SleepFor(absl::Milliseconds(10));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "max_count", MakePacket<int>(max_count).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseInputStream("max_count"));
  MP_ASSERT_OK(graph.WaitUntilDone());
  ASSERT_EQ(max_count, output_packets.size());
  for (int i = 0; i < output_packets.size(); ++i) {
    EXPECT_EQ(i, output_packets[i].Get<int>());
    EXPECT_EQ(Timestamp(i), output_packets[i].Timestamp());
  }
}

// Returns the first packet of the input stream.
class FirstPacketFilterCalculator : public CalculatorBase {
 public:
  FirstPacketFilterCalculator() {}
  ~FirstPacketFilterCalculator() override {}

  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (!seen_first_packet_) {
      cc->Outputs().Index(0).AddPacket(cc->Inputs().Index(0).Value());
      cc->Outputs().Index(0).Close();
      seen_first_packet_ = true;
    }
    return absl::OkStatus();
  }

 private:
  bool seen_first_packet_ = false;
};
REGISTER_CALCULATOR(FirstPacketFilterCalculator);

TEST(CalculatorGraph, SourceLayerInversion) {
  // There are three CountingSourceCalculators, indexed 0, 1, and 2. Each of
  // them outputs 10 packets.
  //
  // CountingSourceCalculator 0 should output 0, 1, 2, 3, ..., 9.
  // CountingSourceCalculator 1 should output 100, 101, 102, 103, ..., 109.
  // CountingSourceCalculator 2 should output 0, 100, 200, 300, ..., 900.
  // However, there is a source layer inversion.
  // CountingSourceCalculator 0 is in source layer 0.
  // CountingSourceCalculator 1 is in source layer 1.
  // CountingSourceCalculator 2 is in source layer 0, but consumes an output
  // side packet generated by a downstream calculator of
  // CountingSourceCalculator 1.
  //
  // This graph will deadlock when CountingSourceCalculator 0 runs to
  // completion and CountingSourceCalculator 1 cannot be activated because
  // CountingSourceCalculator 2 cannot be opened.

  const int max_count = 10;
  const int initial_value1 = 100;
  // Set num_threads to 1 to force sequential execution for deterministic
  // outputs.
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        num_threads: 1
        node {
          calculator: "CountingSourceCalculator"
          output_stream: "count0"
          input_side_packet: "MAX_COUNT:max_count"
          source_layer: 0
        }

        node {
          calculator: "CountingSourceCalculator"
          output_stream: "count1"
          input_side_packet: "MAX_COUNT:max_count"
          input_side_packet: "INITIAL_VALUE:initial_value1"
          source_layer: 1
        }
        node {
          calculator: "FirstPacketFilterCalculator"
          input_stream: "count1"
          output_stream: "first_count1"
        }
        node {
          calculator: "OutputSidePacketInProcessCalculator"
          input_stream: "first_count1"
          output_side_packet: "increment2"
        }

        node {
          calculator: "CountingSourceCalculator"
          output_stream: "count2"
          input_side_packet: "MAX_COUNT:max_count"
          input_side_packet: "INCREMENT:increment2"
          source_layer: 0
        }
      )pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(
      config, {{"max_count", MakePacket<int>(max_count)},
               {"initial_value1", MakePacket<int>(initial_value1)}}));
  absl::Status status = graph.Run();
  EXPECT_EQ(status.code(), absl::StatusCode::kUnknown);
  EXPECT_THAT(status.message(), testing::HasSubstr("deadlock"));
}

// Tests a graph of packet-generator-like calculators, which have no input
// streams and no output streams.
TEST(CalculatorGraph, PacketGeneratorLikeCalculators) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "IntegerOutputSidePacketCalculator"
          output_side_packet: "one"
        }
        node {
          calculator: "IntegerOutputSidePacketCalculator"
          output_side_packet: "another_one"
        }
        node {
          calculator: "SidePacketAdderCalculator"
          input_side_packet: "one"
          input_side_packet: "another_one"
          output_side_packet: "two"
        }
        node {
          calculator: "IntegerOutputSidePacketCalculator"
          output_side_packet: "yet_another_one"
        }
        node {
          calculator: "SidePacketAdderCalculator"
          input_side_packet: "two"
          input_side_packet: "yet_another_one"
          output_side_packet: "three"
        }
        node {
          calculator: "SidePacketToStreamPacketCalculator"
          input_side_packet: "three"
          output_stream: "output"
        }
      )pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  std::vector<Packet> output_packets;
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "output", [&output_packets](const Packet& packet) {
        output_packets.push_back(packet);
        return absl::OkStatus();
      }));
  MP_ASSERT_OK(graph.Run());
  ASSERT_EQ(1, output_packets.size());
  EXPECT_EQ(3, output_packets[0].Get<int>());
  EXPECT_EQ(Timestamp::PostStream(), output_packets[0].Timestamp());
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
        node {
          calculator: "SidePacketToStreamPacketCalculator"
          input_side_packet: "num_of_packets"
          output_stream: "output"
        }
      )pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  std::vector<Packet> output_packets;
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "output", [&output_packets](const Packet& packet) {
        output_packets.push_back(packet);
        return absl::OkStatus();
      }));

  // Run the graph twice.
  int max_count = 100;
  for (int run = 0; run < 1; ++run) {
    output_packets.clear();
    MP_ASSERT_OK(graph.StartRun({}));
    for (int i = 0; i < max_count; ++i) {
      MP_ASSERT_OK(graph.AddPacketToInputStream(
          "input_packets", MakePacket<int>(i).At(Timestamp(i))));
    }
    MP_ASSERT_OK(graph.CloseInputStream("input_packets"));
    MP_ASSERT_OK(graph.WaitUntilDone());
    ASSERT_EQ(1, output_packets.size());
    EXPECT_EQ(max_count, output_packets[0].Get<int>());
    EXPECT_EQ(Timestamp::PostStream(), output_packets[0].Timestamp());
  }
}

TEST(CalculatorGraph, GetOutputSidePacket) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input_packets"
        node {
          calculator: "CountAndOutputSummarySidePacketInCloseCalculator"
          input_stream: "input_packets"
          output_side_packet: "num_of_packets"
        }
        packet_generator {
          packet_generator: "Uint64PacketGenerator"
          output_side_packet: "output_uint64"
        }
        packet_generator {
          packet_generator: "IntSplitterPacketGenerator"
          input_side_packet: "input_uint64"
          output_side_packet: "output_uint32_pair"
        }
      )pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  // Check a packet generated by the PacketGenerator, which is available after
  // graph initialization, can be fetched before graph starts.
  absl::StatusOr<Packet> status_or_packet =
      graph.GetOutputSidePacket("output_uint64");
  MP_ASSERT_OK(status_or_packet);
  EXPECT_EQ(Timestamp::Unset(), status_or_packet.value().Timestamp());
  // IntSplitterPacketGenerator is missing its input side packet and we
  // won't be able to get its output side packet now.
  status_or_packet = graph.GetOutputSidePacket("output_uint32_pair");
  EXPECT_EQ(absl::StatusCode::kUnavailable, status_or_packet.status().code());
  // Run the graph twice.
  int max_count = 100;
  std::map<std::string, Packet> extra_side_packets;
  extra_side_packets.insert({"input_uint64", MakePacket<uint64>(1123)});
  for (int run = 0; run < 1; ++run) {
    MP_ASSERT_OK(graph.StartRun(extra_side_packets));
    status_or_packet = graph.GetOutputSidePacket("output_uint32_pair");
    MP_ASSERT_OK(status_or_packet);
    EXPECT_EQ(Timestamp::Unset(), status_or_packet.value().Timestamp());
    for (int i = 0; i < max_count; ++i) {
      MP_ASSERT_OK(graph.AddPacketToInputStream(
          "input_packets", MakePacket<int>(i).At(Timestamp(i))));
    }
    MP_ASSERT_OK(graph.CloseInputStream("input_packets"));

    // Should return NOT_FOUND for invalid side packets.
    status_or_packet = graph.GetOutputSidePacket("unknown");
    EXPECT_FALSE(status_or_packet.ok());
    EXPECT_EQ(absl::StatusCode::kNotFound, status_or_packet.status().code());
    // Should return the packet after the graph becomes idle.
    MP_ASSERT_OK(graph.WaitUntilIdle());
    status_or_packet = graph.GetOutputSidePacket("num_of_packets");
    MP_ASSERT_OK(status_or_packet);
    EXPECT_EQ(max_count, status_or_packet.value().Get<int>());
    EXPECT_EQ(Timestamp::Unset(), status_or_packet.value().Timestamp());
    // Should stil return a base even before graph is done.
    status_or_packet = graph.GetOutputSidePacket("output_uint64");
    MP_ASSERT_OK(status_or_packet);
    EXPECT_EQ(Timestamp::Unset(), status_or_packet.value().Timestamp());

    MP_ASSERT_OK(graph.WaitUntilDone());

    // Check packets are available after graph is done.
    status_or_packet = graph.GetOutputSidePacket("num_of_packets");
    MP_ASSERT_OK(status_or_packet);
    EXPECT_EQ(max_count, status_or_packet.value().Get<int>());
    EXPECT_EQ(Timestamp::Unset(), status_or_packet.value().Timestamp());
    // Should still return a base packet after graph is done.
    status_or_packet = graph.GetOutputSidePacket("output_uint64");
    MP_ASSERT_OK(status_or_packet);
    EXPECT_EQ(Timestamp::Unset(), status_or_packet.value().Timestamp());
    // Should still return a non-base packet after graph is done.
    status_or_packet = graph.GetOutputSidePacket("output_uint32_pair");
    MP_ASSERT_OK(status_or_packet);
    EXPECT_EQ(Timestamp::Unset(), status_or_packet.value().Timestamp());
  }
}

typedef std::string HugeModel;

// Generates an output-side-packet once for each calculator-graph.
class OutputSidePacketCachedCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->OutputSidePackets().Index(0).Set<HugeModel>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    cc->OutputSidePackets().Index(0).Set(MakePacket<HugeModel>(
        R"(An expensive side-packet created only once per graph)"));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    LOG(FATAL) << "Not reached.";
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(OutputSidePacketCachedCalculator);

// Returns true if two packets hold the same data.
bool Equals(Packet p1, Packet p2) {
  return packet_internal::GetHolder(p1) == packet_internal::GetHolder(p2);
}

TEST(CalculatorGraph, OutputSidePacketCached) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "OutputSidePacketCachedCalculator"
          output_side_packet: "model"
        }
        node {
          calculator: "SidePacketToStreamPacketCalculator"
          input_side_packet: "model"
          output_stream: "output"
        }
      )pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  std::vector<Packet> output_packets;
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "output", [&output_packets](const Packet& packet) {
        output_packets.push_back(packet);
        return absl::OkStatus();
      }));

  // Run the graph three times.
  for (int run = 0; run < 3; ++run) {
    MP_ASSERT_OK(graph.StartRun({}));
    MP_ASSERT_OK(graph.WaitUntilDone());
  }
  ASSERT_EQ(3, output_packets.size());
  for (int run = 0; run < output_packets.size(); ++run) {
    EXPECT_TRUE(Equals(output_packets[0], output_packets[run]));
  }
}

TEST(CalculatorGraph, GeneratorAfterCalculatorOpen) {
  CalculatorGraph graph;
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_side_packet: "offset"
        node {
          calculator: "OutputSidePacketInOpenCalculator"
          input_side_packet: "offset"
          output_side_packet: "offset1"
        }
        packet_generator {
          packet_generator: 'PassThroughGenerator'
          input_side_packet: 'offset1'
          output_side_packet: 'offset_out'
        }
        node {
          calculator: "SidePacketToStreamPacketCalculator"
          input_side_packet: "offset_out"
          output_stream: "output"
        }
      )pb");
  MP_ASSERT_OK(graph.Initialize(config));
  std::vector<Packet> output_packets;
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "output", [&output_packets](const Packet& packet) {
        output_packets.push_back(packet);
        return absl::OkStatus();
      }));
  MP_ASSERT_OK(graph.StartRun({{"offset", MakePacket<TimestampDiff>(100)}}));
  MP_ASSERT_OK(graph.WaitUntilDone());
  ASSERT_EQ(1, output_packets.size());
  EXPECT_EQ(100, output_packets[0].Get<TimestampDiff>().Value());
}

TEST(CalculatorGraph, GeneratorAfterCalculatorProcess) {
  CalculatorGraph graph;
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "offset"
        node {
          calculator: "OutputSidePacketInProcessCalculator"
          input_stream: "offset"
          output_side_packet: "offset"
        }
        packet_generator {
          packet_generator: 'PassThroughGenerator'
          input_side_packet: 'offset'
          output_side_packet: 'offset_out'
        }
        node {
          calculator: "SidePacketToStreamPacketCalculator"
          input_side_packet: "offset_out"
          output_stream: "output"
        }
      )pb");
  MP_ASSERT_OK(graph.Initialize(config));
  std::vector<Packet> output_packets;
  MP_ASSERT_OK(graph.ObserveOutputStream(
      "output", [&output_packets](const Packet& packet) {
        output_packets.push_back(packet);
        return absl::OkStatus();
      }));
  // Run twice to verify that we don't duplicate wrapper nodes.
  for (int run = 0; run < 2; ++run) {
    output_packets.clear();
    MP_ASSERT_OK(graph.StartRun({}));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "offset", MakePacket<TimestampDiff>(100).At(Timestamp(0))));
    MP_ASSERT_OK(graph.CloseInputStream("offset"));
    MP_ASSERT_OK(graph.WaitUntilDone());
    ASSERT_EQ(1, output_packets.size());
    EXPECT_EQ(100, output_packets[0].Get<TimestampDiff>().Value());
  }
}

TEST(CalculatorGraph, GetOutputSidePacketAfterCalculatorIsOpened) {
  CalculatorGraph graph;
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "IntegerOutputSidePacketCalculator"
          output_side_packet: "offset"
        }
      )pb");
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  // Must be called to ensure that the calculator is opened.
  MP_ASSERT_OK(graph.WaitUntilIdle());
  absl::StatusOr<Packet> status_or_packet = graph.GetOutputSidePacket("offset");
  MP_ASSERT_OK(status_or_packet);
  EXPECT_EQ(1, status_or_packet.value().Get<int>());
}

}  // namespace
}  // namespace mediapipe
