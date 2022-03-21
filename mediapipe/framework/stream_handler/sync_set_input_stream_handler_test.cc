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

#include <memory>
#include <random>
#include <tuple>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/calculator_framework.h"
// TODO: Move protos in another CL after the C++ code migration.
#include "absl/base/macros.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/stream_handler/sync_set_input_stream_handler.pb.h"

using RandomEngine = std::mt19937_64;

namespace mediapipe {

namespace {

// The type LambdaCalculator takes.
typedef std::function<absl::Status(const InputStreamShardSet&,
                                   OutputStreamShardSet*)>
    ProcessFunction;

// Helper function to create a tuple (inside an initializer list).
std::tuple<std::string, Timestamp, std::vector<std::string>> CommandTuple(
    std::string stream, Timestamp timestamp,
    std::vector<std::string> expected) {
  return std::make_tuple(stream, timestamp, expected);
}

// Function to take the inputs and produce a diagnostic output string
// and output a packet with a diagnostic output string which includes
// the input timestamp and the ids of each input which is present.
absl::Status InputsToDebugString(const InputStreamShardSet& inputs,
                                 OutputStreamShardSet* outputs) {
  std::string output;
  Timestamp output_timestamp;
  for (CollectionItemId id = inputs.BeginId(); id < inputs.EndId(); ++id) {
    if (!inputs.Get(id).IsEmpty()) {
      if (output.empty()) {
        output_timestamp = inputs.Get(id).Value().Timestamp();
        if (output_timestamp.IsSpecialValue()) {
          output = output_timestamp.DebugString();
        } else {
          output =
              absl::StrCat("Timestamp(", output_timestamp.DebugString(), ")");
        }
      }
      absl::StrAppend(&output, ",", id.value());
    }
  }
  Packet output_packet;
  ABSL_CONST_INIT static absl::Mutex mu(absl::kConstInit);
  static Timestamp static_timestamp = Timestamp(0);
  {
    absl::MutexLock lock(&mu);
    output_packet = MakePacket<std::string>(output).At(static_timestamp);
    ++static_timestamp;
  }
  // TODO Output at output_timestamp once unordered output stream
  // handlers are allowed.
  outputs->Index(0).AddPacket(output_packet);
  return absl::OkStatus();
}

TEST(SyncSetInputStreamHandlerTest, OrdinaryOperation) {
  CalculatorGraphConfig config = ParseTextProtoOrDie<CalculatorGraphConfig>(
      R"pb(
        input_stream: "a"
        input_stream: "b"
        input_stream: "c"
        input_stream: "d"
        input_stream: "e"
        input_stream: "f"
        input_stream: "g"
        input_stream: "h"
        node {
          calculator: "LambdaCalculator"
          input_stream: "a"
          input_stream: "b"
          input_stream: "c"
          input_stream: "d"
          input_stream: "e"
          input_stream: "f"
          input_stream: "g"
          input_stream: "h"
          output_stream: "output"
          input_side_packet: "lambda"
          input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler"
            options {
              [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                  # TODO Update this to use a mix of indexes
                  # and tags once the framework supports it.
                  tag_index: ":0"
                  tag_index: ":2"
                  tag_index: ":4"
                }
                sync_set { tag_index: ":1" tag_index: ":3" }
                sync_set { tag_index: ":5" }
                sync_set { tag_index: ":6" }
                sync_set { tag_index: ":7" }
              }
            }
          }
        })pb");
  // The sync sets by stream name and CollectionItemId.
  //   {a, c, e}, {b, d}, {f}, {g}, {h}
  //   {0, 2, 4}, {1, 3}, {5}, {6}, {7}

  // The tuple is an "command" which consists of the stream name to
  // add a packet to, the input timestamp of the packet, and a list of
  // output summaries expected.  Keep the list of commands separate for
  // each sync set, so that we can combine them in different ways later
  // (better testing their independence).
  std::vector<
      std::vector<std::tuple<std::string, Timestamp, std::vector<std::string>>>>
      command_sets;
  command_sets.push_back({
      CommandTuple("a", Timestamp(0), {}),
      CommandTuple("c", Timestamp(0), {}),
      CommandTuple("a", Timestamp(10), {}),
      CommandTuple("e", Timestamp(0), {"Timestamp(0),0,2,4"}),
      CommandTuple("c", Timestamp(10), {}),
      CommandTuple("e", Timestamp(10), {"Timestamp(10),0,2,4"}),
      CommandTuple("e", Timestamp(20), {}),
      CommandTuple("a", Timestamp(20), {}),
      CommandTuple("c", Timestamp(20), {"Timestamp(20),0,2,4"}),
      CommandTuple("c", Timestamp(30), {}),
      CommandTuple("a", Timestamp(30), {}),
      CommandTuple("a", Timestamp(40), {}),
      CommandTuple("a", Timestamp(50), {}),
      CommandTuple("c", Timestamp(40), {}),
      CommandTuple("a", Timestamp::Done(), {}),
      CommandTuple("e", Timestamp(40),
                   {"Timestamp(30),0,2", "Timestamp(40),0,2,4"}),
      CommandTuple("c", Timestamp(50), {}),
      CommandTuple("e", Timestamp(50), {"Timestamp(50),0,2,4"}),
      CommandTuple("c", Timestamp(60), {}),
      CommandTuple("c", Timestamp(70), {}),
      CommandTuple("c", Timestamp::Done(), {}),
      CommandTuple("e", Timestamp::Done(),
                   {"Timestamp(60),2", "Timestamp(70),2"}),
  });

  command_sets.push_back({
      CommandTuple("b", Timestamp(-300), {}),  //
      CommandTuple("b", Timestamp(-200), {}),  //
      CommandTuple("b", Timestamp(-100), {}),  //
      CommandTuple("d", Timestamp(-200),
                   {"Timestamp(-300),1", "Timestamp(-200),1,3"}),
      CommandTuple("d", Timestamp(-20), {"Timestamp(-100),1"}),  //
      CommandTuple("d", Timestamp(-10), {}),                     //
      CommandTuple("b", Timestamp(0), {"Timestamp(-20),3", "Timestamp(-10),3"}),
      CommandTuple("d", Timestamp(0), {"Timestamp(0),1,3"}),      //
      CommandTuple("d", Timestamp(10), {}),                       //
      CommandTuple("b", Timestamp(10), {"Timestamp(10),1,3"}),    //
      CommandTuple("b", Timestamp(20), {}),                       //
      CommandTuple("d", Timestamp(200), {"Timestamp(20),1"}),     //
      CommandTuple("b", Timestamp(100), {"Timestamp(100),1"}),    //
      CommandTuple("b", Timestamp(200), {"Timestamp(200),1,3"}),  //
      CommandTuple("b", Timestamp(250), {}),                      //
      CommandTuple("b", Timestamp(300), {}),                      //
      CommandTuple("d", Timestamp::Done(),
                   {"Timestamp(250),1", "Timestamp(300),1"}),
      CommandTuple("b", Timestamp::Done(), {}),
  });

  std::vector<std::tuple<std::string, Timestamp, std::vector<std::string>>>
      temp_commands;
  for (Timestamp t = Timestamp(-350); t < Timestamp(350); t += 35) {
    temp_commands.push_back(CommandTuple(
        "f", t, {absl::StrCat("Timestamp(", t.DebugString(), "),5")}));
  }
  temp_commands.push_back(CommandTuple("f", Timestamp::Done(), {}));
  command_sets.push_back(temp_commands);

  command_sets.push_back(
      {CommandTuple("g", Timestamp::PreStream(),
                    {absl::StrCat(Timestamp::PreStream().DebugString(), ",6")}),
       CommandTuple("g", Timestamp::Done(), {})});

  command_sets.push_back(
      {CommandTuple(
           "h", Timestamp::PostStream(),
           {absl::StrCat(Timestamp::PostStream().DebugString(), ",7")}),
       CommandTuple("h", Timestamp::Done(), {})});

  int num_commands = 0;
  std::vector<int> cummulative_num_commands;
  for (int i = 0; i < command_sets.size(); ++i) {
    num_commands += command_sets[i].size();
    cummulative_num_commands.push_back(num_commands);
  }

  RandomEngine rng(testing::UnitTest::GetInstance()->random_seed());
  for (int iter = 0; iter < 1000; ++iter) {
    LOG(INFO) << "Starting command shuffling iteration " << iter;

    // Merge the commands for each sync set together into a serial list.
    // This is done by randomly choosing which list to grab from next.
    std::vector<std::tuple<std::string, Timestamp, std::vector<std::string>>>
        shuffled_commands;
    std::vector<int> current_positions(command_sets.size(), 0);
    while (shuffled_commands.size() < num_commands) {
      // Weight the index chosen by how many commands are in each set.
      int rand_num = rng() % num_commands;
      int command_set_index;
      for (command_set_index = 0;
           rand_num >= cummulative_num_commands[command_set_index];
           ++command_set_index) {
        // Find the index corresponding to this weighted random number.
      }
      // Add the command to the list if they haven't already all been used.
      if (current_positions[command_set_index] <
          command_sets[command_set_index].size()) {
        shuffled_commands.push_back(
            command_sets[command_set_index]
                        [current_positions[command_set_index]]);
        ++current_positions[command_set_index];
        VLOG(2) << "ShuffledCommand (" << std::get<0>(shuffled_commands.back())
                << ", Timestamp(" << std::get<1>(shuffled_commands.back())
                << "))";
      }
    }

    CalculatorGraph graph;

    // Remove one* of the sync sets from the configuration, forcing it's
    // streams into the default sync set, which is otherwise empty.
    // * Actually, also have a possibility of not removing any.
    CalculatorGraphConfig modified_config = config;
    auto* repeated_field =
        modified_config.mutable_node(0)
            ->mutable_input_stream_handler()
            ->mutable_options()
            ->MutableExtension(SyncSetInputStreamHandlerOptions::ext)
            ->mutable_sync_set();
    int index_to_remove = rng() % (repeated_field->size() + 1);
    if (index_to_remove != repeated_field->size()) {
      repeated_field->SwapElements(index_to_remove, repeated_field->size() - 1);
      repeated_field->RemoveLast();
    }
    std::shuffle(repeated_field->begin(), repeated_field->end(), rng);

    VLOG(2) << "Modified configuration: " << modified_config.DebugString();

    // Setup and run the graph.
    MP_ASSERT_OK(graph.Initialize(
        modified_config,
        {{"lambda", MakePacket<ProcessFunction>(InputsToDebugString)}}));
    std::deque<Packet> outputs;
    MP_ASSERT_OK(
        graph.ObserveOutputStream("output", [&outputs](const Packet& packet) {
          outputs.push_back(packet);
          return absl::OkStatus();
        }));
    MP_ASSERT_OK(graph.StartRun({}));
    for (int command_index = 0; command_index < shuffled_commands.size();
         /* command_index is incremented by the inner loop. */) {
      int initial_command_index = command_index;
      int command_batch_size = rng() % 10;
      std::vector<std::string> expected_strings;
      // Push in a batch of commands.
      for (; command_index < shuffled_commands.size() &&
             command_index < initial_command_index + command_batch_size;
           ++command_index) {
        const auto& tup = shuffled_commands[command_index];
        const std::string& stream_name = std::get<0>(tup);
        Timestamp timestamp = std::get<1>(tup);
        expected_strings.insert(expected_strings.end(),
                                std::get<2>(tup).begin(),
                                std::get<2>(tup).end());

        VLOG(1) << "Adding (" << stream_name << ", Timestamp: " << timestamp
                << ")";
        if (timestamp == Timestamp::Done()) {
          MP_ASSERT_OK(graph.CloseInputStream(stream_name));
        } else {
          MP_ASSERT_OK(graph.AddPacketToInputStream(
              stream_name, MakePacket<int>(0).At(timestamp)));
        }
      }
      // Ensure that we produce all packets which we can.
      MP_ASSERT_OK(graph.WaitUntilIdle());

      // Check the output strings (ignoring order, since calculator may
      // have run in parallel).
      // TODO Actually enable parallel process calls.
      std::vector<std::string> actual_strings;
      for (const Packet& output : outputs) {
        actual_strings.push_back(output.Get<std::string>());
        VLOG(1) << "Expecting \"" << actual_strings.back() << "\"";
      }
      if (actual_strings.empty()) {
        VLOG(1) << "Expecting nothing.";
      }
      outputs.clear();
      EXPECT_THAT(actual_strings,
                  testing::UnorderedElementsAreArray(expected_strings));
    }
    MP_ASSERT_OK(graph.WaitUntilDone());
  }
}

}  // namespace
}  // namespace mediapipe
