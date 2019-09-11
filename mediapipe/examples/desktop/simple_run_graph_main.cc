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
// A simple main function to run a MediaPipe graph.

#include "absl/strings/str_split.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/map_util.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");

DEFINE_string(input_side_packets, "",
              "Comma-separated list of key=value pairs specifying side packets "
              "for the CalculatorGraph. All values will be treated as the "
              "string type even if they represent doubles, floats, etc.");

::mediapipe::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);
  std::map<std::string, ::mediapipe::Packet> input_side_packets;
  std::vector<std::string> kv_pairs =
      absl::StrSplit(FLAGS_input_side_packets, ',');
  for (const std::string& kv_pair : kv_pairs) {
    std::vector<std::string> name_and_value = absl::StrSplit(kv_pair, '=');
    RET_CHECK(name_and_value.size() == 2);
    RET_CHECK(!::mediapipe::ContainsKey(input_side_packets, name_and_value[0]));
    input_side_packets[name_and_value[0]] =
        ::mediapipe::MakePacket<std::string>(name_and_value[1]);
  }
  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config, input_side_packets));
  LOG(INFO) << "Start running the calculator graph.";
  return graph.Run();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::mediapipe::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
  } else {
    LOG(INFO) << "Success!";
  }
  return 0;
}
