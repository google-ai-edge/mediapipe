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
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/map_util.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");

ABSL_FLAG(std::string, input_side_packets, "",
          "Comma-separated list of key=value pairs specifying side packets "
          "for the CalculatorGraph. All values will be treated as the "
          "string type even if they represent doubles, floats, etc.");

// Local file output flags.
// Output stream
ABSL_FLAG(std::string, output_stream, "",
          "The output stream to output to the local file in csv format.");
ABSL_FLAG(std::string, output_stream_file, "",
          "The name of the local file to output all packets sent to "
          "the stream specified with --output_stream. ");
ABSL_FLAG(bool, strip_timestamps, false,
          "If true, only the packet contents (without timestamps) will be "
          "written into the local file.");
// Output side packets
ABSL_FLAG(std::string, output_side_packets, "",
          "A CSV of output side packets to output to local file.");
ABSL_FLAG(std::string, output_side_packets_file, "",
          "The name of the local file to output all side packets specified "
          "with --output_side_packets. ");

absl::Status OutputStreamToLocalFile(mediapipe::OutputStreamPoller& poller) {
  std::ofstream file;
  file.open(absl::GetFlag(FLAGS_output_stream_file));
  mediapipe::Packet packet;
  while (poller.Next(&packet)) {
    std::string output_data;
    if (!absl::GetFlag(FLAGS_strip_timestamps)) {
      absl::StrAppend(&output_data, packet.Timestamp().Value(), ",");
    }
    absl::StrAppend(&output_data, packet.Get<std::string>(), "\n");
    file << output_data;
  }
  file.close();
  return absl::OkStatus();
}

absl::Status OutputSidePacketsToLocalFile(mediapipe::CalculatorGraph& graph) {
  if (!absl::GetFlag(FLAGS_output_side_packets).empty() &&
      !absl::GetFlag(FLAGS_output_side_packets_file).empty()) {
    std::ofstream file;
    file.open(absl::GetFlag(FLAGS_output_side_packets_file));
    std::vector<std::string> side_packet_names =
        absl::StrSplit(absl::GetFlag(FLAGS_output_side_packets), ',');
    for (const std::string& side_packet_name : side_packet_names) {
      MP_ASSIGN_OR_RETURN(auto status_or_packet,
                          graph.GetOutputSidePacket(side_packet_name));
      file << absl::StrCat(side_packet_name, ":",
                           status_or_packet.Get<std::string>(), "\n");
    }
    file.close();
  } else {
    RET_CHECK(absl::GetFlag(FLAGS_output_side_packets).empty() &&
              absl::GetFlag(FLAGS_output_side_packets_file).empty())
        << "--output_side_packets and --output_side_packets_file should be "
           "specified in pair.";
  }
  return absl::OkStatus();
}

absl::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      absl::GetFlag(FLAGS_calculator_graph_config_file),
      &calculator_graph_config_contents));
  ABSL_LOG(INFO) << "Get calculator graph config contents: "
                 << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);
  std::map<std::string, mediapipe::Packet> input_side_packets;
  if (!absl::GetFlag(FLAGS_input_side_packets).empty()) {
    std::vector<std::string> kv_pairs =
        absl::StrSplit(absl::GetFlag(FLAGS_input_side_packets), ',');
    for (const std::string& kv_pair : kv_pairs) {
      std::vector<std::string> name_and_value = absl::StrSplit(kv_pair, '=');
      RET_CHECK(name_and_value.size() == 2);
      RET_CHECK(!mediapipe::ContainsKey(input_side_packets, name_and_value[0]));
      input_side_packets[name_and_value[0]] =
          mediapipe::MakePacket<std::string>(name_and_value[1]);
    }
  }
  ABSL_LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config, input_side_packets));
  if (!absl::GetFlag(FLAGS_output_stream).empty() &&
      !absl::GetFlag(FLAGS_output_stream_file).empty()) {
    MP_ASSIGN_OR_RETURN(auto poller, graph.AddOutputStreamPoller(
                                         absl::GetFlag(FLAGS_output_stream)));
    ABSL_LOG(INFO) << "Start running the calculator graph.";
    MP_RETURN_IF_ERROR(graph.StartRun({}));
    MP_RETURN_IF_ERROR(OutputStreamToLocalFile(poller));
  } else {
    RET_CHECK(absl::GetFlag(FLAGS_output_stream).empty() &&
              absl::GetFlag(FLAGS_output_stream_file).empty())
        << "--output_stream and --output_stream_file should be specified in "
           "pair.";
    ABSL_LOG(INFO) << "Start running the calculator graph.";
    MP_RETURN_IF_ERROR(graph.StartRun({}));
  }
  MP_RETURN_IF_ERROR(graph.WaitUntilDone());
  return OutputSidePacketsToLocalFile(graph);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  absl::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    ABSL_LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
