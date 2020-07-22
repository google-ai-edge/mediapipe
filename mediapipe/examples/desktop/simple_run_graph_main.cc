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

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/map_util.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"

DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");

DEFINE_string(input_side_packets, "",
              "Comma-separated list of key=value pairs specifying side packets "
              "for the CalculatorGraph. All values will be treated as the "
              "string type even if they represent doubles, floats, etc.");

// Local file output flags.
// Output stream
DEFINE_string(output_stream, "",
              "The output stream to output to the local file in csv format.");
DEFINE_string(output_stream_file, "",
              "The name of the local file to output all packets sent to "
              "the stream specified with --output_stream. ");
DEFINE_bool(strip_timestamps, false,
            "If true, only the packet contents (without timestamps) will be "
            "written into the local file.");
// Output side packets
DEFINE_string(output_side_packets, "",
              "A CSV of output side packets to output to local file.");
DEFINE_string(output_side_packets_file, "",
              "The name of the local file to output all side packets specified "
              "with --output_side_packets. ");

::mediapipe::Status OutputStreamToLocalFile(
    ::mediapipe::OutputStreamPoller& poller) {
  std::ofstream file;
  file.open(FLAGS_output_stream_file);
  ::mediapipe::Packet packet;
  while (poller.Next(&packet)) {
    std::string output_data;
    if (!FLAGS_strip_timestamps) {
      absl::StrAppend(&output_data, packet.Timestamp().Value(), ",");
    }
    absl::StrAppend(&output_data, packet.Get<std::string>(), "\n");
    file << output_data;
  }
  file.close();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status OutputSidePacketsToLocalFile(
    ::mediapipe::CalculatorGraph& graph) {
  if (!FLAGS_output_side_packets.empty() &&
      !FLAGS_output_side_packets_file.empty()) {
    std::ofstream file;
    file.open(FLAGS_output_side_packets_file);
    std::vector<std::string> side_packet_names =
        absl::StrSplit(FLAGS_output_side_packets, ',');
    for (const std::string& side_packet_name : side_packet_names) {
      ASSIGN_OR_RETURN(auto status_or_packet,
                       graph.GetOutputSidePacket(side_packet_name));
      file << absl::StrCat(side_packet_name, ":",
                           status_or_packet.Get<std::string>(), "\n");
    }
    file.close();
  } else {
    RET_CHECK(FLAGS_output_side_packets.empty() &&
              FLAGS_output_side_packets_file.empty())
        << "--output_side_packets and --output_side_packets_file should be "
           "specified in pair.";
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(::mediapipe::file::GetContents(
      FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  ::mediapipe::CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<::mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);
  std::map<std::string, ::mediapipe::Packet> input_side_packets;
  if (!FLAGS_input_side_packets.empty()) {
    std::vector<std::string> kv_pairs =
        absl::StrSplit(FLAGS_input_side_packets, ',');
    for (const std::string& kv_pair : kv_pairs) {
      std::vector<std::string> name_and_value = absl::StrSplit(kv_pair, '=');
      RET_CHECK(name_and_value.size() == 2);
      RET_CHECK(
          !::mediapipe::ContainsKey(input_side_packets, name_and_value[0]));
      input_side_packets[name_and_value[0]] =
          ::mediapipe::MakePacket<std::string>(name_and_value[1]);
    }
  }
  LOG(INFO) << "Initialize the calculator graph.";
  ::mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config, input_side_packets));
  if (!FLAGS_output_stream.empty() && !FLAGS_output_stream_file.empty()) {
    ASSIGN_OR_RETURN(auto poller,
                     graph.AddOutputStreamPoller(FLAGS_output_stream));
    LOG(INFO) << "Start running the calculator graph.";
    MP_RETURN_IF_ERROR(graph.StartRun({}));
    MP_RETURN_IF_ERROR(OutputStreamToLocalFile(poller));
  } else {
    RET_CHECK(FLAGS_output_stream.empty() && FLAGS_output_stream_file.empty())
        << "--output_stream and --output_stream_file should be specified in "
           "pair.";
    LOG(INFO) << "Start running the calculator graph.";
    MP_RETURN_IF_ERROR(graph.StartRun({}));
  }
  MP_RETURN_IF_ERROR(graph.WaitUntilDone());
  return OutputSidePacketsToLocalFile(graph);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::mediapipe::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
