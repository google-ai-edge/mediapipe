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
// A simple main function to run a MediaPipe graph. Input side packets are read
// from files provided via the command line and output side packets are written
// to disk.
#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_split.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/map_util.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_side_packets, "",
          "Comma-separated list of key=value pairs specifying side packets "
          "and corresponding file paths for the CalculatorGraph. The side "
          "packets are read from the files and fed to the graph as strings "
          "even if they represent doubles, floats, etc.");
ABSL_FLAG(std::string, output_side_packets, "",
          "Comma-separated list of key=value pairs specifying the output "
          "side packets and paths to write to disk for the "
          "CalculatorGraph.");

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
  std::vector<std::string> kv_pairs =
      absl::StrSplit(absl::GetFlag(FLAGS_input_side_packets), ',');
  for (const std::string& kv_pair : kv_pairs) {
    std::vector<std::string> name_and_value = absl::StrSplit(kv_pair, '=');
    RET_CHECK(name_and_value.size() == 2);
    RET_CHECK(!input_side_packets.contains(name_and_value[0]));
    std::string input_side_packet_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        name_and_value[1], &input_side_packet_contents));
    input_side_packets[name_and_value[0]] =
        mediapipe::MakePacket<std::string>(input_side_packet_contents);
  }

  mediapipe::MatrixData inc3_pca_mean_matrix_data,
      inc3_pca_projection_matrix_data, vggish_pca_mean_matrix_data,
      vggish_pca_projection_matrix_data;
  mediapipe::Matrix inc3_pca_mean_matrix, inc3_pca_projection_matrix,
      vggish_pca_mean_matrix, vggish_pca_projection_matrix;

  std::string content;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      "/tmp/mediapipe/inception3_mean_matrix_data.pb", &content));
  inc3_pca_mean_matrix_data.ParseFromString(content);
  mediapipe::MatrixFromMatrixDataProto(inc3_pca_mean_matrix_data,
                                       &inc3_pca_mean_matrix);
  input_side_packets["inception3_pca_mean_matrix"] =
      mediapipe::MakePacket<mediapipe::Matrix>(inc3_pca_mean_matrix);

  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      "/tmp/mediapipe/inception3_projection_matrix_data.pb", &content));
  inc3_pca_projection_matrix_data.ParseFromString(content);
  mediapipe::MatrixFromMatrixDataProto(inc3_pca_projection_matrix_data,
                                       &inc3_pca_projection_matrix);
  input_side_packets["inception3_pca_projection_matrix"] =
      mediapipe::MakePacket<mediapipe::Matrix>(inc3_pca_projection_matrix);

  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      "/tmp/mediapipe/vggish_mean_matrix_data.pb", &content));
  vggish_pca_mean_matrix_data.ParseFromString(content);
  mediapipe::MatrixFromMatrixDataProto(vggish_pca_mean_matrix_data,
                                       &vggish_pca_mean_matrix);
  input_side_packets["vggish_pca_mean_matrix"] =
      mediapipe::MakePacket<mediapipe::Matrix>(vggish_pca_mean_matrix);

  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      "/tmp/mediapipe/vggish_projection_matrix_data.pb", &content));
  vggish_pca_projection_matrix_data.ParseFromString(content);
  mediapipe::MatrixFromMatrixDataProto(vggish_pca_projection_matrix_data,
                                       &vggish_pca_projection_matrix);
  input_side_packets["vggish_pca_projection_matrix"] =
      mediapipe::MakePacket<mediapipe::Matrix>(vggish_pca_projection_matrix);

  ABSL_LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config, input_side_packets));
  ABSL_LOG(INFO) << "Start running the calculator graph.";
  MP_RETURN_IF_ERROR(graph.Run());
  ABSL_LOG(INFO) << "Gathering output side packets.";
  kv_pairs = absl::StrSplit(absl::GetFlag(FLAGS_output_side_packets), ',');
  for (const std::string& kv_pair : kv_pairs) {
    std::vector<std::string> name_and_value = absl::StrSplit(kv_pair, '=');
    RET_CHECK(name_and_value.size() == 2);
    absl::StatusOr<mediapipe::Packet> output_packet =
        graph.GetOutputSidePacket(name_and_value[0]);
    RET_CHECK(output_packet.ok())
        << "Packet " << name_and_value[0] << " was not available.";
    const std::string& serialized_string =
        output_packet.value().Get<std::string>();
    MP_RETURN_IF_ERROR(
        mediapipe::file::SetContents(name_and_value[1], serialized_string));
  }
  return absl::OkStatus();
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
