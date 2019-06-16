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
// Generates row tensors of prescribed length that are initialized to zeros.
// The tensors are placed in an ordered map, which maps the tensors to the
// tensor tags, and emitted as a packet. This generator has been developed
// primarily to generate initialization states for LSTMs.

#include <map>
#include <string>

#include "mediapipe/calculators/tensorflow/graph_tensors_packet_generator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/status_util.h"
#include "tensorflow/core/framework/tensor.h"

namespace mediapipe {

namespace tf = ::tensorflow;

class GraphTensorsPacketGenerator : public PacketGenerator {
 public:
  static ::mediapipe::Status FillExpectations(
      const PacketGeneratorOptions& extendable_options,
      PacketTypeSet* input_side_packets, PacketTypeSet* output_side_packets) {
    RET_CHECK(extendable_options.HasExtension(
        GraphTensorsPacketGeneratorOptions::ext));
    const auto& options = extendable_options.GetExtension(  // NOLINT
        GraphTensorsPacketGeneratorOptions::ext);
    output_side_packets->Index(0)
        .Set<std::unique_ptr<std::map<std::string, tf::Tensor>>>(
            /* "A map of tensor tags and tensors" */);
    RET_CHECK_EQ(options.tensor_tag_size(), options.tensor_num_nodes_size());
    RET_CHECK_GT(options.tensor_tag_size(), 0);
    return ::mediapipe::OkStatus();
  }

  static ::mediapipe::Status Generate(
      const PacketGeneratorOptions& packet_generator_options,
      const PacketSet& input_side_packets, PacketSet* output_side_packets) {
    const GraphTensorsPacketGeneratorOptions& options =
        packet_generator_options.GetExtension(
            GraphTensorsPacketGeneratorOptions::ext);
    // Output bundle packet.
    auto tensor_map = absl::make_unique<std::map<std::string, tf::Tensor>>();

    for (int i = 0; i < options.tensor_tag_size(); ++i) {
      const std::string& tensor_tag = options.tensor_tag(i);
      const int32 tensor_num_nodes = options.tensor_num_nodes(i);
      (*tensor_map)[tensor_tag] =
          tf::Tensor(tf::DT_FLOAT, tf::TensorShape{1, tensor_num_nodes});
      (*tensor_map)[tensor_tag].flat<float>().setZero();
    }
    output_side_packets->Index(0) = AdoptAsUniquePtr(tensor_map.release());
    return ::mediapipe::OkStatus();
  }
};
REGISTER_PACKET_GENERATOR(GraphTensorsPacketGenerator);

}  // namespace mediapipe
