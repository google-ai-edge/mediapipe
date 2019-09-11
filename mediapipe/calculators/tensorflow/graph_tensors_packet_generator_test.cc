// Copyright 2018 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensorflow/graph_tensors_packet_generator.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/packet_generator.pb.h"
#include "mediapipe/framework/packet_set.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/validate_type.h"
#include "tensorflow/core/framework/tensor.h"

namespace mediapipe {

namespace {

namespace tf = ::tensorflow;

// Helper function that creates a row tensor that is initialized to zeros.
tf::Tensor ZeroRowTensor(const int col_length) {
  tf::Tensor tensor(tf::DT_FLOAT, tf::TensorShape{1, col_length});
  tensor.flat<float>().setZero();

  return tensor;
}

class GraphTensorsPacketGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    extendable_options_.Clear();
    generator_options_ = extendable_options_.MutableExtension(
        GraphTensorsPacketGeneratorOptions::ext);
    generator_options_->add_tensor_tag("A");
    generator_options_->add_tensor_num_nodes(3);
    generator_options_->add_tensor_tag("B");
    generator_options_->add_tensor_num_nodes(4);
  }

  void VerifyTensorMap(PacketSet* output_side_packets) {
    const std::map<std::string, tf::Tensor>* tensor_map =
        GetFromUniquePtr<std::map<std::string, tf::Tensor>>(
            output_side_packets->Index(0));

    EXPECT_FALSE(tensor_map->find("A") == tensor_map->end());
    EXPECT_FALSE(tensor_map->find("B") == tensor_map->end());

    tf::Tensor expected_tensor = ZeroRowTensor(3);
    EXPECT_EQ(expected_tensor.DebugString(),
              tensor_map->find("A")->second.DebugString());

    expected_tensor = ZeroRowTensor(4);
    EXPECT_EQ(expected_tensor.DebugString(),
              tensor_map->find("B")->second.DebugString());
  }
  PacketGeneratorOptions extendable_options_;
  GraphTensorsPacketGeneratorOptions* generator_options_;
};

// Test that the tensors are of the right size and shape
TEST_F(GraphTensorsPacketGeneratorTest, VerifyTensorSizeShapeAndValue) {
  PacketSet inputs({});
  PacketSet outputs(1);

  ::mediapipe::Status run_status = tool::RunGenerateAndValidateTypes(
      "GraphTensorsPacketGenerator", extendable_options_, inputs, &outputs);
  MP_EXPECT_OK(run_status) << run_status.message();
  VerifyTensorMap(&outputs);
}

}  // namespace
}  // namespace mediapipe
