// Copyright (c) 2023 Intel Corporation
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

#include "absl/strings/str_replace.h"
#include "mediapipe/calculators/openvino/openvino_inference_calculator_test_common.h"
namespace mediapipe {

// Tests a simple add model that adds two input tensors
TEST(OpenVINOInferenceCalculatorTest, SmokeTest) {
  std::string graph_proto = R"(
    input_stream: "tensor_in"
    node {
      calculator: "OpenVINOInferenceCalculator"
      input_stream: "TENSORS:tensor_in"
      output_stream: "TENSORS:tensor_out"
      options {
        [mediapipe.OpenVINOInferenceCalculatorOptions.ext] {
          model_path: "mediapipe/calculators/openvino/testdata/add.xml"
          $device
        }
      }
    }
  )";
  // Test CPU inference only.
  DoSmokeTest<uint8_t>(/*graph_proto=*/absl::StrReplaceAll(
      graph_proto, {{"$device", "device { cpu {} }"}}));
//  DoSmokeTest<float>(/*graph_proto=*/absl::StrReplaceAll(
//      graph_proto, {{"$device", "device { cpu {} }"}}));
}

// TEST(OpenVINOInferenceCalculatorTest, SmokeTest_ModelAsInputSidePacket) {
//   std::string graph_proto = R"(
//     input_stream: "tensor_in"
//
//     node {
//       calculator: "ConstantSidePacketCalculator"
//       output_side_packet: "PACKET:model_path"
//       options: {
//         [mediapipe.ConstantSidePacketCalculatorOptions.ext]: {
//           packet { string_value: "mediapipe/calculators/openvino/testdata/add.bin" }
//         }
//       }
//     }
//
//     node {
//       calculator: "LocalFileContentsCalculator"
//       input_side_packet: "FILE_PATH:model_path"
//       output_side_packet: "CONTENTS:model_blob"
//     }
//
//     node {
//       calculator: "OpenVINOModelCalculator"
//       input_side_packet: "MODEL_BLOB:model_blob"
//       output_side_packet: "MODEL:model"
//     }
//
//     node {
//       calculator: "OpenVINOInferenceCalculator"
//       input_stream: "TENSORS:tensor_in"
//       output_stream: "TENSORS:tensor_out"
//       input_side_packet: "MODEL:model"
//       options {
//         [mediapipe.OpenVINOInferenceCalculatorOptions.ext] {
//           use_gpu: false
//           device { openvino {} }
//         }
//       }
//     }
//   )";
//   DoSmokeTest<float>(graph_proto);
// }

}  // namespace mediapipe
