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

#include "absl/strings/str_replace.h"
#include "mediapipe/calculators/tflite/tflite_inference_calculator_test_common.h"

namespace mediapipe {

// Tests a simple add model that adds an input tensor to itself.
TEST(TfLiteInferenceCalculatorTest, SmokeTest) {
  std::string graph_proto = R"(
    input_stream: "tensor_in"
    node {
      calculator: "TfLiteInferenceCalculator"
      input_stream: "TENSORS:tensor_in"
      output_stream: "TENSORS:tensor_out"
      options {
        [mediapipe.TfLiteInferenceCalculatorOptions.ext] {
          model_path: "mediapipe/calculators/tflite/testdata/add.bin"
          $delegate
        }
      }
    }
  )";
  // Test CPU inference only.
  DoSmokeTest<float>(/*graph_proto=*/absl::StrReplaceAll(
      graph_proto, {{"$delegate", "delegate { tflite {} }"}}));
  DoSmokeTest<float>(absl::StrReplaceAll(
      graph_proto, {{"$delegate", "delegate { xnnpack {} }"}}));
  DoSmokeTest<float>(absl::StrReplaceAll(
      graph_proto,
      {{"$delegate", "delegate { xnnpack { num_threads: 10 } }"}}));
}

TEST(TfLiteInferenceCalculatorTest, SmokeTest_ModelAsInputSidePacket) {
  std::string graph_proto = R"(
    input_stream: "tensor_in"

    node {
      calculator: "ConstantSidePacketCalculator"
      output_side_packet: "PACKET:model_path"
      options: {
        [mediapipe.ConstantSidePacketCalculatorOptions.ext]: {
          packet { string_value: "mediapipe/calculators/tflite/testdata/add.bin" }
        }
      }
    }

    node {
      calculator: "LocalFileContentsCalculator"
      input_side_packet: "FILE_PATH:model_path"
      output_side_packet: "CONTENTS:model_blob"
    }

    node {
      calculator: "TfLiteModelCalculator"
      input_side_packet: "MODEL_BLOB:model_blob"
      output_side_packet: "MODEL:model"
    }

    node {
      calculator: "TfLiteInferenceCalculator"
      input_stream: "TENSORS:tensor_in"
      output_stream: "TENSORS:tensor_out"
      input_side_packet: "MODEL:model"
      options {
        [mediapipe.TfLiteInferenceCalculatorOptions.ext] {
          use_gpu: false
          delegate { tflite {} }
        }
      }
    }
  )";
  DoSmokeTest<float>(graph_proto);
}

}  // namespace mediapipe
