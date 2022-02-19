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
#include <string>
#include <vector>

#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"  // NOLINT
#include "mediapipe/framework/tool/validate_type.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif  // defined(__APPLE__)

namespace mediapipe {

void DoSmokeTest(const std::string& graph_proto) {
  const int width = 8;
  const int height = 8;
  const int channels = 3;
  // Prepare input tensor.
  auto input_vec = absl::make_unique<std::vector<Tensor>>();
  input_vec->emplace_back(Tensor::ElementType::kFloat32,
                          Tensor::Shape{1, height, width, channels});
  {
    auto view1 = input_vec->back().GetCpuWriteView();
    auto tensor_buffer = view1.buffer<float>();
    ASSERT_NE(tensor_buffer, nullptr);
    for (int i = 0; i < width * height * channels - 1; i++) {
      tensor_buffer[i] = 1;
    }
  }

  // Prepare single calculator graph to and wait for packets.
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(graph_proto);
  std::vector<Packet> output_packets;
  tool::AddVectorSink("tensor_out", &graph_config, &output_packets);
  CalculatorGraph graph(graph_config);
  MP_ASSERT_OK(graph.StartRun({}));

  // Push the tensor into the graph.
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "tensor_in", Adopt(input_vec.release()).At(Timestamp(0))));
  // Wait until the calculator done processing.
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(1, output_packets.size());

  // Get and process results.
  const std::vector<Tensor>& result_vec =
      output_packets[0].Get<std::vector<Tensor>>();
  ASSERT_EQ(1, result_vec.size());

  const Tensor& result = result_vec[0];
  auto view = result.GetCpuReadView();
  auto result_buffer = view.buffer<float>();
  ASSERT_NE(result_buffer, nullptr);
  for (int i = 0; i < width * height * channels - 1; i++) {
    ASSERT_EQ(3, result_buffer[i]);
  }

  // Fully close graph at end, otherwise calculator+tensors are destroyed
  // after calling WaitUntilDone().
  MP_ASSERT_OK(graph.CloseInputStream("tensor_in"));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

// Tests a simple add model that adds an input tensor to itself.
TEST(InferenceCalculatorTest, SmokeTest) {
  std::string graph_proto = R"(
    input_stream: "tensor_in"
    node {
      calculator: "InferenceCalculator"
      input_stream: "TENSORS:tensor_in"
      output_stream: "TENSORS:tensor_out"
      options {
        [mediapipe.InferenceCalculatorOptions.ext] {
          model_path: "mediapipe/calculators/tensor/testdata/add.bin"
          $delegate
        }
      }
    }
  )";
  // Test CPU inference only.
  DoSmokeTest(/*graph_proto=*/absl::StrReplaceAll(
      graph_proto, {{"$delegate", "delegate { tflite {} }"}}));
  DoSmokeTest(absl::StrReplaceAll(graph_proto,
                                  {{"$delegate", "delegate { xnnpack {} }"}}));
  DoSmokeTest(absl::StrReplaceAll(
      graph_proto,
      {{"$delegate", "delegate { xnnpack { num_threads: 10 } }"}}));
}

TEST(InferenceCalculatorTest, SmokeTest_ModelAsInputSidePacket) {
  std::string graph_proto = R"(
    input_stream: "tensor_in"

    node {
      calculator: "ConstantSidePacketCalculator"
      output_side_packet: "PACKET:model_path"
      options: {
        [mediapipe.ConstantSidePacketCalculatorOptions.ext]: {
          packet { string_value: "mediapipe/calculators/tensor/testdata/add.bin" }
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
      calculator: "InferenceCalculator"
      input_stream: "TENSORS:tensor_in"
      output_stream: "TENSORS:tensor_out"
      input_side_packet: "MODEL:model"
      options {
        [mediapipe.InferenceCalculatorOptions.ext] {
          delegate { tflite {} }
        }
      }
    }
  )";
  DoSmokeTest(graph_proto);
}

}  // namespace mediapipe
