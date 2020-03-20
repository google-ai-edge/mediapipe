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
#include "mediapipe/calculators/tflite/tflite_inference_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"  // NOLINT
#include "mediapipe/framework/tool/validate_type.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif  // defined(__APPLE__)

namespace mediapipe {

using ::tflite::Interpreter;

void DoSmokeTest(const std::string& graph_proto) {
  const int width = 8;
  const int height = 8;
  const int channels = 3;

  // Prepare input tensor.
  std::unique_ptr<Interpreter> interpreter(new Interpreter);
  ASSERT_NE(interpreter, nullptr);

  interpreter->AddTensors(1);
  interpreter->SetInputs({0});
  interpreter->SetOutputs({0});
  interpreter->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {3},
                                            TfLiteQuantization());
  int t = interpreter->inputs()[0];
  TfLiteTensor* tensor = interpreter->tensor(t);
  interpreter->ResizeInputTensor(t, {width, height, channels});
  interpreter->AllocateTensors();

  float* tensor_buffer = tensor->data.f;
  ASSERT_NE(tensor_buffer, nullptr);
  for (int i = 0; i < width * height * channels - 1; i++) {
    tensor_buffer[i] = 1;
  }

  auto input_vec = absl::make_unique<std::vector<TfLiteTensor>>();
  input_vec->emplace_back(*tensor);

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
  const std::vector<TfLiteTensor>& result_vec =
      output_packets[0].Get<std::vector<TfLiteTensor>>();
  ASSERT_EQ(1, result_vec.size());

  const TfLiteTensor* result = &result_vec[0];
  float* result_buffer = result->data.f;
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
  DoSmokeTest(
      /*graph_proto=*/absl::StrReplaceAll(graph_proto, {{"$delegate", ""}}));
  DoSmokeTest(/*graph_proto=*/absl::StrReplaceAll(
      graph_proto, {{"$delegate", "delegate { tflite {} }"}}));
  DoSmokeTest(/*graph_proto=*/absl::StrReplaceAll(
      graph_proto, {{"$delegate", "delegate { xnnpack {} }"}}));
  DoSmokeTest(/*graph_proto=*/absl::StrReplaceAll(
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
        }
      }
    }
  )";
  DoSmokeTest(graph_proto);
}

}  // namespace mediapipe
