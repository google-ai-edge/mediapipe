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

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/calculators/tensor/inference_calculator_test_base.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/benchmark.h"
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
namespace {

constexpr int kTensorWidth = 8;
constexpr int kTensorHeight = 8;
constexpr int kTensorChannels = 3;

constexpr char kGraphWithModelPathInOption[] = R"(
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
constexpr char kGraphWithModelAsInputSidePacket[] = R"(
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

std::vector<Tensor> CreateInputs() {
  std::vector<Tensor> input_vec;
  // Prepare input tensor.
  input_vec.emplace_back(
      Tensor::ElementType::kFloat32,
      Tensor::Shape{1, kTensorHeight, kTensorWidth, kTensorChannels});
  {
    auto view = input_vec.back().GetCpuWriteView();
    auto num_elements = input_vec.back().shape().num_elements();
    auto tensor_buffer = view.buffer<float>();
    for (int i = 0; i < num_elements; i++) {
      tensor_buffer[i] = 1;
    }
  }

  return input_vec;
}

void RunGraphThenClose(CalculatorGraph& graph, std::vector<Tensor> input_vec) {
  MP_ASSERT_OK(graph.StartRun({}));

  // Push the tensor into the graph.
  if (!input_vec.empty()) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "tensor_in", MakePacket<std::vector<Tensor>>(std::move(input_vec))
                         .At(Timestamp(0))));
  }
  // Wait until the calculator done processing.
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Fully close graph at end, otherwise calculator+tensors are destroyed
  // after calling WaitUntilDone().
  MP_ASSERT_OK(graph.CloseInputStream("tensor_in"));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

void DoSmokeTest(const std::string& graph_proto) {
  auto input_vec = CreateInputs();

  // Prepare single calculator graph to and wait for packets.
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(graph_proto);
  std::vector<Packet> output_packets;
  tool::AddVectorSink("tensor_out", &graph_config, &output_packets);
  CalculatorGraph graph(graph_config);

  RunGraphThenClose(graph, std::move(input_vec));

  ASSERT_EQ(1, output_packets.size());

  // Get and process results.
  const std::vector<Tensor>& result_vec =
      output_packets[0].Get<std::vector<Tensor>>();
  ASSERT_EQ(1, result_vec.size());

  const Tensor& result = result_vec[0];
  auto view = result.GetCpuReadView();
  auto result_buffer = view.buffer<float>();
  ASSERT_NE(result_buffer, nullptr);
  for (int i = 0; i < result.shape().num_elements(); i++) {
    ASSERT_EQ(3, result_buffer[i]);
  }
}

// Tests a simple add model that adds an input tensor to itself.
TEST(InferenceCalculatorTest, SmokeTest) {
  // Test CPU inference only.
  DoSmokeTest(/*graph_proto=*/absl::StrReplaceAll(
      kGraphWithModelPathInOption, {{"$delegate", "delegate { tflite {} }"}}));
  DoSmokeTest(absl::StrReplaceAll(kGraphWithModelPathInOption,
                                  {{"$delegate", "delegate { xnnpack {} }"}}));
  DoSmokeTest(absl::StrReplaceAll(
      kGraphWithModelPathInOption,
      {{"$delegate", "delegate { xnnpack { num_threads: 10 } }"}}));
}

TEST(InferenceCalculatorTest, ModelAsInputSidePacketSmokeTest) {
  DoSmokeTest(kGraphWithModelAsInputSidePacket);
}

void BM_InitializeCalculator(benchmark::State& state) {
  mediapipe::InferenceCalculatorOptions::Delegate delegate;
  delegate.mutable_tflite();
  RunBenchmarkCalculatorInitialization(state, delegate);
}

BENCHMARK(BM_InitializeCalculator);

}  // namespace
}  // namespace mediapipe
