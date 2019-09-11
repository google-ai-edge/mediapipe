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

#include "mediapipe/calculators/core/split_vector_calculator.h"

#include <memory>
#include <string>
#include <vector>

#include "mediapipe/calculators/core/split_vector_calculator.pb.h"
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

namespace mediapipe {

using ::tflite::Interpreter;

const int width = 1;
const int height = 1;
const int channels = 1;

class SplitTfLiteTensorVectorCalculatorTest : public ::testing::Test {
 protected:
  void TearDown() {
    // Note: Since the pointers contained in this vector will be cleaned up by
    // the interpreter, only ensure that the vector is cleaned up for the next
    // test.
    input_buffers_.clear();
  }

  void PrepareTfLiteTensorVector(int vector_size) {
    ASSERT_NE(interpreter_, nullptr);

    // Prepare input tensors.
    std::vector<int> indices(vector_size);
    for (int i = 0; i < vector_size; ++i) {
      indices[i] = i;
    }
    interpreter_->AddTensors(vector_size);
    interpreter_->SetInputs(indices);

    input_vec_ = absl::make_unique<std::vector<TfLiteTensor>>();
    for (int i = 0; i < vector_size; ++i) {
      interpreter_->SetTensorParametersReadWrite(i, kTfLiteFloat32, "", {3},
                                                 TfLiteQuantization());
      const int tensor_index = interpreter_->inputs()[i];
      interpreter_->ResizeInputTensor(tensor_index, {width, height, channels});
    }

    interpreter_->AllocateTensors();

    // Save the tensor buffer pointers for comparison after the graph runs.
    input_buffers_ = std::vector<float*>(vector_size);
    for (int i = 0; i < vector_size; ++i) {
      const int tensor_index = interpreter_->inputs()[i];
      TfLiteTensor* tensor = interpreter_->tensor(tensor_index);
      float* tensor_buffer = tensor->data.f;
      ASSERT_NE(tensor_buffer, nullptr);
      for (int j = 0; j < width * height * channels; ++j) {
        tensor_buffer[j] = i;
      }
      input_vec_->push_back(*tensor);
      input_buffers_[i] = tensor_buffer;
    }
  }

  void ValidateVectorOutput(std::vector<Packet>& output_packets,
                            int expected_elements, int input_begin_index) {
    ASSERT_EQ(1, output_packets.size());
    const std::vector<TfLiteTensor>& output_vec =
        output_packets[0].Get<std::vector<TfLiteTensor>>();
    ASSERT_EQ(expected_elements, output_vec.size());

    for (int i = 0; i < expected_elements; ++i) {
      const int expected_value = input_begin_index + i;
      const TfLiteTensor* result = &output_vec[i];
      float* result_buffer = result->data.f;
      ASSERT_NE(result_buffer, nullptr);
      ASSERT_EQ(result_buffer, input_buffers_[input_begin_index + i]);
      for (int j = 0; j < width * height * channels; ++j) {
        ASSERT_EQ(expected_value, result_buffer[j]);
      }
    }
  }

  void ValidateElementOutput(std::vector<Packet>& output_packets,
                             int input_begin_index) {
    ASSERT_EQ(1, output_packets.size());

    const TfLiteTensor& result = output_packets[0].Get<TfLiteTensor>();
    float* result_buffer = result.data.f;
    ASSERT_NE(result_buffer, nullptr);
    ASSERT_EQ(result_buffer, input_buffers_[input_begin_index]);

    const int expected_value = input_begin_index;
    for (int j = 0; j < width * height * channels; ++j) {
      ASSERT_EQ(expected_value, result_buffer[j]);
    }
  }

  std::unique_ptr<Interpreter> interpreter_ = absl::make_unique<Interpreter>();
  std::unique_ptr<std::vector<TfLiteTensor>> input_vec_ = nullptr;
  std::vector<float*> input_buffers_;
  std::unique_ptr<CalculatorRunner> runner_ = nullptr;
};

TEST_F(SplitTfLiteTensorVectorCalculatorTest, SmokeTest) {
  ASSERT_NE(interpreter_, nullptr);

  PrepareTfLiteTensorVector(/*vector_size=*/5);
  ASSERT_NE(input_vec_, nullptr);

  // Prepare a graph to use the SplitTfLiteTensorVectorCalculator.
  CalculatorGraphConfig graph_config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"(
            input_stream: "tensor_in"
            node {
              calculator: "SplitTfLiteTensorVectorCalculator"
              input_stream: "tensor_in"
              output_stream: "range_0"
              output_stream: "range_1"
              output_stream: "range_2"
              options {
                [mediapipe.SplitVectorCalculatorOptions.ext] {
                  ranges: { begin: 0 end: 1 }
                  ranges: { begin: 1 end: 4 }
                  ranges: { begin: 4 end: 5 }
                }
              }
            }
          )");
  std::vector<Packet> range_0_packets;
  tool::AddVectorSink("range_0", &graph_config, &range_0_packets);
  std::vector<Packet> range_1_packets;
  tool::AddVectorSink("range_1", &graph_config, &range_1_packets);
  std::vector<Packet> range_2_packets;
  tool::AddVectorSink("range_2", &graph_config, &range_2_packets);

  // Run the graph.
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "tensor_in", Adopt(input_vec_.release()).At(Timestamp(0))));
  // Wait until the calculator finishes processing.
  MP_ASSERT_OK(graph.WaitUntilIdle());

  ValidateVectorOutput(range_0_packets, /*expected_elements=*/1,
                       /*input_begin_index=*/0);
  ValidateVectorOutput(range_1_packets, /*expected_elements=*/3,
                       /*input_begin_index=*/1);
  ValidateVectorOutput(range_2_packets, /*expected_elements=*/1,
                       /*input_begin_index=*/4);

  // Fully close the graph at the end.
  MP_ASSERT_OK(graph.CloseInputStream("tensor_in"));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST_F(SplitTfLiteTensorVectorCalculatorTest, InvalidRangeTest) {
  ASSERT_NE(interpreter_, nullptr);

  // Prepare a graph to use the SplitTfLiteTensorVectorCalculator.
  CalculatorGraphConfig graph_config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"(
            input_stream: "tensor_in"
            node {
              calculator: "SplitTfLiteTensorVectorCalculator"
              input_stream: "tensor_in"
              output_stream: "range_0"
              options {
                [mediapipe.SplitVectorCalculatorOptions.ext] {
                  ranges: { begin: 0 end: 0 }
                }
              }
            }
          )");

  // Run the graph.
  CalculatorGraph graph;
  // The graph should fail running because of an invalid range (begin == end).
  ASSERT_FALSE(graph.Initialize(graph_config).ok());
}

TEST_F(SplitTfLiteTensorVectorCalculatorTest, InvalidOutputStreamCountTest) {
  ASSERT_NE(interpreter_, nullptr);

  // Prepare a graph to use the SplitTfLiteTensorVectorCalculator.
  CalculatorGraphConfig graph_config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"(
            input_stream: "tensor_in"
            node {
              calculator: "SplitTfLiteTensorVectorCalculator"
              input_stream: "tensor_in"
              output_stream: "range_0"
              output_stream: "range_1"
              options {
                [mediapipe.SplitVectorCalculatorOptions.ext] {
                  ranges: { begin: 0 end: 1 }
                }
              }
            }
          )");

  // Run the graph.
  CalculatorGraph graph;
  // The graph should fail running because the number of output streams does not
  // match the number of range elements in the options.
  ASSERT_FALSE(graph.Initialize(graph_config).ok());
}

TEST_F(SplitTfLiteTensorVectorCalculatorTest, SmokeTestElementOnly) {
  ASSERT_NE(interpreter_, nullptr);

  PrepareTfLiteTensorVector(/*vector_size=*/5);
  ASSERT_NE(input_vec_, nullptr);

  // Prepare a graph to use the SplitTfLiteTensorVectorCalculator.
  CalculatorGraphConfig graph_config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"(
            input_stream: "tensor_in"
            node {
              calculator: "SplitTfLiteTensorVectorCalculator"
              input_stream: "tensor_in"
              output_stream: "range_0"
              output_stream: "range_1"
              output_stream: "range_2"
              options {
                [mediapipe.SplitVectorCalculatorOptions.ext] {
                  ranges: { begin: 0 end: 1 }
                  ranges: { begin: 2 end: 3 }
                  ranges: { begin: 4 end: 5 }
                  element_only: true
                }
              }
            }
          )");
  std::vector<Packet> range_0_packets;
  tool::AddVectorSink("range_0", &graph_config, &range_0_packets);
  std::vector<Packet> range_1_packets;
  tool::AddVectorSink("range_1", &graph_config, &range_1_packets);
  std::vector<Packet> range_2_packets;
  tool::AddVectorSink("range_2", &graph_config, &range_2_packets);

  // Run the graph.
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "tensor_in", Adopt(input_vec_.release()).At(Timestamp(0))));
  // Wait until the calculator finishes processing.
  MP_ASSERT_OK(graph.WaitUntilIdle());

  ValidateElementOutput(range_0_packets,
                        /*input_begin_index=*/0);
  ValidateElementOutput(range_1_packets,
                        /*input_begin_index=*/2);
  ValidateElementOutput(range_2_packets,
                        /*input_begin_index=*/4);

  // Fully close the graph at the end.
  MP_ASSERT_OK(graph.CloseInputStream("tensor_in"));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST_F(SplitTfLiteTensorVectorCalculatorTest,
       ElementOnlyDisablesVectorOutputs) {
  // Prepare a graph to use the SplitTfLiteTensorVectorCalculator.
  CalculatorGraphConfig graph_config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"(
            input_stream: "tensor_in"
            node {
              calculator: "SplitTfLiteTensorVectorCalculator"
              input_stream: "tensor_in"
              output_stream: "range_0"
              output_stream: "range_1"
              output_stream: "range_2"
              options {
                [mediapipe.SplitVectorCalculatorOptions.ext] {
                  ranges: { begin: 0 end: 1 }
                  ranges: { begin: 1 end: 4 }
                  ranges: { begin: 4 end: 5 }
                  element_only: true
                }
              }
            }
          )");

  // Run the graph.
  CalculatorGraph graph;
  ASSERT_FALSE(graph.Initialize(graph_config).ok());
}

}  // namespace mediapipe
