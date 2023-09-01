// Copyright 2022 The MediaPipe Authors.
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

#include <functional>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "mediapipe/calculators/tensor/feedback_tensors_calculator.pb.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace {

using ::mediapipe::CalculatorGraphConfig;
using ::testing::ElementsAreArray;
using ::testing::Not;
using Tensors = std::vector<Tensor>;

template <typename T>
struct TensorElementType {
  static constexpr Tensor::ElementType value = Tensor::ElementType::kNone;
};

template <>
struct TensorElementType<float> {
  static constexpr Tensor::ElementType value = Tensor::ElementType::kFloat32;
};

template <>
struct TensorElementType<std::int8_t> {
  static constexpr Tensor::ElementType value = Tensor::ElementType::kInt8;
};

template <>
struct TensorElementType<std::uint8_t> {
  static constexpr Tensor::ElementType value = Tensor::ElementType::kUInt8;
};

template <>
struct TensorElementType<std::int32_t> {
  static constexpr Tensor::ElementType value = Tensor::ElementType::kInt32;
};

template <typename T>
Tensor MakeTensor(std::initializer_list<int> shape,
                  std::initializer_list<T> values) {
  Tensor tensor(TensorElementType<T>::value, shape);
  ABSL_CHECK_EQ(values.size(), tensor.shape().num_elements())
      << "The size of `values` is incompatible with `shape`";
  absl::c_copy(values, tensor.GetCpuWriteView().buffer<T>());
  return tensor;
}

template <typename T>
void ValidateTensor(const Tensor& tensor,
                    const std::vector<int>& expected_shape,
                    const std::vector<T>& expected_values) {
  ASSERT_EQ(tensor.element_type(), TensorElementType<T>::value);
  EXPECT_EQ(tensor.shape().dims, expected_shape);
  EXPECT_EQ(tensor.shape().num_elements(), expected_values.size());

  auto* tensor_buffer = tensor.GetCpuReadView().buffer<T>();
  const std::vector<T> tensor_values(
      tensor_buffer, tensor_buffer + tensor.shape().num_elements());
  EXPECT_THAT(tensor_values, ElementsAreArray(expected_values));
}

TEST(FeedbackTensorsCalculatorTest, AppendsFeedback) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "input"
    input_stream: "feedback"
    node {
      calculator: "FeedbackTensorsCalculator"
      input_stream: "INPUT_TENSORS:input"
      input_stream: "FEEDBACK_TENSORS:feedback"
      output_stream: "TENSORS:output"
      options: {
        [mediapipe.FeedbackTensorsCalculatorOptions.ext] {
          feedback_tensor_shape: { dims: 2 dims: 3 }
          location: APPENDED
        }
      }
    }
  )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  auto initial_input_tensors = std::make_unique<Tensors>();
  initial_input_tensors->push_back(
      MakeTensor<std::int32_t>({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input", Adopt(initial_input_tensors.release()).At(Timestamp(1))));
  // At the beginning, the loopback packet with the model feedback is missing.
  // The calculator has to assume it's all-zero with the shape from the options.

  auto later_input_tensors = std::make_unique<Tensors>();
  later_input_tensors->push_back(
      MakeTensor<std::int32_t>({2, 4}, {8, 7, 6, 5, 4, 3, 2, 1}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input", Adopt(later_input_tensors.release()).At(Timestamp(2))));
  auto later_feedback_tensors = std::make_unique<Tensors>();
  later_feedback_tensors->push_back(
      MakeTensor({2, 3}, {-1.f, -2.f, -3.f, -4.f, -5.f, -6.f}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "feedback", Adopt(later_feedback_tensors.release()).At(Timestamp(2))));

  MP_ASSERT_OK(graph.CloseAllInputStreams())
      << "Couldn't close the graph inputs";
  MP_ASSERT_OK(graph.WaitUntilDone()) << "Couldn't finalize the graph run";

  ASSERT_EQ(output_packets.size(), 2);

  const Tensors& initial_combined_tensors = output_packets[0].Get<Tensors>();
  ASSERT_EQ(initial_combined_tensors.size(), 2);
  ValidateTensor<std::int32_t>(initial_combined_tensors[0],
                               /*expected_shape=*/{2, 4},
                               /*expected_values=*/{1, 2, 3, 4, 5, 6, 7, 8});
  // The initial feedback is zero.
  ValidateTensor<float>(initial_combined_tensors[1], /*expected_shape=*/{2, 3},
                        /*expected_values=*/{0.f, 0.f, 0.f, 0.f, 0.f, 0.f});

  const Tensors& later_combined_tensors = output_packets[1].Get<Tensors>();
  ASSERT_EQ(later_combined_tensors.size(), 2);
  ValidateTensor<std::int32_t>(later_combined_tensors[0],
                               /*expected_shape=*/{2, 4},
                               /*expected_values=*/{8, 7, 6, 5, 4, 3, 2, 1});
  // Afterwards, the provided feedback is passed through.
  ValidateTensor<float>(
      later_combined_tensors[1], /*expected_shape=*/{2, 3},
      /*expected_values=*/{-1.f, -2.f, -3.f, -4.f, -5.f, -6.f});
}

TEST(FeedbackTensorsCalculatorTest, PrependsFeedback) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "input"
    input_stream: "feedback"
    node {
      calculator: "FeedbackTensorsCalculator"
      input_stream: "INPUT_TENSORS:input"
      input_stream: "FEEDBACK_TENSORS:feedback"
      output_stream: "TENSORS:output"
      options: {
        [mediapipe.FeedbackTensorsCalculatorOptions.ext] {
          feedback_tensor_shape: { dims: 3 dims: 2 }
          location: PREPENDED
        }
      }
    }
  )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  auto initial_input_tensors = std::make_unique<Tensors>();
  initial_input_tensors->push_back(
      MakeTensor<std::int8_t>({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input", Adopt(initial_input_tensors.release()).At(Timestamp(1))));
  // At the beginning, the loopback packet with the model feedback is missing.
  // The calculator has to assume it's all-zero with the shape from the options.

  auto later_input_tensors = std::make_unique<Tensors>();
  later_input_tensors->push_back(
      MakeTensor<std::int8_t>({2, 4}, {8, 7, 6, 5, 4, 3, 2, 1}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input", Adopt(later_input_tensors.release()).At(Timestamp(2))));
  auto later_feedback_tensors = std::make_unique<Tensors>();
  later_feedback_tensors->push_back(
      MakeTensor({3, 2}, {-1.f, -2.f, -3.f, -4.f, -5.f, -6.f}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "feedback", Adopt(later_feedback_tensors.release()).At(Timestamp(2))));

  MP_ASSERT_OK(graph.CloseAllInputStreams())
      << "Couldn't close the graph inputs";
  MP_ASSERT_OK(graph.WaitUntilDone()) << "Couldn't finalize the graph run";

  ASSERT_EQ(output_packets.size(), 2);

  const Tensors& initial_combined_tensors = output_packets[0].Get<Tensors>();
  ASSERT_EQ(initial_combined_tensors.size(), 2);
  // The initial feedback is zero.
  ValidateTensor<float>(initial_combined_tensors[0], /*expected_shape=*/{3, 2},
                        /*expected_values=*/{0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
  ValidateTensor<std::int8_t>(initial_combined_tensors[1],
                              /*expected_shape=*/{2, 4},
                              /*expected_values=*/{1, 2, 3, 4, 5, 6, 7, 8});

  const Tensors& later_combined_tensors = output_packets[1].Get<Tensors>();
  ASSERT_EQ(later_combined_tensors.size(), 2);
  // Afterwards, the provided feedback is passed through.
  ValidateTensor<float>(
      later_combined_tensors[0], /*expected_shape=*/{3, 2},
      /*expected_values=*/{-1.f, -2.f, -3.f, -4.f, -5.f, -6.f});
  ValidateTensor<std::int8_t>(later_combined_tensors[1],
                              /*expected_shape=*/{2, 4},
                              /*expected_values=*/{8, 7, 6, 5, 4, 3, 2, 1});
}

TEST(FeedbackTensorsCalculatorTest, NoFeedback) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "input"
    input_stream: "feedback"
    node {
      calculator: "FeedbackTensorsCalculator"
      input_stream: "INPUT_TENSORS:input"
      input_stream: "FEEDBACK_TENSORS:feedback"
      output_stream: "TENSORS:output"
      options: {
        [mediapipe.FeedbackTensorsCalculatorOptions.ext] {
          feedback_tensor_shape: { dims: 3 dims: 4 }
          location: NONE
        }
      }
    }
  )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  auto initial_input_tensors = std::make_unique<Tensors>();
  initial_input_tensors->push_back(
      MakeTensor<std::uint8_t>({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input", Adopt(initial_input_tensors.release()).At(Timestamp(1))));
  // At the beginning, the loopback packet with the model feedback is missing.

  auto later_input_tensors = std::make_unique<Tensors>();
  later_input_tensors->push_back(
      MakeTensor<std::uint8_t>({2, 4}, {8, 7, 6, 5, 4, 3, 2, 1}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input", Adopt(later_input_tensors.release()).At(Timestamp(2))));
  // This feedback should be ignored due to `location: NONE`.
  auto later_feedback_tensors = std::make_unique<Tensors>();
  later_feedback_tensors->push_back(
      MakeTensor({2, 3}, {-1.f, -2.f, -3.f, -4.f, -5.f, -6.f}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "feedback", Adopt(later_feedback_tensors.release()).At(Timestamp(2))));

  MP_ASSERT_OK(graph.CloseAllInputStreams())
      << "Couldn't close the graph inputs";
  MP_ASSERT_OK(graph.WaitUntilDone()) << "Couldn't finalize the graph run";

  ASSERT_EQ(output_packets.size(), 2);

  const Tensors& initial_combined_tensors = output_packets[0].Get<Tensors>();
  ASSERT_EQ(initial_combined_tensors.size(), 1);
  ValidateTensor<std::uint8_t>(initial_combined_tensors[0],
                               /*expected_shape=*/{2, 4},
                               /*expected_values=*/{1, 2, 3, 4, 5, 6, 7, 8});
  // No feedback due to `location: NONE`.

  const Tensors& later_combined_tensors = output_packets[1].Get<Tensors>();
  ASSERT_EQ(later_combined_tensors.size(), 1);
  ValidateTensor<std::uint8_t>(later_combined_tensors[0],
                               /*expected_shape=*/{2, 4},
                               /*expected_values=*/{8, 7, 6, 5, 4, 3, 2, 1});
}

TEST(FeedbackTensorsCalculatorTest, ChecksTensorNumber) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "input"
    input_stream: "feedback"
    node {
      calculator: "FeedbackTensorsCalculator"
      input_stream: "INPUT_TENSORS:input"
      input_stream: "FEEDBACK_TENSORS:feedback"
      output_stream: "TENSORS:output"
      options: {
        [mediapipe.FeedbackTensorsCalculatorOptions.ext] {
          num_feedback_tensors: 2
          feedback_tensor_shape: { dims: 2 dims: 3 }
          location: PREPENDED
        }
      }
    }
  )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  auto initial_input_tensors = std::make_unique<Tensors>();
  initial_input_tensors->push_back(
      MakeTensor<std::uint8_t>({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input", Adopt(initial_input_tensors.release()).At(Timestamp(1))));
  // At the beginning, the loopback packet with the model feedback is missing.

  auto later_input_tensors = std::make_unique<Tensors>();
  later_input_tensors->push_back(
      MakeTensor<std::uint8_t>({2, 4}, {8, 7, 6, 5, 4, 3, 2, 1}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input", Adopt(later_input_tensors.release()).At(Timestamp(2))));
  // This feedback should be ignored due to `location: NONE`.
  auto later_feedback_tensors = std::make_unique<Tensors>();
  later_feedback_tensors->push_back(
      MakeTensor({2, 3}, {-1.f, -2.f, -3.f, -4.f, -5.f, -6.f}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "feedback", Adopt(later_feedback_tensors.release()).At(Timestamp(2))));

  MP_ASSERT_OK(graph.CloseAllInputStreams())
      << "Couldn't close the graph inputs";
  EXPECT_THAT(graph.WaitUntilDone(), Not(IsOk()))
      << "Tensor number mismatch missed";
}

TEST(FeedbackTensorsCalculatorTest, ChecksShape) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "input"
    input_stream: "feedback"
    node {
      calculator: "FeedbackTensorsCalculator"
      input_stream: "INPUT_TENSORS:input"
      input_stream: "FEEDBACK_TENSORS:feedback"
      output_stream: "TENSORS:output"
      options: {
        [mediapipe.FeedbackTensorsCalculatorOptions.ext] {
          feedback_tensor_shape: { dims: 3 dims: 4 }
          location: APPENDED
        }
      }
    }
  )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("output", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  auto initial_input_tensors = std::make_unique<Tensors>();
  initial_input_tensors->push_back(
      MakeTensor<std::uint8_t>({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input", Adopt(initial_input_tensors.release()).At(Timestamp(1))));
  // At the beginning, the loopback packet with the model feedback is missing.

  auto later_input_tensors = std::make_unique<Tensors>();
  later_input_tensors->push_back(
      MakeTensor<std::uint8_t>({2, 4}, {8, 7, 6, 5, 4, 3, 2, 1}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input", Adopt(later_input_tensors.release()).At(Timestamp(2))));
  // This feedback should be ignored due to `location: NONE`.
  auto later_feedback_tensors = std::make_unique<Tensors>();
  later_feedback_tensors->push_back(
      MakeTensor({2, 3}, {-1.f, -2.f, -3.f, -4.f, -5.f, -6.f}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "feedback", Adopt(later_feedback_tensors.release()).At(Timestamp(2))));

  MP_ASSERT_OK(graph.CloseAllInputStreams())
      << "Couldn't close the graph inputs";
  EXPECT_THAT(graph.WaitUntilDone(), Not(IsOk()))
      << "Tensor shape mismatch missed";
}

}  // namespace
}  // namespace mediapipe
