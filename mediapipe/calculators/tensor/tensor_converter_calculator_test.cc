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

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/memory_manager_service.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"  // NOLINT
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"  // NOLINT
#include "mediapipe/framework/tool/validate_type.h"

namespace mediapipe {
using RandomEngine = std::mt19937_64;
using ::testing::HasSubstr;
const uint32_t kSeed = 1234;
const int kNumSizes = 8;
const int sizes[kNumSizes][2] = {{1, 1}, {12, 1}, {1, 9},   {2, 2},
                                 {5, 3}, {7, 13}, {16, 32}, {101, 2}};

class TensorConverterCalculatorTest : public ::testing::Test {
 protected:
  // Adds a packet with a matrix filled with random values in [0,1].
  void AddRandomMatrix(int num_rows, int num_columns, uint32_t seed,
                       bool row_major_matrix = false) {
    RandomEngine random(kSeed);
    std::uniform_real_distribution<> uniform_dist(0, 1.0);
    auto matrix = std::make_unique<Matrix>();
    matrix->resize(num_rows, num_columns);
    if (row_major_matrix) {
      for (int y = 0; y < num_rows; ++y) {
        for (int x = 0; x < num_columns; ++x) {
          float value = uniform_dist(random);
          (*matrix)(y, x) = value;
        }
      }
    } else {
      for (int x = 0; x < num_columns; ++x) {
        for (int y = 0; y < num_rows; ++y) {
          float value = uniform_dist(random);
          (*matrix)(y, x) = value;
        }
      }
    }
    MP_ASSERT_OK(graph_->AddPacketToInputStream(
        "matrix", Adopt(matrix.release()).At(Timestamp(0))));
  }

  std::unique_ptr<CalculatorGraph> graph_;
};

TEST_F(TensorConverterCalculatorTest, RandomMatrixColMajor) {
  for (int size_index = 0; size_index < kNumSizes; ++size_index) {
    const int num_rows = sizes[size_index][0];
    const int num_columns = sizes[size_index][1];

    // Run the calculator and verify that one output is generated.
    CalculatorGraphConfig graph_config =
        mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
          input_stream: "matrix"
          node {
            calculator: "TensorConverterCalculator"
            input_stream: "MATRIX:matrix"
            output_stream: "TENSORS:tensor"
            options {
              [mediapipe.TensorConverterCalculatorOptions.ext] {
                row_major_matrix: false
              }
            }
          }
        )pb");
    std::vector<Packet> output_packets;
    tool::AddVectorSink("tensor", &graph_config, &output_packets);

    // Run the graph.
    graph_ = std::make_unique<CalculatorGraph>();
    MP_ASSERT_OK(graph_->Initialize(graph_config));
    MP_ASSERT_OK(graph_->StartRun({}));

    // Push the tensor into the graph.
    AddRandomMatrix(num_rows, num_columns, kSeed, /*row_major_matrix=*/false);

    // Wait until the calculator done processing.
    MP_ASSERT_OK(graph_->WaitUntilIdle());
    ASSERT_EQ(output_packets.size(), 1);

    // Get and process results.
    const std::vector<Tensor>& tensor_vec =
        output_packets[0].Get<std::vector<Tensor>>();
    ASSERT_EQ(tensor_vec.size(), 1);

    const Tensor* tensor = &tensor_vec[0];
    EXPECT_EQ(Tensor::ElementType::kFloat32, tensor->element_type());

    // Verify that the data is correct.
    RandomEngine random(kSeed);
    std::uniform_real_distribution<> uniform_dist(0, 1.0);
    auto view = tensor->GetCpuReadView();
    auto tensor_buffer = view.buffer<float>();
    for (int i = 0; i < num_rows * num_columns; ++i) {
      const float expected = uniform_dist(random);
      EXPECT_FLOAT_EQ(tensor_buffer[i], expected) << "at i = " << i;
    }

    // Fully close graph at end, otherwise calculator+tensors are destroyed
    // after calling WaitUntilDone().
    MP_ASSERT_OK(graph_->CloseInputStream("matrix"));
    MP_ASSERT_OK(graph_->WaitUntilDone());

    graph_.reset();
  }
}

TEST_F(TensorConverterCalculatorTest, RandomMatrixRowMajor) {
  for (int size_index = 0; size_index < kNumSizes; ++size_index) {
    const int num_rows = sizes[size_index][0];
    const int num_columns = sizes[size_index][1];

    // Run the calculator and verify that one output is generated.
    CalculatorGraphConfig graph_config =
        mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
          input_stream: "matrix"
          node {
            calculator: "TensorConverterCalculator"
            input_stream: "MATRIX:matrix"
            output_stream: "TENSORS:tensor"
            options {
              [mediapipe.TensorConverterCalculatorOptions.ext] {
                row_major_matrix: true
              }
            }
          }
        )pb");
    std::vector<Packet> output_packets;
    tool::AddVectorSink("tensor", &graph_config, &output_packets);

    // Run the graph.
    graph_ = std::make_unique<CalculatorGraph>();
    MP_ASSERT_OK(graph_->Initialize(graph_config));
    MP_ASSERT_OK(graph_->StartRun({}));

    // Push the tensor into the graph.
    AddRandomMatrix(num_rows, num_columns, kSeed, /*row_major_matrix=*/true);

    // Wait until the calculator done processing.
    MP_ASSERT_OK(graph_->WaitUntilIdle());
    ASSERT_EQ(output_packets.size(), 1);

    // Get and process results.
    const std::vector<Tensor>& tensor_vec =
        output_packets[0].Get<std::vector<Tensor>>();
    ASSERT_EQ(tensor_vec.size(), 1);

    const Tensor* tensor = &tensor_vec[0];
    EXPECT_EQ(Tensor::ElementType::kFloat32, tensor->element_type());

    // Verify that the data is correct.
    RandomEngine random(kSeed);
    std::uniform_real_distribution<> uniform_dist(0, 1.0);
    auto view = tensor->GetCpuReadView();
    auto tensor_buffer = view.buffer<float>();
    for (int i = 0; i < num_rows * num_columns; ++i) {
      const float expected = uniform_dist(random);
      EXPECT_EQ(tensor_buffer[i], expected) << "at i = " << i;
    }

    // Fully close graph at end, otherwise calculator+tensors are destroyed
    // after calling WaitUntilDone().
    MP_ASSERT_OK(graph_->CloseInputStream("matrix"));
    MP_ASSERT_OK(graph_->WaitUntilDone());

    graph_.reset();
  }
}

TEST_F(TensorConverterCalculatorTest, CustomDivAndSub) {
  CalculatorGraph graph;
  // Run the calculator and verify that one output is generated.
  CalculatorGraphConfig graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input_image"
        node {
          calculator: "TensorConverterCalculator"
          input_stream: "IMAGE:input_image"
          output_stream: "TENSORS:tensor"
          options {
            [mediapipe.TensorConverterCalculatorOptions.ext] {
              row_major_matrix: true
              use_custom_normalization: true
              custom_div: 2.0
              custom_sub: 33.0
            }
          }
        }
      )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("tensor", &graph_config, &output_packets);

  // Run the graph.
  MP_ASSERT_OK(graph.SetServiceObject(kMemoryManagerService,
                                      std::make_shared<MemoryManager>()));
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  auto input_image = std::make_unique<ImageFrame>(ImageFormat::GRAY8, 1, 1);
  cv::Mat mat = mediapipe::formats::MatView(input_image.get());
  mat.at<uint8_t>(0, 0) = 200;
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_image", Adopt(input_image.release()).At(Timestamp(0))));

  // Wait until the calculator done processing.
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Get and process results.
  const std::vector<Tensor>& tensor_vec =
      output_packets[0].Get<std::vector<Tensor>>();
  ASSERT_EQ(tensor_vec.size(), 1);

  const Tensor* tensor = &tensor_vec[0];
  EXPECT_EQ(Tensor::ElementType::kFloat32, tensor->element_type());
  auto view = tensor->GetCpuReadView();
  EXPECT_FLOAT_EQ(*view.buffer<float>(), 67.0f);

  // Fully close graph at end, otherwise calculator+tensors are destroyed
  // after calling WaitUntilDone().
  MP_ASSERT_OK(graph.CloseInputStream("input_image"));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST_F(TensorConverterCalculatorTest, SetOutputRange) {
  std::vector<std::pair<float, float>> range_values = {
      std::make_pair(0.0, 1.0), std::make_pair(-1.0, 1.0),
      std::make_pair(-0.5, 0.5)};
  for (std::pair<float, float> range : range_values) {
    CalculatorGraph graph;
    CalculatorGraphConfig graph_config =
        mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
            R"pb(
              input_stream: "input_image"
              node {
                calculator: "TensorConverterCalculator"
                input_stream: "IMAGE:input_image"
                output_stream: "TENSORS:tensor"
                options {
                  [mediapipe.TensorConverterCalculatorOptions.ext] {
                    output_tensor_float_range { min: $0 max: $1 }
                  }
                }
              }
            )pb",
            /*$0=*/range.first,
            /*$1=*/range.second));
    std::vector<Packet> output_packets;
    tool::AddVectorSink("tensor", &graph_config, &output_packets);

    // Run the graph.
    MP_ASSERT_OK(graph.Initialize(graph_config));
    MP_ASSERT_OK(graph.StartRun({}));
    auto input_image = std::make_unique<ImageFrame>(ImageFormat::GRAY8, 1, 1);
    cv::Mat mat = mediapipe::formats::MatView(input_image.get());
    mat.at<uint8_t>(0, 0) = 200;
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "input_image", Adopt(input_image.release()).At(Timestamp(0))));

    // Wait until the calculator finishes processing.
    MP_ASSERT_OK(graph.WaitUntilIdle());
    ASSERT_EQ(output_packets.size(), 1);

    // Get and process results.
    const std::vector<Tensor>& tensor_vec =
        output_packets[0].Get<std::vector<Tensor>>();
    ASSERT_EQ(tensor_vec.size(), 1);

    const Tensor* tensor = &tensor_vec[0];

    // Calculate the expected normalized value:
    float expected_value =
        range.first + (200 * (range.second - range.first)) / 255.0;

    EXPECT_EQ(tensor->element_type(), Tensor::ElementType::kFloat32);
    auto view = tensor->GetCpuReadView();
    float actual_value = *view.buffer<float>();
    EXPECT_FLOAT_EQ(actual_value, expected_value);

    // Fully close graph at end, otherwise calculator+tensors are destroyed
    // after calling WaitUntilDone().
    MP_ASSERT_OK(graph.CloseInputStream("input_image"));
    MP_ASSERT_OK(graph.WaitUntilDone());
  }
}

TEST_F(TensorConverterCalculatorTest,
       ShouldConvertImageWithDefaultOutputRange) {
  CalculatorGraph graph;
  CalculatorGraphConfig graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            input_stream: "input_image"
            node {
              calculator: "TensorConverterCalculator"
              input_stream: "IMAGE:input_image"
              output_stream: "TENSORS:tensor"
              options {
                [mediapipe.TensorConverterCalculatorOptions.ext] {
                  zero_center: false
                }
              }
            }
          )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("tensor", &graph_config, &output_packets);

  // Run the graph.
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  auto input_image = std::make_unique<ImageFrame>(ImageFormat::GRAY8, 1, 1);
  cv::Mat mat = mediapipe::formats::MatView(input_image.get());
  mat.at<uint8_t>(0, 0) = 200;
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_image", Adopt(input_image.release()).At(Timestamp(0))));

  // Wait until the calculator finishes processing.
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_packets.size(), 1);

  // Get and process results.
  const std::vector<Tensor>& tensor_vec =
      output_packets[0].Get<std::vector<Tensor>>();
  ASSERT_EQ(tensor_vec.size(), 1);

  const Tensor* tensor = &tensor_vec[0];

  // Calculate the expected normalized value:
  float expected_value = 200.0 / 255.0;

  EXPECT_EQ(tensor->element_type(), Tensor::ElementType::kFloat32);
  auto view = tensor->GetCpuReadView();
  float actual_value = *view.buffer<float>();
  EXPECT_FLOAT_EQ(actual_value, expected_value);

  // Fully close graph at end, otherwise calculator+tensors are destroyed
  // after calling WaitUntilDone().
  MP_ASSERT_OK(graph.CloseInputStream("input_image"));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST_F(TensorConverterCalculatorTest, FlipVertically) {
  CalculatorGraph graph;
  CalculatorGraphConfig graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input_image"
        node {
          calculator: "TensorConverterCalculator"
          input_stream: "IMAGE:input_image"
          output_stream: "TENSORS:tensor"
          options {
            [mediapipe.TensorConverterCalculatorOptions.ext] {
              flip_vertically: true
              output_tensor_float_range { min: 0 max: 255 }
            }
          }
        }
      )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("tensor", &graph_config, &output_packets);

  // Run the graph.
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  auto input_image = std::make_unique<ImageFrame>(ImageFormat::GRAY8, 1, 2);
  cv::Mat mat = mediapipe::formats::MatView(input_image.get());
  constexpr uint8_t kY0Value = 100;
  constexpr uint8_t kY1Value = 200;
  mat.at<uint8_t>(0, 0) = kY0Value;
  mat.at<uint8_t>(1, 0) = kY1Value;  // Note: y, x!
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_image", Adopt(input_image.release()).At(Timestamp(0))));

  // Wait until the calculator finishes processing.
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_packets.size(), 1);

  // Get and process results.
  const std::vector<Tensor>& tensor_vec =
      output_packets[0].Get<std::vector<Tensor>>();
  ASSERT_EQ(tensor_vec.size(), 1);

  const Tensor* tensor = &tensor_vec[0];

  EXPECT_EQ(tensor->element_type(), Tensor::ElementType::kFloat32);
  const float* dataf = tensor->GetCpuReadView().buffer<float>();
  EXPECT_EQ(static_cast<int>(roundf(dataf[0])), kY1Value);  // Y0, Y1 flipped!
  EXPECT_EQ(static_cast<int>(roundf(dataf[1])), kY0Value);

  // Fully close graph at end, otherwise calculator+tensors are destroyed
  // after calling WaitUntilDone().
  MP_ASSERT_OK(graph.CloseInputStream("input_image"));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST_F(TensorConverterCalculatorTest,
       CannotSpecifyBothFlipVerticallyAndGpuOrigin) {
  CalculatorGraph graph;
  CalculatorGraphConfig graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input_image"
        node {
          calculator: "TensorConverterCalculator"
          input_stream: "IMAGE:input_image"
          output_stream: "TENSORS:tensor"
          options {
            [mediapipe.TensorConverterCalculatorOptions.ext] {
              flip_vertically: true
              gpu_origin: TOP_LEFT
              output_tensor_float_range { min: 0 max: 255 }
            }
          }
        }
      )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("tensor", &graph_config, &output_packets);

  // Run the graph.
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  // Processing should fail as we specified both flip_vertically and gpu_origin.
  EXPECT_THAT(
      graph.WaitUntilIdle(),
      StatusIs(
          absl::StatusCode::kFailedPrecondition,
          HasSubstr(
              "Cannot specify both flip_vertically and gpu_origin options")));
}

TEST_F(TensorConverterCalculatorTest, GpuOriginIsIgnoredWithCpuImage) {
  CalculatorGraph graph;
  CalculatorGraphConfig graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input_image"
        node {
          calculator: "TensorConverterCalculator"
          input_stream: "IMAGE:input_image"
          output_stream: "TENSORS:tensor"
          options {
            [mediapipe.TensorConverterCalculatorOptions.ext] {
              gpu_origin: CONVENTIONAL
              output_tensor_float_range { min: 0 max: 255 }
            }
          }
        }
      )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("tensor", &graph_config, &output_packets);

  // Run the graph.
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  auto input_image = std::make_unique<ImageFrame>(ImageFormat::GRAY8, 1, 2);
  cv::Mat mat = mediapipe::formats::MatView(input_image.get());
  constexpr uint8_t kY0Value = 100;
  constexpr uint8_t kY1Value = 200;
  mat.at<uint8_t>(0, 0) = kY0Value;
  mat.at<uint8_t>(1, 0) = kY1Value;  // Note: y, x!
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_image", Adopt(input_image.release()).At(Timestamp(0))));

  // Wait until the calculator finishes processing.
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_packets.size(), 1);

  // Get and process results.
  const std::vector<Tensor>& tensor_vec =
      output_packets[0].Get<std::vector<Tensor>>();
  ASSERT_EQ(tensor_vec.size(), 1);

  const Tensor* tensor = &tensor_vec[0];

  EXPECT_EQ(tensor->element_type(), Tensor::ElementType::kFloat32);
  const float* dataf = tensor->GetCpuReadView().buffer<float>();
  EXPECT_EQ(static_cast<int>(roundf(dataf[0])), kY0Value);  // Not flipped!
  EXPECT_EQ(static_cast<int>(roundf(dataf[1])), kY1Value);

  // Fully close graph at end, otherwise calculator+tensors are destroyed
  // after calling WaitUntilDone().
  MP_ASSERT_OK(graph.CloseInputStream("input_image"));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

}  // namespace mediapipe
