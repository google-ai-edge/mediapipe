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

#include <random>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "mediapipe/calculators/tflite/tflite_converter_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"  // NOLINT
#include "mediapipe/framework/tool/validate_type.h"
#include "tensorflow/lite/interpreter.h"

namespace mediapipe {
namespace {

constexpr char kTransposeOptionsString[] =
    "[mediapipe.TfLiteConverterCalculatorOptions.ext]: {"
    "row_major_matrix: True}";

}  // namespace

using RandomEngine = std::mt19937_64;
using testing::Eq;
const uint32 kSeed = 1234;
const int kNumSizes = 8;
const int sizes[kNumSizes][2] = {{1, 1}, {12, 1}, {1, 9},   {2, 2},
                                 {5, 3}, {7, 13}, {16, 32}, {101, 2}};

class TfLiteConverterCalculatorTest : public ::testing::Test {
 protected:
  // Adds a packet with a matrix filled with random values in [0,1].
  void AddRandomMatrix(int num_rows, int num_columns, uint32 seed,
                       bool row_major_matrix = false) {
    RandomEngine random(kSeed);
    std::uniform_real_distribution<> uniform_dist(0, 1.0);
    auto matrix = ::absl::make_unique<Matrix>();
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

TEST_F(TfLiteConverterCalculatorTest, RandomMatrixColMajor) {
  for (int size_index = 0; size_index < kNumSizes; ++size_index) {
    const int num_rows = sizes[size_index][0];
    const int num_columns = sizes[size_index][1];

    // Run the calculator and verify that one output is generated.
    CalculatorGraphConfig graph_config =
        mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
          input_stream: "matrix"
          node {
            calculator: "TfLiteConverterCalculator"
            input_stream: "MATRIX:matrix"
            output_stream: "TENSORS:tensor"
            options {
              [mediapipe.TfLiteConverterCalculatorOptions.ext] {
                row_major_matrix: false
              }
            }
          }
        )pb");
    std::vector<Packet> output_packets;
    tool::AddVectorSink("tensor", &graph_config, &output_packets);

    // Run the graph.
    graph_ = absl::make_unique<CalculatorGraph>();
    MP_ASSERT_OK(graph_->Initialize(graph_config));
    MP_ASSERT_OK(graph_->StartRun({}));

    // Push the tensor into the graph.
    AddRandomMatrix(num_rows, num_columns, kSeed, /*row_major_matrix=*/false);

    // Wait until the calculator done processing.
    MP_ASSERT_OK(graph_->WaitUntilIdle());
    EXPECT_EQ(1, output_packets.size());

    // Get and process results.
    const std::vector<TfLiteTensor>& tensor_vec =
        output_packets[0].Get<std::vector<TfLiteTensor>>();
    EXPECT_EQ(1, tensor_vec.size());

    const TfLiteTensor* tensor = &tensor_vec[0];
    EXPECT_EQ(kTfLiteFloat32, tensor->type);

    // Verify that the data is correct.
    RandomEngine random(kSeed);
    std::uniform_real_distribution<> uniform_dist(0, 1.0);
    const float* tensor_buffer = tensor->data.f;
    for (int i = 0; i < num_rows * num_columns; ++i) {
      const float expected = uniform_dist(random);
      EXPECT_EQ(expected, tensor_buffer[i]) << "at i = " << i;
    }

    // Fully close graph at end, otherwise calculator+tensors are destroyed
    // after calling WaitUntilDone().
    MP_ASSERT_OK(graph_->CloseInputStream("matrix"));
    MP_ASSERT_OK(graph_->WaitUntilDone());

    graph_.reset();
  }
}

TEST_F(TfLiteConverterCalculatorTest, RandomMatrixRowMajor) {
  for (int size_index = 0; size_index < kNumSizes; ++size_index) {
    const int num_rows = sizes[size_index][0];
    const int num_columns = sizes[size_index][1];

    // Run the calculator and verify that one output is generated.
    CalculatorGraphConfig graph_config =
        mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
          input_stream: "matrix"
          node {
            calculator: "TfLiteConverterCalculator"
            input_stream: "MATRIX:matrix"
            output_stream: "TENSORS:tensor"
            options {
              [mediapipe.TfLiteConverterCalculatorOptions.ext] {
                row_major_matrix: true
              }
            }
          }
        )pb");
    std::vector<Packet> output_packets;
    tool::AddVectorSink("tensor", &graph_config, &output_packets);

    // Run the graph.
    graph_ = absl::make_unique<CalculatorGraph>();
    MP_ASSERT_OK(graph_->Initialize(graph_config));
    MP_ASSERT_OK(graph_->StartRun({}));

    // Push the tensor into the graph.
    AddRandomMatrix(num_rows, num_columns, kSeed, /*row_major_matrix=*/true);

    // Wait until the calculator done processing.
    MP_ASSERT_OK(graph_->WaitUntilIdle());
    EXPECT_EQ(1, output_packets.size());

    // Get and process results.
    const std::vector<TfLiteTensor>& tensor_vec =
        output_packets[0].Get<std::vector<TfLiteTensor>>();
    EXPECT_EQ(1, tensor_vec.size());

    const TfLiteTensor* tensor = &tensor_vec[0];
    EXPECT_EQ(kTfLiteFloat32, tensor->type);

    // Verify that the data is correct.
    RandomEngine random(kSeed);
    std::uniform_real_distribution<> uniform_dist(0, 1.0);
    const float* tensor_buffer = tensor->data.f;
    for (int i = 0; i < num_rows * num_columns; ++i) {
      const float expected = uniform_dist(random);
      EXPECT_EQ(expected, tensor_buffer[i]) << "at i = " << i;
    }

    // Fully close graph at end, otherwise calculator+tensors are destroyed
    // after calling WaitUntilDone().
    MP_ASSERT_OK(graph_->CloseInputStream("matrix"));
    MP_ASSERT_OK(graph_->WaitUntilDone());

    graph_.reset();
  }
}

TEST_F(TfLiteConverterCalculatorTest, CustomDivAndSub) {
  CalculatorGraph graph;
  // Run the calculator and verify that one output is generated.
  CalculatorGraphConfig graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input_image"
        node {
          calculator: "TfLiteConverterCalculator"
          input_stream: "IMAGE:input_image"
          output_stream: "TENSORS:tensor"
          options {
            [mediapipe.TfLiteConverterCalculatorOptions.ext] {
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
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  auto input_image = absl::make_unique<ImageFrame>(ImageFormat::GRAY8, 1, 1);
  cv::Mat mat = mediapipe::formats::MatView(input_image.get());
  mat.at<uint8>(0, 0) = 200;
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_image", Adopt(input_image.release()).At(Timestamp(0))));

  // Wait until the calculator done processing.
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Get and process results.
  const std::vector<TfLiteTensor>& tensor_vec =
      output_packets[0].Get<std::vector<TfLiteTensor>>();
  EXPECT_EQ(1, tensor_vec.size());

  const TfLiteTensor* tensor = &tensor_vec[0];
  EXPECT_EQ(kTfLiteFloat32, tensor->type);
  EXPECT_FLOAT_EQ(67.0f, *tensor->data.f);

  // Fully close graph at end, otherwise calculator+tensors are destroyed
  // after calling WaitUntilDone().
  MP_ASSERT_OK(graph.CloseInputStream("input_image"));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST_F(TfLiteConverterCalculatorTest, SetOutputRange) {
  std::vector<std::pair<float, float>> range_values = {
      std::make_pair(0.0, 1.0), std::make_pair(-1.0, 1.0),
      std::make_pair(-0.5, 0.5)};
  for (std::pair<float, float> range : range_values) {
    CalculatorGraph graph;
    CalculatorGraphConfig graph_config =
        mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
            absl::Substitute(R"(
        input_stream: "input_image"
        node {
          calculator: "TfLiteConverterCalculator"
          input_stream: "IMAGE:input_image"
          output_stream: "TENSORS:tensor"
          options {
            [mediapipe.TfLiteConverterCalculatorOptions.ext] {
              output_tensor_float_range {
                min: $0
                max: $1
              }
            }
          }
        }
        )",
                             /*$0=*/range.first,
                             /*$1=*/range.second));
    std::vector<Packet> output_packets;
    tool::AddVectorSink("tensor", &graph_config, &output_packets);

    // Run the graph.
    MP_ASSERT_OK(graph.Initialize(graph_config));
    MP_ASSERT_OK(graph.StartRun({}));
    auto input_image = absl::make_unique<ImageFrame>(ImageFormat::GRAY8, 1, 1);
    cv::Mat mat = mediapipe::formats::MatView(input_image.get());
    mat.at<uint8>(0, 0) = 200;
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "input_image", Adopt(input_image.release()).At(Timestamp(0))));

    // Wait until the calculator finishes processing.
    MP_ASSERT_OK(graph.WaitUntilIdle());
    EXPECT_THAT(output_packets.size(), Eq(1));

    // Get and process results.
    const std::vector<TfLiteTensor>& tensor_vec =
        output_packets[0].Get<std::vector<TfLiteTensor>>();
    EXPECT_THAT(tensor_vec.size(), Eq(1));

    const TfLiteTensor* tensor = &tensor_vec[0];

    // Calculate the expected normalized value:
    float normalized_value =
        range.first + (200 * (range.second - range.first)) / 255.0;

    EXPECT_THAT(tensor->type, Eq(kTfLiteFloat32));
    EXPECT_THAT(normalized_value,
                testing::FloatNear(*tensor->data.f,
                                   2.0f * std::abs(*tensor->data.f) *
                                       std::numeric_limits<float>::epsilon()));

    // Fully close graph at end, otherwise calculator+tensors are destroyed
    // after calling WaitUntilDone().
    MP_ASSERT_OK(graph.CloseInputStream("input_image"));
    MP_ASSERT_OK(graph.WaitUntilDone());
  }
}

}  // namespace mediapipe
