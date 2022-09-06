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

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using ::mediapipe::ParseTextProtoOrDie;
using ::testing::HasSubstr;
using Node = ::mediapipe::CalculatorGraphConfig::Node;

constexpr char kCalculatorConfig[] = R"pb(
  calculator: "TensorsDequantizationCalculator"
  input_stream: "TENSORS:input"
  output_stream: "TENSORS:output"
)pb";

// Compares the provided tensor contents with the expected values.
void ValidateResult(const Tensor& actual, const std::vector<float>& expected) {
  EXPECT_EQ(actual.element_type(), Tensor::ElementType::kFloat32);
  EXPECT_EQ(expected.size(), actual.shape().num_elements());
  auto view = actual.GetCpuReadView();
  auto buffer = view.buffer<float>();
  for (int i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(expected[i], buffer[i]);
  }
}

class TensorsDequantizationCalculatorTest : public ::testing::Test {
 protected:
  TensorsDequantizationCalculatorTest()
      : runner_(ParseTextProtoOrDie<Node>(kCalculatorConfig)) {}

  template <typename T>
  void PushTensor(Tensor::ElementType type, std::vector<T> tensor,
                  std::optional<Tensor::QuantizationParameters>
                      quantization_params = std::nullopt) {
    auto tensors = std::make_unique<std::vector<Tensor>>();
    if (quantization_params.has_value()) {
      tensors->emplace_back(type,
                            Tensor::Shape{static_cast<int>(tensor.size())},
                            quantization_params.value());
    } else {
      tensors->emplace_back(type,
                            Tensor::Shape{static_cast<int>(tensor.size())});
    }
    auto view = tensors->back().GetCpuWriteView();
    auto buffer = view.buffer<T>();
    std::copy(tensor.begin(), tensor.end(), buffer);
    runner_.MutableInputs()->Tag("TENSORS").packets.push_back(
        Adopt(tensors.release()).At(Timestamp(0)));
  }

  const Tensor& GetOutput() {
    return runner_.Outputs()
        .Get("TENSORS", 0)
        .packets[0]
        .Get<std::vector<Tensor>>()[0];
  }

  CalculatorRunner runner_;
};

TEST_F(TensorsDequantizationCalculatorTest, FailsWithFloatTensors) {
  std::vector<float> tensor = {0, 1};
  PushTensor(Tensor::ElementType::kFloat32, tensor);

  auto status = runner_.Run();

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("Unsupported input tensor type"));
}

TEST_F(TensorsDequantizationCalculatorTest, FailsWithInt32Tensors) {
  std::vector<int32_t> tensor = {0, 1};
  PushTensor(Tensor::ElementType::kInt32, tensor);

  auto status = runner_.Run();

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("Unsupported input tensor type"));
}

TEST_F(TensorsDequantizationCalculatorTest, SucceedsWithUInt8Tensors) {
  std::vector<uint8_t> tensor = {0, 127, 255};
  PushTensor(Tensor::ElementType::kUInt8, tensor,
             Tensor::QuantizationParameters{1.0f / 127, 127});

  MP_ASSERT_OK(runner_.Run());

  ValidateResult(GetOutput(), {-1, 0, 1.007874});
}

TEST_F(TensorsDequantizationCalculatorTest, SucceedsWithInt8Tensors) {
  std::vector<int8_t> tensor = {-128, 0, 127};
  PushTensor(Tensor::ElementType::kInt8, tensor,
             Tensor::QuantizationParameters{1.0f / 127, 0});

  MP_ASSERT_OK(runner_.Run());

  ValidateResult(GetOutput(), {-1.007874, 0, 1});
}

}  // namespace
}  // namespace mediapipe
