/* Copyright 2024 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"

using ::mediapipe::Packet;
using ::mediapipe::Tensor;
using ::mediapipe::Timestamp;
using ::testing::_;
using ::testing::HasSubstr;
using ::testing::Matcher;
using ::testing::Test;
using ::testing::Types;

namespace mediapipe {
namespace {

// Matcher to verify that a status is an RCheck error with the given message.
Matcher<const absl::Status&> IsRCheckError(const char* expected) {
  return StatusIs(_, HasSubstr(expected));
}

template <typename T>
class VectorToTensorCalculatorTest : public Test {
 public:
  absl::Status RunE2ETest(const std::vector<T>& input,
                          bool output_dynamic_tensor_shape);

  static std::vector<T> MakeTestVector(int size);

 private:
  static std::vector<T> ConvertTensorToVector(const Tensor& tensor);

  static absl::StatusOr<std::vector<Packet>> RunsVectorToTensorCalculator(
      const std::vector<T>& input, bool output_dynamic_tensor_shape);
};

template <typename T>
absl::Status VectorToTensorCalculatorTest<T>::RunE2ETest(
    const std::vector<T>& input, bool output_dynamic_tensor_shape) {
  MP_ASSIGN_OR_RETURN(
      const auto packet_dump,
      RunsVectorToTensorCalculator(input, output_dynamic_tensor_shape));

  auto& result_tensor = packet_dump.at(0).template Get<Tensor>();
  EXPECT_EQ(result_tensor.shape().is_dynamic, output_dynamic_tensor_shape);
  const auto result_vector = ConvertTensorToVector(result_tensor);
  EXPECT_EQ(result_vector, input);
  return absl::OkStatus();
}

template <typename T>
std::vector<T> VectorToTensorCalculatorTest<T>::MakeTestVector(int size) {
  std::vector<T> vector;
  vector.reserve(size);
  for (int i = 0; i < size; ++i) {
    vector.push_back(i);
  }
  return vector;
}

template <typename T>
std::vector<T> VectorToTensorCalculatorTest<T>::ConvertTensorToVector(
    const Tensor& tensor) {
  auto read_view = tensor.GetCpuReadView();
  auto read_view_ptr = read_view.buffer<T>();
  const int num_elements = tensor.shape().num_elements();
  std::vector<T> vector(num_elements);
  std::copy(read_view_ptr, read_view_ptr + num_elements, vector.begin());
  return vector;
}

template <typename T>
absl::StatusOr<std::vector<Packet>>
VectorToTensorCalculatorTest<T>::RunsVectorToTensorCalculator(
    const std::vector<T>& input, bool output_dynamic_tensor_shape) {
  const std::string graph_config = absl::Substitute(
      R"(
            input_stream: "input"
            output_stream: "output"
            node: { calculator: "VectorToTensorCalculator"
                    input_stream: "VECTOR:input"
                    output_stream: "TENSOR:output"
                    node_options: {
                      [type.googleapis.com/mediapipe.VectorToTensorCalculatorOptions] {
                        output_dynamic_tensor_shape: $0
                      }
                    }
            }
          )",
      output_dynamic_tensor_shape ? "true" : "false");
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(graph_config);
  std::vector<Packet> packet_dump;
  tool::AddVectorSink("output", &config, &packet_dump);
  // Create and start graph.
  CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));
  MP_RETURN_IF_ERROR(graph.StartRun({}));
  // Send input.
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "input", MakePacket<std::vector<T>>(input).At(Timestamp(0))));
  MP_RETURN_IF_ERROR(graph.WaitUntilIdle());
  // Finish processing.
  MP_RETURN_IF_ERROR(graph.CloseAllInputStreams());
  MP_RETURN_IF_ERROR(graph.WaitUntilDone());
  return packet_dump;
}

using TestTypes = Types<float, uint8_t, int8_t, int32_t, char, bool, int64_t>;
TYPED_TEST_SUITE(VectorToTensorCalculatorTest, TestTypes);

TYPED_TEST(VectorToTensorCalculatorTest, ShouldConvertVectorToTensor) {
  const auto input = this->MakeTestVector(123);
  if (std::is_same_v<TypeParam, int64_t>) {
    EXPECT_THAT(this->RunE2ETest(input, /*output_dynamic_tensor_shape=*/false),
                StatusIs(absl::StatusCode::kInvalidArgument));
  } else {
    MP_EXPECT_OK(
        this->RunE2ETest(input, /*output_dynamic_tensor_shape=*/false));
  }
}

TYPED_TEST(VectorToTensorCalculatorTest, ShouldFailOnEmptyInputVector) {
  if (!std::is_same_v<TypeParam, int64_t>) {
    EXPECT_THAT(this->RunE2ETest({}, /*output_dynamic_tensor_shape=*/false),
                IsRCheckError("Input vector is empty"));
  }
}

TYPED_TEST(VectorToTensorCalculatorTest, ShouldCreateDynamicTensor) {
  const auto input = this->MakeTestVector(123);
  if (!std::is_same_v<TypeParam, int64_t>) {
    MP_EXPECT_OK(this->RunE2ETest(input, /*output_dynamic_tensor_shape=*/true));
  }
}

}  // namespace
}  // namespace mediapipe
