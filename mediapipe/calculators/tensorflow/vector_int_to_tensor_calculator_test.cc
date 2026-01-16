// Copyright 2018 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensorflow/vector_int_to_tensor_calculator_options.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gtest.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mediapipe {

namespace {

namespace tf = ::tensorflow;

constexpr char kSingleIntTag[] = "SINGLE_INT";
constexpr char kTensorOutTag[] = "TENSOR_OUT";
constexpr char kVectorIntTag[] = "VECTOR_INT";

class VectorIntToTensorCalculatorTest : public ::testing::Test {
 protected:
  void SetUpRunner(
      const VectorIntToTensorCalculatorOptions::InputSize input_size,
      const tensorflow::DataType tensor_data_type, const bool transpose,
      const bool single_value) {
    SetUpRunner(input_size, tensor_data_type, transpose, single_value,
                /*scalar_output=*/false);
  }

  void SetUpRunner(
      const VectorIntToTensorCalculatorOptions::InputSize input_size,
      const tensorflow::DataType tensor_data_type, const bool transpose,
      const bool single_value, const bool scalar_output) {
    CalculatorGraphConfig::Node config;
    config.set_calculator("VectorIntToTensorCalculator");
    if (single_value) {
      config.add_input_stream("SINGLE_INT:input_int");
    } else {
      config.add_input_stream("VECTOR_INT:input_int");
    }
    config.add_output_stream("TENSOR_OUT:output_tensor");
    auto options = config.mutable_options()->MutableExtension(
        VectorIntToTensorCalculatorOptions::ext);
    options->set_input_size(input_size);
    options->set_transpose(transpose);
    options->set_tensor_data_type(tensor_data_type);
    options->set_scalar_output(scalar_output);
    runner_ = std::make_unique<CalculatorRunner>(config);
  }

  void TestConvertFromVectoVectorInt(const bool transpose) {
    SetUpRunner(VectorIntToTensorCalculatorOptions::INPUT_2D,
                tensorflow::DT_INT32, transpose, false);
    auto input =
        std::make_unique<std::vector<std::vector<int>>>(2, std::vector<int>(2));
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        input->at(i).at(j) = i * 2 + j;
      }
    }

    const int64_t time = 1234;
    runner_->MutableInputs()
        ->Tag(kVectorIntTag)
        .packets.push_back(Adopt(input.release()).At(Timestamp(time)));

    EXPECT_TRUE(runner_->Run().ok());

    const std::vector<Packet>& output_packets =
        runner_->Outputs().Tag(kTensorOutTag).packets;
    EXPECT_EQ(output_packets.size(), 1);
    EXPECT_EQ(output_packets[0].Timestamp().Value(), time);
    const tf::Tensor& output_tensor = output_packets[0].Get<tf::Tensor>();

    EXPECT_EQ(output_tensor.dims(), 2);
    EXPECT_EQ(output_tensor.dtype(), tf::DT_INT32);
    const auto matrix = output_tensor.matrix<int>();

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        if (!transpose) {
          EXPECT_EQ(i * 2 + j, matrix(i, j));
        } else {
          EXPECT_EQ(j * 2 + i, matrix(i, j));
        }
      }
    }
  }

  std::unique_ptr<CalculatorRunner> runner_;
};

TEST_F(VectorIntToTensorCalculatorTest, TestSingleValue) {
  SetUpRunner(VectorIntToTensorCalculatorOptions::INPUT_1D,
              tensorflow::DT_INT32, false, true);
  const int64_t time = 1234;
  runner_->MutableInputs()
      ->Tag(kSingleIntTag)
      .packets.push_back(MakePacket<int>(1).At(Timestamp(time)));

  ASSERT_TRUE(runner_->Run().ok());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kTensorOutTag).packets;
  ASSERT_EQ(1, output_packets.size());
  ASSERT_EQ(time, output_packets[0].Timestamp().Value());
  const tf::Tensor& output_tensor = output_packets[0].Get<tf::Tensor>();

  EXPECT_EQ(output_tensor.dims(), 1);
  EXPECT_EQ(output_tensor.dtype(), tf::DT_INT32);
  const auto vec = output_tensor.vec<int32_t>();
  EXPECT_EQ(vec(0), 1);
}

TEST_F(VectorIntToTensorCalculatorTest, TestOneDim) {
  SetUpRunner(VectorIntToTensorCalculatorOptions::INPUT_1D,
              tensorflow::DT_INT32, false, false);
  auto input = std::make_unique<std::vector<int>>(5);
  for (int i = 0; i < 5; ++i) {
    input->at(i) = i;
  }
  const int64_t time = 1234;
  runner_->MutableInputs()
      ->Tag(kVectorIntTag)
      .packets.push_back(Adopt(input.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kTensorOutTag).packets;
  EXPECT_EQ(output_packets.size(), 1);
  EXPECT_EQ(output_packets[0].Timestamp().Value(), time);
  const tf::Tensor& output_tensor = output_packets[0].Get<tf::Tensor>();

  EXPECT_EQ(output_tensor.dims(), 1);
  EXPECT_EQ(output_tensor.dtype(), tf::DT_INT32);
  const auto vec = output_tensor.vec<int32_t>();

  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(i, vec(i));
  }
}

TEST_F(VectorIntToTensorCalculatorTest, TestTwoDims) {
  for (bool transpose : {false, true}) {
    TestConvertFromVectoVectorInt(transpose);
  }
}

TEST_F(VectorIntToTensorCalculatorTest, TestInt64) {
  SetUpRunner(VectorIntToTensorCalculatorOptions::INPUT_1D,
              tensorflow::DT_INT64, false, true);
  const int64_t time = 1234;
  runner_->MutableInputs()
      ->Tag(kSingleIntTag)
      .packets.push_back(MakePacket<int64_t>(1LL << 31).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kTensorOutTag).packets;
  EXPECT_EQ(output_packets.size(), 1);
  EXPECT_EQ(output_packets[0].Timestamp().Value(), time);
  const tf::Tensor& output_tensor = output_packets[0].Get<tf::Tensor>();

  EXPECT_EQ(output_tensor.dims(), 1);
  EXPECT_EQ(output_tensor.dtype(), tf::DT_INT64);
  const auto vec = output_tensor.vec<int64_t>();
  // 1LL << 31 overflows the positive int and becomes negative.
  EXPECT_EQ(static_cast<int64_t>(1LL << 31), vec(0));
}

TEST_F(VectorIntToTensorCalculatorTest, TestUint8) {
  SetUpRunner(VectorIntToTensorCalculatorOptions::INPUT_1D,
              tensorflow::DT_UINT8, false, false);
  auto input = std::make_unique<std::vector<int>>(5);
  for (int i = 0; i < 5; ++i) {
    input->at(i) = i;
  }
  const int64_t time = 1234;
  runner_->MutableInputs()
      ->Tag(kVectorIntTag)
      .packets.push_back(Adopt(input.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kTensorOutTag).packets;
  EXPECT_EQ(output_packets.size(), 1);
  EXPECT_EQ(output_packets[0].Timestamp().Value(), time);
  const tf::Tensor& output_tensor = output_packets[0].Get<tf::Tensor>();

  EXPECT_EQ(output_tensor.dims(), 1);
  EXPECT_EQ(output_tensor.dtype(), tf::DT_UINT8);
  const auto vec = output_tensor.vec<uint8_t>();

  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(i, vec(i));
  }
}

TEST_F(VectorIntToTensorCalculatorTest, TestSingleUnsignedValue) {
  SetUpRunner(VectorIntToTensorCalculatorOptions::INPUT_1D,
              tensorflow::DT_UINT32, false, true);
  const int64_t time = 1234;
  runner_->MutableInputs()
      ->Tag(kSingleIntTag)
      .packets.push_back(MakePacket<uint32_t>(1).At(Timestamp(time)));

  ASSERT_TRUE(runner_->Run().ok());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kTensorOutTag).packets;
  EXPECT_EQ(output_packets.size(), 1);
  EXPECT_EQ(output_packets[0].Timestamp().Value(), time);
  const tf::Tensor& output_tensor = output_packets[0].Get<tf::Tensor>();

  EXPECT_EQ(output_tensor.dims(), 1);
  EXPECT_EQ(output_tensor.dtype(), tf::DT_UINT32);
  const auto vec = output_tensor.vec<uint32_t>();
  EXPECT_EQ(vec(0), 1);
}

TEST_F(VectorIntToTensorCalculatorTest, TestSingleValueScalar) {
  SetUpRunner(VectorIntToTensorCalculatorOptions::INPUT_1D,
              tensorflow::DT_INT32, false, true, true);
  const int64_t time = 1234;
  runner_->MutableInputs()
      ->Tag(kSingleIntTag)
      .packets.push_back(MakePacket<int>(1).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kTensorOutTag).packets;
  EXPECT_EQ(output_packets.size(), 1);
  EXPECT_EQ(output_packets[0].Timestamp().Value(), time);
  const tf::Tensor& output_tensor = output_packets[0].Get<tf::Tensor>();

  EXPECT_EQ(output_tensor.dims(), 0);
  EXPECT_EQ(output_tensor.dtype(), tf::DT_INT32);
  EXPECT_EQ(output_tensor.scalar<int32_t>()(), 1);
}

TEST_F(VectorIntToTensorCalculatorTest, TestOneDimToScalar) {
  SetUpRunner(VectorIntToTensorCalculatorOptions::INPUT_1D,
              tensorflow::DT_INT32, false, false, true);
  auto input = std::make_unique<std::vector<int>>(1);
  input->at(0) = 7;
  const int64_t time = 1234;
  runner_->MutableInputs()
      ->Tag(kVectorIntTag)
      .packets.push_back(Adopt(input.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kTensorOutTag).packets;
  EXPECT_EQ(output_packets.size(), 1);
  EXPECT_EQ(output_packets[0].Timestamp().Value(), time);
  const tf::Tensor& output_tensor = output_packets[0].Get<tf::Tensor>();

  EXPECT_EQ(output_tensor.dims(), 0);
  EXPECT_EQ(output_tensor.dtype(), tf::DT_INT32);
  EXPECT_EQ(output_tensor.scalar<int32_t>()(), 7);
}

}  // namespace
}  // namespace mediapipe
