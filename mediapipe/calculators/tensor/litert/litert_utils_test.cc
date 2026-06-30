// Copyright 2024 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensor/litert/litert_utils.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_model.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

struct TensorTypePair {
  std::string name;
  litert::ElementType litert_type;
  Tensor::ElementType mp_type;
};

std::vector<TensorTypePair> GetTensorTypePairs() {
  return {
      {"Float16", litert::ElementType::Float16, Tensor::ElementType::kFloat16},
      {"Float32", litert::ElementType::Float32, Tensor::ElementType::kFloat32},
      {"UInt8", litert::ElementType::UInt8, Tensor::ElementType::kUInt8},
      {"Int8", litert::ElementType::Int8, Tensor::ElementType::kInt8},
      {"Int32", litert::ElementType::Int32, Tensor::ElementType::kInt32},
      {"Int64", litert::ElementType::Int64, Tensor::ElementType::kInt64},
      {"Bool", litert::ElementType::Bool, Tensor::ElementType::kBool}};
}

TEST(LiteRtUtilsTest, ShouldSuccessfullyCompareTensorTypes) {
  const auto tensor_type_pairs = GetTensorTypePairs();
  for (int i = 0; i < tensor_type_pairs.size(); ++i) {
    for (int j = 0; j < tensor_type_pairs.size(); ++j) {
      const auto& config_i = tensor_type_pairs[i];
      const auto& config_j = tensor_type_pairs[j];

      litert::RankedTensorType litert_tensor_type(
          config_i.litert_type, litert::Layout(litert::Dimensions{1, 1}));
      Tensor mp_tensor(config_j.mp_type, Tensor::Shape({1, 1}),
                       /*memory_manager=*/nullptr,
                       /*alignment=*/0);
      const bool are_equal = i == j;
      const auto status = AreTensorSpecsEqual(litert_tensor_type, mp_tensor);
      if (are_equal) {
        MP_EXPECT_OK(status);
      } else {
        EXPECT_THAT(status, testing::status::StatusIs(::util::error::INTERNAL));
      }
    }
  }
}

TEST(LiteRtUtilsTest, ShouldDetectEqualTensorShapes) {
  constexpr int kNumElements = 1;
  litert::RankedTensorType litert_tensor_type(
      litert::ElementType::Float32,
      litert::Layout(litert::Dimensions{1, kNumElements}));
  Tensor mp_tensor(Tensor::ElementType::kFloat32,
                   Tensor::Shape({1, kNumElements}),
                   /*memory_manager=*/nullptr,
                   /*alignment=*/0);
  MP_EXPECT_OK(AreTensorSpecsEqual(litert_tensor_type, mp_tensor));
}

TEST(LiteRtUtilsTest, ShouldDetectScalarTensorShapes) {
  // Scalar input in TensorFlow is described by an empty shape.
  // In MediaPipe, we represent it as a shape with a single dimension of size 1.
  litert::Dimensions empty_dimensions;
  litert::RankedTensorType litert_tensor_type(
      litert::ElementType::Float32,
      litert::Layout(std::move(empty_dimensions)));
  Tensor mp_tensor(Tensor::ElementType::kFloat32, Tensor::Shape({1, 1}),
                   /*memory_manager=*/nullptr,
                   /*alignment=*/0);
  MP_EXPECT_OK(AreTensorSpecsEqual(litert_tensor_type, mp_tensor));
}

TEST(LiteRtUtilsTest, ShouldDetectDifferentTensorShapes) {
  litert::RankedTensorType litert_tensor_type(
      litert::ElementType::Float32, litert::Layout(litert::Dimensions{1, 2}));
  Tensor mp_tensor(Tensor::ElementType::kFloat32, Tensor::Shape({1, 1}),
                   /*memory_manager=*/nullptr,
                   /*alignment=*/0);
  EXPECT_THAT(AreTensorSpecsEqual(litert_tensor_type, mp_tensor),
              testing::status::StatusIs(
                  absl::StatusCode::kInternal,
                  testing::HasSubstr("LiteRt [2] and MediaPipe [1]")));
}

class CreateTensorFromLiteRtRankedTensorTypeTest
    : public ::testing::TestWithParam<TensorTypePair> {};

TEST_P(CreateTensorFromLiteRtRankedTensorTypeTest,
       ShouldAllocateTensorWithLiteRtTensorSpecs) {
  const int kNumElements = 4;
  const auto& config = GetParam();

  litert::RankedTensorType litert_tensor_type(
      config.litert_type, litert::Layout(litert::Dimensions{1, kNumElements}));

  MP_ASSERT_OK_AND_ASSIGN(
      Tensor mp_tensor,
      CreateTensorFromLiteRtRankedTensorType(
          litert_tensor_type,
          /*memory_manager=*/nullptr, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT));
  EXPECT_EQ(mp_tensor.element_type(), config.mp_type);
  EXPECT_EQ(mp_tensor.shape().num_elements(), kNumElements);
}

INSTANTIATE_TEST_SUITE_P(
    AllocateTensorWithLiteRtTensorSpecsParamTest,
    CreateTensorFromLiteRtRankedTensorTypeTest,
    ::testing::ValuesIn(GetTensorTypePairs()),
    [](const testing::TestParamInfo<
        CreateTensorFromLiteRtRankedTensorTypeTest::ParamType>& info) {
      return info.param.name;
    });

}  // namespace
}  // namespace mediapipe
