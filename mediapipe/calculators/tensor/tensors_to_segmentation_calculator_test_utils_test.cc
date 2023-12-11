// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensor/tensors_to_segmentation_calculator_test_utils.h"

#include <vector>

#include "mediapipe/calculators/tensor/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe::tensors_to_segmentation_utils {
namespace {

using Options = ::mediapipe::TensorsToSegmentationCalculatorOptions;

TEST(TensorsToSegmentationCalculatorTestUtilsTest,
     ActivationTypeToStringWorksCorrectly) {
  EXPECT_EQ(ActivationTypeToString(Options::NONE), "NONE");
  EXPECT_EQ(ActivationTypeToString(Options::SIGMOID), "SIGMOID");
  EXPECT_EQ(ActivationTypeToString(Options::SOFTMAX), "SOFTMAX");
}

TEST(TensorsToSegmentationCalculatorTestUtilsTest,
     ArrayFloatToUnsignedCharWorksCorrectly) {
  std::vector<float> input = {1.0, 2.0, 3.0};
  std::vector<unsigned char> expected = {1, 2, 3};
  EXPECT_EQ(ArrayFloatToUnsignedChar(input), expected);
}

TEST(TensorsToSegmentationCalculatorTestUtilsTest,
     MakeRedAlphaMatrixWorksCorrectly) {
  std::vector<float> input = {1.0, 2.0, 3.0};
  std::vector<float> expected = {1.0, 0.0, 0.0, 1.0, 2.0, 0.0,
                                 0.0, 2.0, 3.0, 0.0, 0.0, 3.0};
  EXPECT_EQ(MakeRedAlphaMatrix(input), expected);
}

}  // namespace
}  // namespace mediapipe::tensors_to_segmentation_utils
