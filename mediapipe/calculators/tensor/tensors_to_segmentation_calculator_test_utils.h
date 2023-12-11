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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_SEGMENTATION_CALCULATOR_TEST_UTILS_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_SEGMENTATION_CALCULATOR_TEST_UTILS_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mediapipe/calculators/tensor/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/framework/calculator.pb.h"

namespace mediapipe {
namespace tensors_to_segmentation_utils {
std::string ActivationTypeToString(
    const mediapipe::TensorsToSegmentationCalculatorOptions::Activation&
        activation);

std::vector<unsigned char> ArrayFloatToUnsignedChar(
    const std::vector<float>& array);

std::vector<float> MakeRedAlphaMatrix(const std::vector<float>& values);

mediapipe::CalculatorGraphConfig CreateGraphConfigForTest(
    bool test_gpu,
    const mediapipe::TensorsToSegmentationCalculatorOptions::Activation&
        activation);

struct FormattingTestCase {
  std::string test_name;
  std::vector<float> inputs;
  std::vector<float> expected_outputs;
  mediapipe::TensorsToSegmentationCalculatorOptions::Activation activation;
  int rows = 1;
  int cols = 1;
  int rows_new = 1;
  int cols_new = 1;
  int channels = 1;
  double max_abs_diff = 1e-7;
};
}  // namespace tensors_to_segmentation_utils
}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_SEGMENTATION_CALCULATOR_TEST_UTILS_H_
