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

#include <string>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/strings/substitute.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/port/parse_text_proto.h"

namespace mediapipe {
namespace tensors_to_segmentation_utils {

std::string ActivationTypeToString(
    const TensorsToSegmentationCalculatorOptions::Activation& activation) {
  switch (activation) {
    case TensorsToSegmentationCalculatorOptions::NONE:
      return "NONE";
    case TensorsToSegmentationCalculatorOptions::SIGMOID:
      return "SIGMOID";
    case TensorsToSegmentationCalculatorOptions::SOFTMAX:
      return "SOFTMAX";
  }
  ABSL_LOG(FATAL) << "Unknown activation type: " << activation;
  return "UNKNOWN";
}

std::vector<unsigned char> ArrayFloatToUnsignedChar(
    const std::vector<float>& array) {
  std::vector<unsigned char> result;
  result.reserve(array.size());
  for (int i = 0; i < array.size(); ++i) {
    result.push_back(static_cast<unsigned char>(array[i]));
  }
  return result;
}

std::vector<float> MakeRedAlphaMatrix(const std::vector<float>& values) {
  std::vector<float> result;
  result.reserve(values.size() * 4);
  for (const float& value : values) {
    result.push_back(value);
    result.push_back(0);
    result.push_back(0);
    result.push_back(value);
  }
  return result;
}

// For GPU tests, the input tensor needs to be moved to GPU, using
// TensorViewRequestor. After calculation, the output needs to be moved back
// to CPU, using ToImageCalculator. The output is an ImageFrame.
mediapipe::CalculatorGraphConfig CreateGraphConfigForTest(
    bool test_gpu,
    const TensorsToSegmentationCalculatorOptions::Activation& activation) {
  std::string pre_process = R"pb(
    node {
      calculator: "mediapipe.aimatter.TensorViewRequestor"
      input_stream: "TENSORS:tensors"
      output_stream: "TENSORS:tensors_gpu"
      options {
        [mediapipe.aimatter.TensorViewRequestorOptions.ext] { gpu {} }
      }
    }
  )pb";
  std::string post_process = R"pb(
    node {
      calculator: "FromImageCalculator"
      input_stream: "IMAGE:image_as_mask_gpu"
      output_stream: "IMAGE_CPU:image_as_mask"
    }
  )pb";
  return mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
      absl::Substitute(
          R"pb(
            input_stream: "tensors"
            input_stream: "size" $0
            node {
              calculator: "TensorsToSegmentationCalculator"
              input_stream: "TENSORS:tensors$1"
              input_stream: "OUTPUT_SIZE:size"
              output_stream: "MASK:image_as_mask$2"
              options: {
                [mediapipe.TensorsToSegmentationCalculatorOptions.ext] {
                  activation: $3
                  gpu_origin: TOP_LEFT
                }
              }
            } $4
          )pb",
          test_gpu ? pre_process : "", test_gpu ? "_gpu" : "",
          test_gpu ? "_gpu" : "", ActivationTypeToString(activation),
          test_gpu ? post_process : ""));
}
}  // namespace tensors_to_segmentation_utils
}  // namespace mediapipe
