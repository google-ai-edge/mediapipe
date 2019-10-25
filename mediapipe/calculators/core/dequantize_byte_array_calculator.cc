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

#include <cfloat>

#include "mediapipe/calculators/core/dequantize_byte_array_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/status.h"

// Dequantizes a byte array to a vector of floats.
//
// Example config:
//   node {
//     calculator: "DequantizeByteArrayCalculator"
//     input_stream: "ENCODED:encoded"
//     output_stream: "FLOAT_VECTOR:float_vector"
//     options {
//       [mediapipe.DequantizeByteArrayCalculatorOptions.ext]: {
//         max_quantized_value: 2
//         min_quantized_value: -2
//       }
//     }
//   }
namespace mediapipe {

class DequantizeByteArrayCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag("ENCODED").Set<std::string>();
    cc->Outputs().Tag("FLOAT_VECTOR").Set<std::vector<float>>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    const auto options =
        cc->Options<::mediapipe::DequantizeByteArrayCalculatorOptions>();
    if (!options.has_max_quantized_value() ||
        !options.has_min_quantized_value()) {
      return ::mediapipe::InvalidArgumentError(
          "Both max_quantized_value and min_quantized_value must be provided "
          "in DequantizeByteArrayCalculatorOptions.");
    }
    float max_quantized_value = options.max_quantized_value();
    float min_quantized_value = options.min_quantized_value();
    if (max_quantized_value < min_quantized_value + FLT_EPSILON) {
      return ::mediapipe::InvalidArgumentError(
          "max_quantized_value must be greater than min_quantized_value.");
    }
    float range = max_quantized_value - min_quantized_value;
    scalar_ = range / 255.0;
    bias_ = (range / 512.0) + min_quantized_value;
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    const std::string& encoded =
        cc->Inputs().Tag("ENCODED").Value().Get<std::string>();
    std::vector<float> float_vector;
    float_vector.reserve(encoded.length());
    for (int i = 0; i < encoded.length(); ++i) {
      float_vector.push_back(
          static_cast<unsigned char>(encoded.at(i)) * scalar_ + bias_);
    }
    cc->Outputs()
        .Tag("FLOAT_VECTOR")
        .AddPacket(MakePacket<std::vector<float>>(float_vector)
                       .At(cc->InputTimestamp()));
    return ::mediapipe::OkStatus();
  }

 private:
  float scalar_;
  float bias_;
};

REGISTER_CALCULATOR(DequantizeByteArrayCalculator);

}  // namespace mediapipe
