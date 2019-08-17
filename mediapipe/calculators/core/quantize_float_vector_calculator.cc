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
#include <memory>
#include <string>
#include <vector>

#include "mediapipe/calculators/core/quantize_float_vector_calculator.pb.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/status.h"

// Quantizes a vector of floats to a std::string so that each float becomes a
// byte in the [0, 255] range. Any value above max_quantized_value or below
// min_quantized_value will be saturated to '/xFF' or '/0'.
//
// Example config:
//   node {
//     calculator: "QuantizeFloatVectorCalculator"
//     input_stream: "FLOAT_VECTOR:float_vector"
//     output_stream: "ENCODED:encoded"
//     options {
//       [mediapipe.QuantizeFloatVectorCalculatorOptions.ext]: {
//         max_quantized_value: 64
//         min_quantized_value: -64
//       }
//     }
//   }
namespace mediapipe {

class QuantizeFloatVectorCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag("FLOAT_VECTOR").Set<std::vector<float>>();
    cc->Outputs().Tag("ENCODED").Set<std::string>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    const auto options =
        cc->Options<::mediapipe::QuantizeFloatVectorCalculatorOptions>();
    if (!options.has_max_quantized_value() ||
        !options.has_min_quantized_value()) {
      return ::mediapipe::InvalidArgumentError(
          "Both max_quantized_value and min_quantized_value must be provided "
          "in QuantizeFloatVectorCalculatorOptions.");
    }
    max_quantized_value_ = options.max_quantized_value();
    min_quantized_value_ = options.min_quantized_value();
    if (max_quantized_value_ < min_quantized_value_ + FLT_EPSILON) {
      return ::mediapipe::InvalidArgumentError(
          "max_quantized_value must be greater than min_quantized_value.");
    }
    range_ = max_quantized_value_ - min_quantized_value_;
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    const std::vector<float>& float_vector =
        cc->Inputs().Tag("FLOAT_VECTOR").Value().Get<std::vector<float>>();
    int feature_size = float_vector.size();
    std::string encoded_features;
    encoded_features.reserve(feature_size);
    for (int i = 0; i < feature_size; i++) {
      float old_value = float_vector[i];
      if (old_value < min_quantized_value_) {
        old_value = min_quantized_value_;
      }
      if (old_value > max_quantized_value_) {
        old_value = max_quantized_value_;
      }
      unsigned char encoded = static_cast<unsigned char>(
          (old_value - min_quantized_value_) * (255.0 / range_));
      encoded_features += encoded;
    }
    cc->Outputs().Tag("ENCODED").AddPacket(
        MakePacket<std::string>(encoded_features).At(cc->InputTimestamp()));
    return ::mediapipe::OkStatus();
  }

 private:
  float max_quantized_value_;
  float min_quantized_value_;
  float range_;
};

REGISTER_CALCULATOR(QuantizeFloatVectorCalculator);

}  // namespace mediapipe
