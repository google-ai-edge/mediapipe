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

#include "mediapipe/calculators/util/association_calculator.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/rectangle.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

// A subclass of AssociationCalculator<T> for NormalizedRect. Example use case:
// node {
//   calculator: "AssociationNormRectCalculator"
//   input_stream: "input_vec_0"
//   input_stream: "input_vec_1"
//   input_stream: "input_vec_2"
//   output_stream: "output_vec"
//   options {
//     [mediapipe.AssociationCalculatorOptions.ext] {
//       min_similarity_threshold: 0.1
//     }
// }
class AssociationNormRectCalculator
    : public AssociationCalculator<::mediapipe::NormalizedRect> {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    return AssociationCalculator<::mediapipe::NormalizedRect>::GetContract(cc);
  }

  absl::Status Open(CalculatorContext* cc) override {
    return AssociationCalculator<::mediapipe::NormalizedRect>::Open(cc);
  }

  absl::Status Process(CalculatorContext* cc) override {
    return AssociationCalculator<::mediapipe::NormalizedRect>::Process(cc);
  }

  absl::Status Close(CalculatorContext* cc) override {
    return AssociationCalculator<::mediapipe::NormalizedRect>::Close(cc);
  }

 protected:
  absl::StatusOr<Rectangle_f> GetRectangle(
      const ::mediapipe::NormalizedRect& input) override {
    if (!input.has_x_center() || !input.has_y_center() || !input.has_width() ||
        !input.has_height()) {
      return absl::InternalError("Missing dimensions in NormalizedRect.");
    }
    const float xmin = input.x_center() - input.width() / 2.0;
    const float ymin = input.y_center() - input.height() / 2.0;
    // TODO: Support rotation for rectangle.
    return Rectangle_f(xmin, ymin, input.width(), input.height());
  }
};

REGISTER_CALCULATOR(AssociationNormRectCalculator);

}  // namespace mediapipe
