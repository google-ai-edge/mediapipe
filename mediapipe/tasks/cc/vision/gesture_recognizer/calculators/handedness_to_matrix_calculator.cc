/* Copyright 2022 The MediaPipe Authors.

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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/handedness_util.h"

// TODO Update to use API2
namespace mediapipe {
namespace api2 {

namespace {

using ::mediapipe::tasks::vision::gesture_recognizer::GetRightHandScore;

constexpr char kHandednessTag[] = "HANDEDNESS";
constexpr char kHandednessMatrixTag[] = "HANDEDNESS_MATRIX";

absl::StatusOr<std::unique_ptr<Matrix>> HandednessToMatrix(
    const mediapipe::ClassificationList& classification_list) {
  // Feature value is the probability that the hand is a right hand.
  MP_ASSIGN_OR_RETURN(float score, GetRightHandScore(classification_list));
  auto matrix = Matrix(1, 1);
  matrix(0, 0) = score;
  auto result = std::make_unique<Matrix>();
  *result = matrix;
  return result;
}

}  // namespace

// Convert single hand handedness into a matrix.
//
// Input:
//   HANDEDNESS - Single hand handedness.
// Output:
//   HANDEDNESS_MATRIX - Matrix for handedness.
//
// Usage example:
// node {
//   calculator: "HandednessToMatrixCalculator"
//   input_stream: "HANDEDNESS:handedness"
//   output_stream: "HANDEDNESS_MATRIX:handedness_matrix"
// }
class HandednessToMatrixCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kHandednessTag).Set<mediapipe::ClassificationList>();
    cc->Outputs().Tag(kHandednessMatrixTag).Set<Matrix>();
    return absl::OkStatus();
  }

  // TODO remove this after change to API2, because Setting offset
  // to 0 is the default in API2
  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override;
};

REGISTER_CALCULATOR(HandednessToMatrixCalculator);

absl::Status HandednessToMatrixCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().Tag(kHandednessTag).IsEmpty()) {
    return absl::OkStatus();
  }
  auto handedness =
      cc->Inputs().Tag(kHandednessTag).Get<mediapipe::ClassificationList>();

  MP_ASSIGN_OR_RETURN(auto handedness_matrix, HandednessToMatrix(handedness));
  cc->Outputs()
      .Tag(kHandednessMatrixTag)
      .Add(handedness_matrix.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

}  // namespace api2
}  // namespace mediapipe
