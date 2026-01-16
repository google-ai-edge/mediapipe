/* Copyright 2025 The MediaPipe Authors.

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
#include "mediapipe/tasks/cc/vision/gesture_recognizer/calculators/handedness_to_matrix_calculator.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/handedness_util.h"

namespace mediapipe {
namespace tasks {

namespace {

using ::mediapipe::api3::Calculator;
using ::mediapipe::api3::CalculatorContext;
using ::mediapipe::tasks::vision::gesture_recognizer::GetRightHandScore;

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

class HandednessToMatrixNodeImpl
    : public Calculator<HandednessToMatrixNode, HandednessToMatrixNodeImpl> {
 public:
  absl::Status Process(CalculatorContext<HandednessToMatrixNode>& cc) override {
    if (!cc.in_handedness) {
      return absl::OkStatus();
    }
    const ClassificationList& handedness = cc.in_handedness.GetOrDie();

    MP_ASSIGN_OR_RETURN(auto handedness_matrix, HandednessToMatrix(handedness));
    cc.out_handedness_matrix.Send(std::move(handedness_matrix));
    return absl::OkStatus();
  }
};

}  // namespace tasks
}  // namespace mediapipe
