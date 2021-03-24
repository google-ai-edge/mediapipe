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

#include <cmath>
#include <vector>

#include "Eigen/Core"
#include "mediapipe/calculators/util/landmarks_to_floats_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

namespace {

constexpr char kLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kFloatsTag[] = "FLOATS";
constexpr char kMatrixTag[] = "MATRIX";

}  // namespace

// Converts a vector of landmarks to a vector of floats or a matrix.
// Input:
//   NORM_LANDMARKS: A NormalizedLandmarkList proto.
//
// Output:
//   FLOATS(optional): A vector of floats from flattened landmarks.
//   MATRIX(optional): A matrix of floats of the landmarks.
//
// Usage example:
// node {
//   calculator: "LandmarksToFloatsCalculator"
//   input_stream: "NORM_LANDMARKS:landmarks"
//   output_stream: "MATRIX:landmark_matrix"
// }
class LandmarksToFloatsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kLandmarksTag).Set<NormalizedLandmarkList>();
    RET_CHECK(cc->Outputs().HasTag(kFloatsTag) ||
              cc->Outputs().HasTag(kMatrixTag));
    if (cc->Outputs().HasTag(kFloatsTag)) {
      cc->Outputs().Tag(kFloatsTag).Set<std::vector<float>>();
    }
    if (cc->Outputs().HasTag(kMatrixTag)) {
      cc->Outputs().Tag(kMatrixTag).Set<Matrix>();
    }

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    const auto& options =
        cc->Options<::mediapipe::LandmarksToFloatsCalculatorOptions>();
    num_dimensions_ = options.num_dimensions();
    // Currently number of dimensions must be within [1, 3].
    RET_CHECK_GE(num_dimensions_, 1);
    RET_CHECK_LE(num_dimensions_, 3);
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    // Only process if there's input landmarks.
    if (cc->Inputs().Tag(kLandmarksTag).IsEmpty()) {
      return absl::OkStatus();
    }

    const auto& input_landmarks =
        cc->Inputs().Tag(kLandmarksTag).Get<NormalizedLandmarkList>();

    if (cc->Outputs().HasTag(kFloatsTag)) {
      auto output_floats = absl::make_unique<std::vector<float>>();
      for (int i = 0; i < input_landmarks.landmark_size(); ++i) {
        const NormalizedLandmark& landmark = input_landmarks.landmark(i);
        output_floats->emplace_back(landmark.x());
        if (num_dimensions_ > 1) {
          output_floats->emplace_back(landmark.y());
        }
        if (num_dimensions_ > 2) {
          output_floats->emplace_back(landmark.z());
        }
      }

      cc->Outputs()
          .Tag(kFloatsTag)
          .Add(output_floats.release(), cc->InputTimestamp());
    } else {
      auto output_matrix = absl::make_unique<Matrix>();
      output_matrix->setZero(num_dimensions_, input_landmarks.landmark_size());
      for (int i = 0; i < input_landmarks.landmark_size(); ++i) {
        (*output_matrix)(0, i) = input_landmarks.landmark(i).x();
        if (num_dimensions_ > 1) {
          (*output_matrix)(1, i) = input_landmarks.landmark(i).y();
        }
        if (num_dimensions_ > 2) {
          (*output_matrix)(2, i) = input_landmarks.landmark(i).z();
        }
      }
      cc->Outputs()
          .Tag(kMatrixTag)
          .Add(output_matrix.release(), cc->InputTimestamp());
    }
    return absl::OkStatus();
  }

 private:
  int num_dimensions_ = 0;
};
REGISTER_CALCULATOR(LandmarksToFloatsCalculator);

}  // namespace mediapipe
