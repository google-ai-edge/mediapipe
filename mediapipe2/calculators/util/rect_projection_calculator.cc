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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe {

namespace {

constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kNormReferenceRectTag[] = "NORM_REFERENCE_RECT";

}  // namespace

// Projects rectangle from reference coordinate system (defined by reference
// rectangle) to original coordinate system (in which this reference rectangle
// is defined).
//
// Inputs:
//   NORM_RECT - A NormalizedRect to be projected.
//   NORM_REFERENCE_RECT - A NormalizedRect that represents reference coordinate
//     system for NORM_RECT and is defined in original coordinates.
//
// Outputs:
//   NORM_RECT: A NormalizedRect projected to the original coordinates.
//
// Example config:
//   node {
//     calculator: "RectProjectionCalculator"
//     input_stream: "NORM_RECT:face_rect"
//     input_stream: "NORM_REFERENCE_RECT:face_reference_rect"
//     output_stream: "NORM_RECT:projected_face_rect"
//   }
//
class RectProjectionCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
};
REGISTER_CALCULATOR(RectProjectionCalculator);

absl::Status RectProjectionCalculator::GetContract(CalculatorContract* cc) {
  cc->Inputs().Tag(kNormRectTag).Set<NormalizedRect>();
  cc->Inputs().Tag(kNormReferenceRectTag).Set<NormalizedRect>();
  cc->Outputs().Tag(kNormRectTag).Set<NormalizedRect>();
  return absl::OkStatus();
}

absl::Status RectProjectionCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  return absl::OkStatus();
}

absl::Status RectProjectionCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().Tag(kNormRectTag).IsEmpty()) {
    return absl::OkStatus();
  }

  const auto& rect = cc->Inputs().Tag(kNormRectTag).Get<NormalizedRect>();
  const auto& reference_rect =
      cc->Inputs().Tag(kNormReferenceRectTag).Get<NormalizedRect>();

  // Project center.
  const float x = rect.x_center() - 0.5f;
  const float y = rect.y_center() - 0.5f;
  const float angle = reference_rect.rotation();
  float new_x = std::cos(angle) * x - std::sin(angle) * y;
  float new_y = std::sin(angle) * x + std::cos(angle) * y;
  new_x = new_x * reference_rect.width() + reference_rect.x_center();
  new_y = new_y * reference_rect.height() + reference_rect.y_center();

  // Project size.
  const float new_width = rect.width() * reference_rect.width();
  const float new_height = rect.height() * reference_rect.height();

  // Project rotation.
  const float new_rotation = rect.rotation() + reference_rect.rotation();

  auto new_rect = absl::make_unique<NormalizedRect>();
  new_rect->set_x_center(new_x);
  new_rect->set_y_center(new_y);
  new_rect->set_width(new_width);
  new_rect->set_height(new_height);
  new_rect->set_rotation(new_rotation);

  cc->Outputs().Tag(kNormRectTag).Add(new_rect.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

}  // namespace mediapipe
