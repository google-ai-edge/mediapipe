// Copyright 2020 The MediaPipe Authors.
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

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.h"
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"
namespace mediapipe {

namespace {

constexpr int kNumFaceLandmarkConnections = 124;
// Pairs of landmark indices to be rendered with connections.
constexpr int kFaceLandmarkConnections[] = {
    // Lips.
    61, 146, 146, 91, 91, 181, 181, 84, 84, 17, 17, 314, 314, 405, 405, 321,
    321, 375, 375, 291, 61, 185, 185, 40, 40, 39, 39, 37, 37, 0, 0, 267, 267,
    269, 269, 270, 270, 409, 409, 291, 78, 95, 95, 88, 88, 178, 178, 87, 87, 14,
    14, 317, 317, 402, 402, 318, 318, 324, 324, 308, 78, 191, 191, 80, 80, 81,
    81, 82, 82, 13, 13, 312, 312, 311, 311, 310, 310, 415, 415, 308,
    // Left eye.
    33, 7, 7, 163, 163, 144, 144, 145, 145, 153, 153, 154, 154, 155, 155, 133,
    33, 246, 246, 161, 161, 160, 160, 159, 159, 158, 158, 157, 157, 173, 173,
    133,
    // Left eyebrow.
    46, 53, 53, 52, 52, 65, 65, 55, 70, 63, 63, 105, 105, 66, 66, 107,
    // Right eye.
    263, 249, 249, 390, 390, 373, 373, 374, 374, 380, 380, 381, 381, 382, 382,
    362, 263, 466, 466, 388, 388, 387, 387, 386, 386, 385, 385, 384, 384, 398,
    398, 362,
    // Right eyebrow.
    276, 283, 283, 282, 282, 295, 295, 285, 300, 293, 293, 334, 334, 296, 296,
    336,
    // Face oval.
    10, 338, 338, 297, 297, 332, 332, 284, 284, 251, 251, 389, 389, 356, 356,
    454, 454, 323, 323, 361, 361, 288, 288, 397, 397, 365, 365, 379, 379, 378,
    378, 400, 400, 377, 377, 152, 152, 148, 148, 176, 176, 149, 149, 150, 150,
    136, 136, 172, 172, 58, 58, 132, 132, 93, 93, 234, 234, 127, 127, 162, 162,
    21, 21, 54, 54, 103, 103, 67, 67, 109, 109, 10};

}  // namespace

// A calculator that converts face landmarks to RenderData proto for
// visualization. Ignores landmark_connections specified in
// LandmarksToRenderDataCalculatorOptions, if any, and always uses a fixed set
// of landmark connections specific to face landmark (defined in
// kFaceLandmarkConnections[] above).
//
// Example config:
// node {
//   calculator: "FaceLandmarksToRenderDataCalculator"
//   input_stream: "NORM_LANDMARKS:landmarks"
//   output_stream: "RENDER_DATA:render_data"
//   options {
//     [LandmarksToRenderDataCalculatorOptions.ext] {
//       landmark_color { r: 0 g: 255 b: 0 }
//       connection_color { r: 0 g: 255 b: 0 }
//       thickness: 4.0
//     }
//   }
// }
class FaceLandmarksToRenderDataCalculator
    : public LandmarksToRenderDataCalculator {
 public:
  ::mediapipe::Status Open(CalculatorContext* cc) override;
};
REGISTER_CALCULATOR(FaceLandmarksToRenderDataCalculator);

::mediapipe::Status FaceLandmarksToRenderDataCalculator::Open(
    CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  options_ = cc->Options<mediapipe::LandmarksToRenderDataCalculatorOptions>();

  for (int i = 0; i < kNumFaceLandmarkConnections; ++i) {
    landmark_connections_.push_back(kFaceLandmarkConnections[i * 2]);
    landmark_connections_.push_back(kFaceLandmarkConnections[i * 2 + 1]);
  }

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
