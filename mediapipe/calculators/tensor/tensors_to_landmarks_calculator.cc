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

#include "mediapipe/calculators/tensor/tensors_to_landmarks_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

namespace {

inline float Sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

float ApplyActivation(
    ::mediapipe::TensorsToLandmarksCalculatorOptions::Activation activation,
    float value) {
  switch (activation) {
    case ::mediapipe::TensorsToLandmarksCalculatorOptions::SIGMOID:
      return Sigmoid(value);
      break;
    default:
      return value;
  }
}

}  // namespace

// A calculator for converting Tensors from regression models into landmarks.
// Note that if the landmarks in the tensor has more than 5 dimensions, only the
// first 5 dimensions will be converted to [x,y,z, visibility, presence]. The
// latter two fields may also stay unset if such attributes are not supported in
// the model.
//
// Input:
//  TENSORS - Vector of Tensors of type kFloat32. Only the first tensor will be
//  used. The size of the values must be (num_dimension x num_landmarks).
//
//  FLIP_HORIZONTALLY (optional): Whether to flip landmarks horizontally or
//  not. Overrides corresponding side packet and/or field in the calculator
//  options.
//
//  FLIP_VERTICALLY (optional): Whether to flip landmarks vertically or not.
//  Overrides corresponding side packet and/or field in the calculator options.
//
// Input side packet:
//   FLIP_HORIZONTALLY (optional): Whether to flip landmarks horizontally or
//   not. Overrides the corresponding field in the calculator options.
//
//   FLIP_VERTICALLY (optional): Whether to flip landmarks vertically or not.
//   Overrides the corresponding field in the calculator options.
//
// Output:
//  LANDMARKS(optional) - Result MediaPipe landmarks.
//  NORM_LANDMARKS(optional) - Result MediaPipe normalized landmarks.
//
// Notes:
//   To output normalized landmarks, user must provide the original input image
//   size to the model using calculator option input_image_width and
//   input_image_height.
// Usage example:
// node {
//   calculator: "TensorsToLandmarksCalculator"
//   input_stream: "TENSORS:landmark_tensors"
//   output_stream: "LANDMARKS:landmarks"
//   output_stream: "NORM_LANDMARKS:landmarks"
//   options: {
//     [mediapipe.TensorsToLandmarksCalculatorOptions.ext] {
//       num_landmarks: 21
//
//       input_image_width: 256
//       input_image_height: 256
//     }
//   }
// }
class TensorsToLandmarksCalculator : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kInTensors{"TENSORS"};
  static constexpr Input<bool>::SideFallback::Optional kFlipHorizontally{
      "FLIP_HORIZONTALLY"};
  static constexpr Input<bool>::SideFallback::Optional kFlipVertically{
      "FLIP_VERTICALLY"};
  static constexpr Output<LandmarkList>::Optional kOutLandmarkList{"LANDMARKS"};
  static constexpr Output<NormalizedLandmarkList>::Optional
      kOutNormalizedLandmarkList{"NORM_LANDMARKS"};
  MEDIAPIPE_NODE_CONTRACT(kInTensors, kFlipHorizontally, kFlipVertically,
                          kOutLandmarkList, kOutNormalizedLandmarkList);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  absl::Status LoadOptions(CalculatorContext* cc);
  int num_landmarks_ = 0;
  ::mediapipe::TensorsToLandmarksCalculatorOptions options_;
};
MEDIAPIPE_REGISTER_NODE(TensorsToLandmarksCalculator);

absl::Status TensorsToLandmarksCalculator::Open(CalculatorContext* cc) {
  MP_RETURN_IF_ERROR(LoadOptions(cc));

  if (kOutNormalizedLandmarkList(cc).IsConnected()) {
    RET_CHECK(options_.has_input_image_height() &&
              options_.has_input_image_width())
        << "Must provide input width/height for getting normalized landmarks.";
  }
  if (kOutLandmarkList(cc).IsConnected() &&
      (options_.flip_horizontally() || options_.flip_vertically() ||
       kFlipHorizontally(cc).IsConnected() ||
       kFlipVertically(cc).IsConnected())) {
    RET_CHECK(options_.has_input_image_height() &&
              options_.has_input_image_width())
        << "Must provide input width/height for using flipping when outputting "
           "landmarks in absolute coordinates.";
  }
  return absl::OkStatus();
}

absl::Status TensorsToLandmarksCalculator::Process(CalculatorContext* cc) {
  if (kInTensors(cc).IsEmpty()) {
    return absl::OkStatus();
  }
  bool flip_horizontally =
      kFlipHorizontally(cc).GetOr(options_.flip_horizontally());
  bool flip_vertically = kFlipVertically(cc).GetOr(options_.flip_vertically());

  const auto& input_tensors = *kInTensors(cc);
  RET_CHECK(input_tensors[0].element_type() == Tensor::ElementType::kFloat32);
  int num_values = input_tensors[0].shape().num_elements();
  const int num_dimensions = num_values / num_landmarks_;
  ABSL_CHECK_GT(num_dimensions, 0);

  auto view = input_tensors[0].GetCpuReadView();
  auto raw_landmarks = view.buffer<float>();

  LandmarkList output_landmarks;

  for (int ld = 0; ld < num_landmarks_; ++ld) {
    const int offset = ld * num_dimensions;
    Landmark* landmark = output_landmarks.add_landmark();

    if (flip_horizontally) {
      landmark->set_x(options_.input_image_width() - raw_landmarks[offset]);
    } else {
      landmark->set_x(raw_landmarks[offset]);
    }
    if (num_dimensions > 1) {
      if (flip_vertically) {
        landmark->set_y(options_.input_image_height() -
                        raw_landmarks[offset + 1]);
      } else {
        landmark->set_y(raw_landmarks[offset + 1]);
      }
    }
    if (num_dimensions > 2) {
      landmark->set_z(raw_landmarks[offset + 2]);
    }
    if (num_dimensions > 3) {
      landmark->set_visibility(ApplyActivation(options_.visibility_activation(),
                                               raw_landmarks[offset + 3]));
    }
    if (num_dimensions > 4) {
      landmark->set_presence(ApplyActivation(options_.presence_activation(),
                                             raw_landmarks[offset + 4]));
    }
  }

  // Output normalized landmarks if required.
  if (kOutNormalizedLandmarkList(cc).IsConnected()) {
    NormalizedLandmarkList output_norm_landmarks;
    for (int i = 0; i < output_landmarks.landmark_size(); ++i) {
      const Landmark& landmark = output_landmarks.landmark(i);
      NormalizedLandmark* norm_landmark = output_norm_landmarks.add_landmark();
      norm_landmark->set_x(landmark.x() / options_.input_image_width());
      norm_landmark->set_y(landmark.y() / options_.input_image_height());
      // Scale Z coordinate as X + allow additional uniform normalization.
      norm_landmark->set_z(landmark.z() / options_.input_image_width() /
                           options_.normalize_z());
      if (landmark.has_visibility()) {  // Set only if supported in the model.
        norm_landmark->set_visibility(landmark.visibility());
      }
      if (landmark.has_presence()) {  // Set only if supported in the model.
        norm_landmark->set_presence(landmark.presence());
      }
    }
    kOutNormalizedLandmarkList(cc).Send(std::move(output_norm_landmarks));
  }

  // Output absolute landmarks.
  if (kOutLandmarkList(cc).IsConnected()) {
    kOutLandmarkList(cc).Send(std::move(output_landmarks));
  }

  return absl::OkStatus();
}

absl::Status TensorsToLandmarksCalculator::LoadOptions(CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  options_ = cc->Options<::mediapipe::TensorsToLandmarksCalculatorOptions>();
  RET_CHECK(options_.has_num_landmarks());
  num_landmarks_ = options_.num_landmarks();

  return absl::OkStatus();
}
}  // namespace api2
}  // namespace mediapipe
