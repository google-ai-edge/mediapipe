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

#include "absl/log/absl_check.h"
#include "mediapipe/calculators/tflite/tflite_tensors_to_landmarks_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "tensorflow/lite/interpreter.h"

namespace mediapipe {

namespace {

inline float Sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

float ApplyActivation(
    ::mediapipe::TfLiteTensorsToLandmarksCalculatorOptions::Activation
        activation,
    float value) {
  switch (activation) {
    case ::mediapipe::TfLiteTensorsToLandmarksCalculatorOptions::SIGMOID:
      return Sigmoid(value);
      break;
    default:
      return value;
  }
}

}  // namespace

// A calculator for converting TFLite tensors from regression models into
// landmarks. Note that if the landmarks in the tensor has more than 5
// dimensions, only the first 5 dimensions will be converted to
// [x,y,z, visibility, presence]. The latter two fields may also stay unset if
// such attributes are not supported in the model.
//
// Input:
//  TENSORS - Vector of TfLiteTensor of type kTfLiteFloat32. Only the first
//            tensor will be used. The size of the values must be
//            (num_dimension x num_landmarks).
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
//   calculator: "TfLiteTensorsToLandmarksCalculator"
//   input_stream: "TENSORS:landmark_tensors"
//   output_stream: "LANDMARKS:landmarks"
//   output_stream: "NORM_LANDMARKS:landmarks"
//   options: {
//     [mediapipe.TfLiteTensorsToLandmarksCalculatorOptions.ext] {
//       num_landmarks: 21
//
//       input_image_width: 256
//       input_image_height: 256
//     }
//   }
// }
class TfLiteTensorsToLandmarksCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  absl::Status LoadOptions(CalculatorContext* cc);
  int num_landmarks_ = 0;
  bool flip_vertically_ = false;
  bool flip_horizontally_ = false;

  ::mediapipe::TfLiteTensorsToLandmarksCalculatorOptions options_;
};
REGISTER_CALCULATOR(TfLiteTensorsToLandmarksCalculator);

absl::Status TfLiteTensorsToLandmarksCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  if (cc->Inputs().HasTag("TENSORS")) {
    cc->Inputs().Tag("TENSORS").Set<std::vector<TfLiteTensor>>();
  }

  if (cc->Inputs().HasTag("FLIP_HORIZONTALLY")) {
    cc->Inputs().Tag("FLIP_HORIZONTALLY").Set<bool>();
  }

  if (cc->Inputs().HasTag("FLIP_VERTICALLY")) {
    cc->Inputs().Tag("FLIP_VERTICALLY").Set<bool>();
  }

  if (cc->InputSidePackets().HasTag("FLIP_HORIZONTALLY")) {
    cc->InputSidePackets().Tag("FLIP_HORIZONTALLY").Set<bool>();
  }

  if (cc->InputSidePackets().HasTag("FLIP_VERTICALLY")) {
    cc->InputSidePackets().Tag("FLIP_VERTICALLY").Set<bool>();
  }

  if (cc->Outputs().HasTag("LANDMARKS")) {
    cc->Outputs().Tag("LANDMARKS").Set<LandmarkList>();
  }

  if (cc->Outputs().HasTag("NORM_LANDMARKS")) {
    cc->Outputs().Tag("NORM_LANDMARKS").Set<NormalizedLandmarkList>();
  }

  return absl::OkStatus();
}

absl::Status TfLiteTensorsToLandmarksCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  MP_RETURN_IF_ERROR(LoadOptions(cc));

  if (cc->Outputs().HasTag("NORM_LANDMARKS")) {
    RET_CHECK(options_.has_input_image_height() &&
              options_.has_input_image_width())
        << "Must provide input width/height for getting normalized landmarks.";
  }
  if (cc->Outputs().HasTag("LANDMARKS") &&
      (options_.flip_vertically() || options_.flip_horizontally() ||
       cc->InputSidePackets().HasTag("FLIP_HORIZONTALLY") ||
       cc->InputSidePackets().HasTag("FLIP_VERTICALLY"))) {
    RET_CHECK(options_.has_input_image_height() &&
              options_.has_input_image_width())
        << "Must provide input width/height for using flip_vertically option "
           "when outputting landmarks in absolute coordinates.";
  }

  flip_horizontally_ =
      cc->InputSidePackets().HasTag("FLIP_HORIZONTALLY")
          ? cc->InputSidePackets().Tag("FLIP_HORIZONTALLY").Get<bool>()
          : options_.flip_horizontally();

  flip_vertically_ =
      cc->InputSidePackets().HasTag("FLIP_VERTICALLY")
          ? cc->InputSidePackets().Tag("FLIP_VERTICALLY").Get<bool>()
          : options_.flip_vertically();

  return absl::OkStatus();
}

absl::Status TfLiteTensorsToLandmarksCalculator::Process(
    CalculatorContext* cc) {
  // Override values if specified so.
  if (cc->Inputs().HasTag("FLIP_HORIZONTALLY") &&
      !cc->Inputs().Tag("FLIP_HORIZONTALLY").IsEmpty()) {
    flip_horizontally_ = cc->Inputs().Tag("FLIP_HORIZONTALLY").Get<bool>();
  }
  if (cc->Inputs().HasTag("FLIP_VERTICALLY") &&
      !cc->Inputs().Tag("FLIP_VERTICALLY").IsEmpty()) {
    flip_vertically_ = cc->Inputs().Tag("FLIP_VERTICALLY").Get<bool>();
  }

  if (cc->Inputs().Tag("TENSORS").IsEmpty()) {
    return absl::OkStatus();
  }

  const auto& input_tensors =
      cc->Inputs().Tag("TENSORS").Get<std::vector<TfLiteTensor>>();

  const TfLiteTensor* raw_tensor = &input_tensors[0];

  int num_values = 1;
  for (int i = 0; i < raw_tensor->dims->size; ++i) {
    num_values *= raw_tensor->dims->data[i];
  }
  const int num_dimensions = num_values / num_landmarks_;
  ABSL_CHECK_GT(num_dimensions, 0);

  const float* raw_landmarks = raw_tensor->data.f;

  LandmarkList output_landmarks;

  for (int ld = 0; ld < num_landmarks_; ++ld) {
    const int offset = ld * num_dimensions;
    Landmark* landmark = output_landmarks.add_landmark();

    if (flip_horizontally_) {
      landmark->set_x(options_.input_image_width() - raw_landmarks[offset]);
    } else {
      landmark->set_x(raw_landmarks[offset]);
    }
    if (num_dimensions > 1) {
      if (flip_vertically_) {
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
  if (cc->Outputs().HasTag("NORM_LANDMARKS")) {
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
    cc->Outputs()
        .Tag("NORM_LANDMARKS")
        .AddPacket(MakePacket<NormalizedLandmarkList>(output_norm_landmarks)
                       .At(cc->InputTimestamp()));
  }

  // Output absolute landmarks.
  if (cc->Outputs().HasTag("LANDMARKS")) {
    cc->Outputs()
        .Tag("LANDMARKS")
        .AddPacket(MakePacket<LandmarkList>(output_landmarks)
                       .At(cc->InputTimestamp()));
  }

  return absl::OkStatus();
}

absl::Status TfLiteTensorsToLandmarksCalculator::LoadOptions(
    CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  options_ =
      cc->Options<::mediapipe::TfLiteTensorsToLandmarksCalculatorOptions>();
  num_landmarks_ = options_.num_landmarks();

  return absl::OkStatus();
}
}  // namespace mediapipe
