#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "absl/status/status.h"

namespace mediapipe {
namespace api2 {

class ConfidenceNormalizedLandmarkMergerCalculator : public CalculatorBase {
 public:
  static constexpr Input<NormalizedLandmarkList> kInLandmarks{"LANDMARKS"};
  static constexpr Input<float> kInConfidence{"CONFIDENCE"};
  static constexpr Output<NormalizedLandmarkList> kOutLandmarks{"LANDMARKS"};

  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Process(CalculatorContext* cc) override;
};

absl::Status ConfidenceNormalizedLandmarkMergerCalculator::GetContract(
    CalculatorContract* cc) {
  cc->Inputs().Tag(kInLandmarks.Tag()).Set<NormalizedLandmarkList>();
  cc->Inputs().Tag(kInConfidence.Tag()).Set<float>();
  cc->Outputs().Tag(kOutLandmarks.Tag()).Set<NormalizedLandmarkList>();
  return absl::OkStatus();
}

absl::Status ConfidenceNormalizedLandmarkMergerCalculator::Process(
    CalculatorContext* cc) {
  if (cc->Inputs().Tag(kInLandmarks.Tag()).IsEmpty() ||
      cc->Inputs().Tag(kInConfidence.Tag()).IsEmpty()) {
    return absl::OkStatus();
  }

  const auto& input_landmarks = 
      cc->Inputs().Tag(kInLandmarks.Tag()).Get<NormalizedLandmarkList>();
  const float confidence = 
      cc->Inputs().Tag(kInConfidence.Tag()).Get<float>();

  auto output_landmarks = 
      absl::make_unique<NormalizedLandmarkList>(input_landmarks);
  output_landmarks->set_detection_confidence(confidence);

  cc->Outputs()
      .Tag(kOutLandmarks.Tag())
      .Add(output_landmarks.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

// Calculator for LandmarkList
class ConfidenceLandmarkMergerCalculator : public CalculatorBase {
 public:
  static constexpr Input<LandmarkList> kInLandmarks{"LANDMARKS"};
  static constexpr Input<float> kInConfidence{"CONFIDENCE"};
  static constexpr Output<LandmarkList> kOutLandmarks{"LANDMARKS"};

  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Process(CalculatorContext* cc) override;
};

absl::Status ConfidenceLandmarkMergerCalculator::GetContract(
    CalculatorContract* cc) {
  cc->Inputs().Tag(kInLandmarks.Tag()).Set<LandmarkList>();
  cc->Inputs().Tag(kInConfidence.Tag()).Set<float>();
  cc->Outputs().Tag(kOutLandmarks.Tag()).Set<LandmarkList>();
  return absl::OkStatus();
}

absl::Status ConfidenceLandmarkMergerCalculator::Process(
    CalculatorContext* cc) {
  if (cc->Inputs().Tag(kInLandmarks.Tag()).IsEmpty() ||
      cc->Inputs().Tag(kInConfidence.Tag()).IsEmpty()) {
    return absl::OkStatus();
  }

  const auto& input_landmarks = 
      cc->Inputs().Tag(kInLandmarks.Tag()).Get<LandmarkList>();
  const float confidence = 
      cc->Inputs().Tag(kInConfidence.Tag()).Get<float>();

  auto output_landmarks = absl::make_unique<LandmarkList>(input_landmarks);
  output_landmarks->set_detection_confidence(confidence);

  cc->Outputs()
      .Tag(kOutLandmarks.Tag())
      .Add(output_landmarks.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

// Registration for both calculators
REGISTER_CALCULATOR(ConfidenceNormalizedLandmarkMergerCalculator);
REGISTER_CALCULATOR(ConfidenceLandmarkMergerCalculator);

}  // namespace api2
}  // namespace mediapipe
