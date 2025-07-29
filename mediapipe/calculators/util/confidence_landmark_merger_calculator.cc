#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "absl/status/status.h"

namespace mediapipe {
namespace api2 {

template <typename LandmarkType>
class ConfidenceMergerCalculator : public CalculatorBase {
 public:
  static constexpr Input<LandmarkType> kInLandmarks{"LANDMARKS"};
  static constexpr Input<float> kInConfidence{"CONFIDENCE"};
  static constexpr Output<LandmarkType> kOutLandmarks{"LANDMARKS"};

  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kInLandmarks.Tag()).template Set<LandmarkType>();
    cc->Inputs().Tag(kInConfidence.Tag()).template Set<float>();
    cc->Outputs().Tag(kOutLandmarks.Tag()).template Set<LandmarkType>();
    return absl::OkStatus();
  }
  absl::Status Open(CalculatorContext* cc) final {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }
  absl::Status Process(CalculatorContext* cc) override {
    if (cc -> Inputs().Tag(kInLandmarks.Tag()).IsEmpty()|| cc -> Inputs().Tag(kInConfidence.Tag()).IsEmpty()) {
    } else {
      const auto& input_landmarks = 
          cc->Inputs().Tag(kInLandmarks.Tag()).template Get<LandmarkType>();
      const float confidence = 
          cc->Inputs().Tag(kInConfidence.Tag()).template Get<float>();
        
      auto output_landmarks = 
          absl::make_unique<LandmarkType>(input_landmarks);

      output_landmarks->set_detection_confidence(confidence);
      output_landmarks->set_has_detection_confidence_set(true);

      cc->Outputs()
          .Tag(kOutLandmarks.Tag())
          .Add(output_landmarks.release(), cc->InputTimestamp()); 
    }
    return absl::OkStatus();

  }
};

using ConfidenceNormalizedLandmarkMergerCalculator = ConfidenceMergerCalculator<NormalizedLandmarkList>;
using ConfidenceLandmarkMergerCalculator = ConfidenceMergerCalculator<LandmarkList>;

REGISTER_CALCULATOR(ConfidenceNormalizedLandmarkMergerCalculator);
REGISTER_CALCULATOR(ConfidenceLandmarkMergerCalculator);

}  // namespace api2
}  // namespace mediapipe
