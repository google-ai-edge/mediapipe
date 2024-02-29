#include "detection_classification_result_calculator.h"

#include "../utils/data_structures.h"

namespace mediapipe {
absl::Status DetectionClassificationResultCalculator::GetContract(
    CalculatorContract *cc) {
  cc->Inputs().Tag("DETECTION").Set<geti::InferenceResult>();
  cc->Inputs()
      .Tag("DETECTION_CLASSIFICATIONS")
      .Set<std::vector<geti::RectanglePrediction>>();
  cc->Outputs()
      .Tag("DETECTION_CLASSIFICATION_RESULT")
      .Set<geti::InferenceResult>();

  return absl::OkStatus();
}
absl::Status DetectionClassificationResultCalculator::Open(
    CalculatorContext *cc) {
  return absl::OkStatus();
}
absl::Status DetectionClassificationResultCalculator::GetiProcess(
    CalculatorContext *cc) {
  const auto &detection =
      cc->Inputs().Tag("DETECTION").Get<geti::InferenceResult>();

  if (!cc->Inputs().Tag("DETECTION_CLASSIFICATIONS").IsEmpty()) {
    const auto &classifications =
        cc->Inputs()
            .Tag("DETECTION_CLASSIFICATIONS")
            .Get<std::vector<geti::RectanglePrediction>>();

    if (classifications.size() > 0) {
      std::unique_ptr<geti::InferenceResult> result =
          std::make_unique<geti::InferenceResult>();

      result->rectangles = classifications;
      result->saliency_maps = detection.saliency_maps;
      result->roi = detection.roi;

      cc->Outputs()
          .Tag("DETECTION_CLASSIFICATION_RESULT")
          .Add(result.release(), cc->InputTimestamp());

      return absl::OkStatus();
    }
  }

  std::unique_ptr<geti::InferenceResult> result(
      new geti::InferenceResult(detection));

  cc->Outputs()
      .Tag("DETECTION_CLASSIFICATION_RESULT")
      .Add(result.release(), cc->InputTimestamp());

  return absl::OkStatus();
}
absl::Status DetectionClassificationResultCalculator::Close(
    CalculatorContext *cc) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(DetectionClassificationResultCalculator);

}  // namespace mediapipe
