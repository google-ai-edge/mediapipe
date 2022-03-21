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

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/examples/desktop/autoflip/calculators/localization_to_region_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace autoflip {

// This calculator converts detections from ObjectLocalizationCalculator to
// SalientRegion protos that can be used for downstream processing.
class LocalizationToRegionCalculator : public mediapipe::CalculatorBase {
 public:
  LocalizationToRegionCalculator();
  ~LocalizationToRegionCalculator() override {}
  LocalizationToRegionCalculator(const LocalizationToRegionCalculator&) =
      delete;
  LocalizationToRegionCalculator& operator=(
      const LocalizationToRegionCalculator&) = delete;

  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;

 private:
  // Calculator options.
  LocalizationToRegionCalculatorOptions options_;
};
REGISTER_CALCULATOR(LocalizationToRegionCalculator);

LocalizationToRegionCalculator::LocalizationToRegionCalculator() {}

namespace {

constexpr char kRegionsTag[] = "REGIONS";
constexpr char kDetectionsTag[] = "DETECTIONS";

// Converts an object detection to a autoflip SignalType.  Returns true if the
// string label has a autoflip label.
bool MatchType(const std::string& label, SignalType* type) {
  if (label == "person") {
    type->set_standard(SignalType::HUMAN);
    return true;
  }
  if (label == "car" || label == "truck") {
    type->set_standard(SignalType::CAR);
    return true;
  }
  if (label == "dog" || label == "cat" || label == "bird" || label == "horse") {
    type->set_standard(SignalType::PET);
    return true;
  }
  return false;
}

// Converts a detection to a SalientRegion with a given label.
void FillSalientRegion(const mediapipe::Detection& detection,
                       const SignalType& label, SalientRegion* region) {
  const auto& location = detection.location_data().relative_bounding_box();
  region->mutable_location_normalized()->set_x(location.xmin());
  region->mutable_location_normalized()->set_y(location.ymin());
  region->mutable_location_normalized()->set_width(location.width());
  region->mutable_location_normalized()->set_height(location.height());
  region->set_score(1.0);
  *region->mutable_signal_type() = label;
}

}  // namespace

absl::Status LocalizationToRegionCalculator::GetContract(
    mediapipe::CalculatorContract* cc) {
  cc->Inputs().Tag(kDetectionsTag).Set<std::vector<mediapipe::Detection>>();
  cc->Outputs().Tag(kRegionsTag).Set<DetectionSet>();
  return absl::OkStatus();
}

absl::Status LocalizationToRegionCalculator::Open(
    mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<LocalizationToRegionCalculatorOptions>();

  return absl::OkStatus();
}

absl::Status LocalizationToRegionCalculator::Process(
    mediapipe::CalculatorContext* cc) {
  const auto& annotations =
      cc->Inputs().Tag(kDetectionsTag).Get<std::vector<mediapipe::Detection>>();
  auto regions = ::absl::make_unique<DetectionSet>();
  for (const auto& detection : annotations) {
    RET_CHECK_EQ(detection.label().size(), 1)
        << "Number of labels not equal to one.";
    SignalType autoflip_label;
    if (MatchType(detection.label(0), &autoflip_label) &&
        options_.output_standard_signals()) {
      FillSalientRegion(detection, autoflip_label, regions->add_detections());
    }
    if (options_.output_all_signals()) {
      SignalType object;
      object.set_standard(SignalType::OBJECT);
      FillSalientRegion(detection, object, regions->add_detections());
    }
  }

  cc->Outputs().Tag(kRegionsTag).Add(regions.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

}  // namespace autoflip
}  // namespace mediapipe
