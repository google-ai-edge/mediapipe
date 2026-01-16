// Copyright 2021 The MediaPipe Authors.
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

#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "mediapipe/calculators/util/filter_detections_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

const char kInputDetectionsTag[] = "INPUT_DETECTIONS";
const char kImageSizeTag[] = "IMAGE_SIZE";  //  <width, height>
const char kOutputDetectionsTag[] = "OUTPUT_DETECTIONS";

//
// Calculator to filter out detections that do not meet the criteria specified
// in options.
//
class FilterDetectionsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().HasTag(kInputDetectionsTag));
    RET_CHECK(cc->Outputs().HasTag(kOutputDetectionsTag));

    cc->Inputs().Tag(kInputDetectionsTag).Set<std::vector<Detection>>();
    cc->Outputs().Tag(kOutputDetectionsTag).Set<std::vector<Detection>>();

    if (cc->Inputs().HasTag(kImageSizeTag)) {
      cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
    }

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    options_ = cc->Options<mediapipe::FilterDetectionsCalculatorOptions>();

    if (options_.has_min_pixel_size() || options_.has_max_pixel_size()) {
      RET_CHECK(cc->Inputs().HasTag(kImageSizeTag));
    }

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    const auto& input_detections =
        cc->Inputs().Tag(kInputDetectionsTag).Get<std::vector<Detection>>();
    auto output_detections = absl::make_unique<std::vector<Detection>>();

    int image_width = 0;
    int image_height = 0;
    if (cc->Inputs().HasTag(kImageSizeTag)) {
      std::tie(image_width, image_height) =
          cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();
    }

    for (const Detection& detection : input_detections) {
      if (options_.has_min_score()) {
        RET_CHECK_GT(detection.score_size(), 0);
        // Note: only score at index 0 supported.
        if (detection.score(0) < options_.min_score()) {
          continue;
        }
      }
      // Matches rect_size in
      // mediapipe/calculators/util/rect_to_render_scale_calculator.cc
      const float rect_size =
          std::max(detection.location_data().relative_bounding_box().width() *
                       image_width,
                   detection.location_data().relative_bounding_box().height() *
                       image_height);
      if (options_.has_min_pixel_size()) {
        if (rect_size < options_.min_pixel_size()) {
          continue;
        }
      }
      if (options_.has_max_pixel_size()) {
        if (rect_size > options_.max_pixel_size()) {
          continue;
        }
      }
      output_detections->push_back(detection);
    }

    cc->Outputs()
        .Tag(kOutputDetectionsTag)
        .Add(output_detections.release(), cc->InputTimestamp());

    return absl::OkStatus();
  }

 private:
  mediapipe::FilterDetectionsCalculatorOptions options_;
};

REGISTER_CALCULATOR(FilterDetectionsCalculator);

}  // namespace mediapipe
