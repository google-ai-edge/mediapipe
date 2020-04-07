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

#include <math.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/util/labels_to_render_data_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"

namespace mediapipe {

constexpr float kFontHeightScale = 1.25f;

// A calculator takes in pairs of labels and scores or classifications, outputs
// generates render data. Either both "LABELS" and "SCORES" or "CLASSIFICATIONS"
// must be present.
//
// Usage example:
// node {
//   calculator: "LabelsToRenderDataCalculator"
//   input_stream: "LABELS:labels"
//   input_stream: "SCORES:scores"
//   output_stream: "VIDEO_PRESTREAM:video_header"
//   options {
//     [LabelsToRenderDataCalculatorOptions.ext] {
//       color { r: 255 g: 0 b: 0 }
//       color { r: 0 g: 255 b: 0 }
//       color { r: 0 g: 0 b: 255 }
//       thickness: 2.0
//       font_height_px: 20
//       max_num_labels: 3
//       font_face: 1
//       location: TOP_LEFT
//     }
//   }
// }
class LabelsToRenderDataCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  LabelsToRenderDataCalculatorOptions options_;
  int num_colors_ = 0;
  int video_width_ = 0;
  int video_height_ = 0;
  int label_height_px_ = 0;
  int label_left_px_ = 0;
};
REGISTER_CALCULATOR(LabelsToRenderDataCalculator);

::mediapipe::Status LabelsToRenderDataCalculator::GetContract(
    CalculatorContract* cc) {
  if (cc->Inputs().HasTag("CLASSIFICATIONS")) {
    cc->Inputs().Tag("CLASSIFICATIONS").Set<ClassificationList>();
  } else {
    RET_CHECK(cc->Inputs().HasTag("LABELS"))
        << "Must provide input stream \"LABELS\"";
    cc->Inputs().Tag("LABELS").Set<std::vector<std::string>>();
    if (cc->Inputs().HasTag("SCORES")) {
      cc->Inputs().Tag("SCORES").Set<std::vector<float>>();
    }
  }
  if (cc->Inputs().HasTag("VIDEO_PRESTREAM")) {
    cc->Inputs().Tag("VIDEO_PRESTREAM").Set<VideoHeader>();
  }
  cc->Outputs().Tag("RENDER_DATA").Set<RenderData>();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status LabelsToRenderDataCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  options_ = cc->Options<LabelsToRenderDataCalculatorOptions>();
  num_colors_ = options_.color_size();
  label_height_px_ = std::ceil(options_.font_height_px() * kFontHeightScale);
  return ::mediapipe::OkStatus();
}

::mediapipe::Status LabelsToRenderDataCalculator::Process(
    CalculatorContext* cc) {
  if (cc->Inputs().HasTag("VIDEO_PRESTREAM") &&
      cc->InputTimestamp() == Timestamp::PreStream()) {
    const VideoHeader& video_header =
        cc->Inputs().Tag("VIDEO_PRESTREAM").Get<VideoHeader>();
    video_width_ = video_header.width;
    video_height_ = video_header.height;
    return ::mediapipe::OkStatus();
  } else {
    CHECK_EQ(options_.location(), LabelsToRenderDataCalculatorOptions::TOP_LEFT)
        << "Only TOP_LEFT is supported without VIDEO_PRESTREAM.";
  }

  std::vector<std::string> labels;
  std::vector<float> scores;
  if (cc->Inputs().HasTag("CLASSIFICATIONS")) {
    const ClassificationList& classifications =
        cc->Inputs().Tag("CLASSIFICATIONS").Get<ClassificationList>();
    labels.resize(classifications.classification_size());
    scores.resize(classifications.classification_size());
    for (int i = 0; i < classifications.classification_size(); ++i) {
      labels[i] = classifications.classification(i).label();
      scores[i] = classifications.classification(i).score();
    }
  } else {
    const std::vector<std::string>& label_vector =
        cc->Inputs().Tag("LABELS").Get<std::vector<std::string>>();
    labels.resize(label_vector.size());
    for (int i = 0; i < label_vector.size(); ++i) {
      labels[i] = label_vector[i];
    }

    if (cc->Inputs().HasTag("SCORES")) {
      std::vector<float> score_vector =
          cc->Inputs().Tag("SCORES").Get<std::vector<float>>();
      CHECK_EQ(label_vector.size(), score_vector.size());
      scores.resize(label_vector.size());
      for (int i = 0; i < label_vector.size(); ++i) {
        scores[i] = score_vector[i];
      }
    }
  }

  RenderData render_data;
  int num_label = std::min((int)labels.size(), options_.max_num_labels());
  int label_baseline_px = options_.vertical_offset_px();
  if (options_.location() == LabelsToRenderDataCalculatorOptions::TOP_LEFT) {
    label_baseline_px += label_height_px_;
  } else if (options_.location() ==
             LabelsToRenderDataCalculatorOptions::BOTTOM_LEFT) {
    label_baseline_px += video_height_ - label_height_px_ * (num_label - 1);
  }
  label_left_px_ = options_.horizontal_offset_px();
  for (int i = 0; i < num_label; ++i) {
    auto* label_annotation = render_data.add_render_annotations();
    label_annotation->set_thickness(options_.thickness());
    if (num_colors_ > 0) {
      *(label_annotation->mutable_color()) = options_.color(i % num_colors_);
    } else {
      label_annotation->mutable_color()->set_r(255);
      label_annotation->mutable_color()->set_g(0);
      label_annotation->mutable_color()->set_b(0);
    }

    auto* text = label_annotation->mutable_text();
    std::string display_text = labels[i];
    if (cc->Inputs().HasTag("SCORES")) {
      absl::StrAppend(&display_text, ":", scores[i]);
    }
    text->set_display_text(display_text);
    text->set_font_height(options_.font_height_px());
    text->set_left(label_left_px_);
    text->set_baseline(label_baseline_px + i * label_height_px_);
    text->set_font_face(options_.font_face());
  }
  cc->Outputs()
      .Tag("RENDER_DATA")
      .AddPacket(MakePacket<RenderData>(render_data).At(cc->InputTimestamp()));

  return ::mediapipe::OkStatus();
}
}  // namespace mediapipe
