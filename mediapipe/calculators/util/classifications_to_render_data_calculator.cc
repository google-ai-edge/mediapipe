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

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mediapipe/calculators/util/classifications_to_render_data_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"
namespace mediapipe {

namespace {

constexpr char kClassificationsTag[] = "CLASSIFICATIONS";
constexpr char kRenderDataTag[] = "RENDER_DATA";

constexpr char kSceneLabelLabel[] = "LABEL";

}  // namespace

// A calculator that converts Classification proto to RenderData proto for
// visualization.
//
// ClassificationList is the format for encoding one or more classifications of
// an image.
//
// The text(s) for "score label(_id)" will be shown starting on top left image
// corner.
//
// Example config:
// node {
//   calculator: "ClassificationsToRenderDataCalculator"
//   input_stream: "CLASSIFICATIONS:classifications"
//   output_stream: "RENDER_DATA:render_data"
//   options {
//     [ClassificationsToRenderDataCalculatorOptions.ext] {
//       text_delimiter: " <- "
//       thickness: 2.0
//       color { r: 0 g: 0 b: 255 }
//       text: { font_height: 2.0 }
//     }
//   }
// }
class ClassificationsToRenderDataCalculator : public CalculatorBase {
 public:
  ClassificationsToRenderDataCalculator() {}
  ~ClassificationsToRenderDataCalculator() override {}
  ClassificationsToRenderDataCalculator(
      const ClassificationsToRenderDataCalculator&) = delete;
  ClassificationsToRenderDataCalculator& operator=(
      const ClassificationsToRenderDataCalculator&) = delete;

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;

  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  // These utility methods are supposed to be used only by this class. No
  // external client should depend on them. Due to C++ style guide unnamed
  // namespace should not be used in header files. So, these has been defined
  // as private static methods.
  static void SetRenderAnnotationColorThickness(
      const ClassificationsToRenderDataCalculatorOptions& options,
      RenderAnnotation* render_annotation);

  static void SetTextCoordinate(bool normalized, double left, double baseline,
                                RenderAnnotation::Text* text);

  static void AddLabel(
      int ith, const Classification& classification,
      const ClassificationsToRenderDataCalculatorOptions& options,
      float text_line_height, RenderData* render_data);
};
REGISTER_CALCULATOR(ClassificationsToRenderDataCalculator);

::mediapipe::Status ClassificationsToRenderDataCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kClassificationsTag));
  cc->Inputs().Tag(kClassificationsTag).Set<ClassificationList>();
  cc->Outputs().Tag(kRenderDataTag).Set<RenderData>();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ClassificationsToRenderDataCalculator::Open(
    CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  return ::mediapipe::OkStatus();
}

::mediapipe::Status ClassificationsToRenderDataCalculator::Process(
    CalculatorContext* cc) {
  const auto& classifications = cc->Inputs()
                                    .Tag(kClassificationsTag)
                                    .Get<ClassificationList>()
                                    .classification();
  if (classifications.empty()) {
    return ::mediapipe::OkStatus();
  }

  const auto& options =
      cc->Options<ClassificationsToRenderDataCalculatorOptions>();

  auto render_data = absl::make_unique<RenderData>();
  render_data->set_scene_class(options.scene_class());

  auto text_line_height =
      (options.text().font_height() / (double)classifications.size()) / 10;

  int ith = 0;
  for (const auto& classification : classifications) {
    AddLabel(ith++, classification, options, text_line_height,
             render_data.get());
  }

  cc->Outputs()
      .Tag(kRenderDataTag)
      .Add(render_data.release(), cc->InputTimestamp());
  return ::mediapipe::OkStatus();
}

void ClassificationsToRenderDataCalculator::SetRenderAnnotationColorThickness(
    const ClassificationsToRenderDataCalculatorOptions& options,
    RenderAnnotation* render_annotation) {
  render_annotation->mutable_color()->set_r(options.color().r());
  render_annotation->mutable_color()->set_g(options.color().g());
  render_annotation->mutable_color()->set_b(options.color().b());
  render_annotation->set_thickness(options.thickness());
}

void ClassificationsToRenderDataCalculator::SetTextCoordinate(
    bool normalized, double left, double baseline,
    RenderAnnotation::Text* text) {
  text->set_normalized(normalized);
  text->set_left(normalized ? std::max(left, 0.0) : left);
  // Normalized coordinates must be between 0.0 and 1.0, if they are used.
  text->set_baseline(normalized ? std::min(baseline, 1.0) : baseline);
}

void ClassificationsToRenderDataCalculator::AddLabel(
    int ith, const Classification& classification,
    const ClassificationsToRenderDataCalculatorOptions& options,
    float text_line_height, RenderData* render_data) {
  std::string label = classification.label();
  if (label.empty()) {
    label = absl::StrCat("index=", classification.index());
  }
  std::string score_and_label =
      absl::StrCat(classification.score(), options.text_delimiter(), label);

  // Add the render annotations for "score label"
  auto* label_annotation = render_data->add_render_annotations();
  label_annotation->set_scene_tag(kSceneLabelLabel);
  SetRenderAnnotationColorThickness(options, label_annotation);
  auto* text = label_annotation->mutable_text();
  *text = options.text();
  text->set_display_text(score_and_label);
  text->set_font_height(text_line_height);
  SetTextCoordinate(true, 0.0, 0.0 + (ith + 1) * text_line_height, text);
}

}  // namespace mediapipe
