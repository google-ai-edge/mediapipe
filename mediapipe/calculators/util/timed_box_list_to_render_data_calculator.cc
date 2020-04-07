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

#include <algorithm>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mediapipe/calculators/util/timed_box_list_to_render_data_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"
#include "mediapipe/util/tracking/box_tracker.pb.h"
#include "mediapipe/util/tracking/tracking.pb.h"

namespace mediapipe {

namespace {

constexpr char kTimedBoxListTag[] = "BOX_LIST";
constexpr char kRenderDataTag[] = "RENDER_DATA";

void AddTimedBoxProtoToRenderData(
    const TimedBoxProto& box_proto,
    const TimedBoxListToRenderDataCalculatorOptions& options,
    RenderData* render_data) {
  if (box_proto.has_quad() && box_proto.quad().vertices_size() > 0 &&
      box_proto.quad().vertices_size() % 2 == 0) {
    const int num_corners = box_proto.quad().vertices_size() / 2;
    for (int i = 0; i < num_corners; ++i) {
      const int next_corner = (i + 1) % num_corners;
      auto* line_annotation = render_data->add_render_annotations();
      line_annotation->mutable_color()->set_r(options.box_color().r());
      line_annotation->mutable_color()->set_g(options.box_color().g());
      line_annotation->mutable_color()->set_b(options.box_color().b());
      line_annotation->set_thickness(options.thickness());
      RenderAnnotation::Line* line = line_annotation->mutable_line();
      line->set_normalized(true);
      line->set_x_start(box_proto.quad().vertices(i * 2));
      line->set_y_start(box_proto.quad().vertices(i * 2 + 1));
      line->set_x_end(box_proto.quad().vertices(next_corner * 2));
      line->set_y_end(box_proto.quad().vertices(next_corner * 2 + 1));
    }
  } else {
    auto* rect_annotation = render_data->add_render_annotations();
    rect_annotation->mutable_color()->set_r(options.box_color().r());
    rect_annotation->mutable_color()->set_g(options.box_color().g());
    rect_annotation->mutable_color()->set_b(options.box_color().b());
    rect_annotation->set_thickness(options.thickness());
    RenderAnnotation::Rectangle* rect = rect_annotation->mutable_rectangle();
    rect->set_normalized(true);
    rect->set_left(box_proto.left());
    rect->set_right(box_proto.right());
    rect->set_top(box_proto.top());
    rect->set_bottom(box_proto.bottom());
    rect->set_rotation(box_proto.rotation());
  }

  if (box_proto.has_label()) {
    auto* label_annotation = render_data->add_render_annotations();
    label_annotation->mutable_color()->set_r(options.box_color().r());
    label_annotation->mutable_color()->set_g(options.box_color().g());
    label_annotation->mutable_color()->set_b(options.box_color().b());
    label_annotation->set_thickness(options.thickness());
    RenderAnnotation::Text* text = label_annotation->mutable_text();
    text->set_display_text(box_proto.label());
    text->set_normalized(true);
    constexpr float text_left_start = 0.2f;
    text->set_left((1.0f - text_left_start) * box_proto.left() +
                   text_left_start * box_proto.right());
    constexpr float text_baseline = 0.6f;
    text->set_baseline(text_baseline * box_proto.bottom() +
                       (1.0f - text_baseline) * box_proto.top());
    constexpr float text_height = 0.1f;
    text->set_font_height(std::min(box_proto.bottom() - box_proto.top(),
                                   box_proto.right() - box_proto.left()) *
                          text_height);
  }
}

}  // namespace

// A calculator that converts TimedBoxProtoList proto to RenderData proto for
// visualization. If the input TimedBoxProto contains `quad` field, this
// calculator will draw a quadrilateral based on it. Otherwise this calculator
// will draw a rotated rectangle based on `top`, `bottom`, `left`, `right` and
// `rotation` fields
//
// Example config:
// node {
//   calculator: "TimedBoxListToRenderDataCalculator"
//   input_stream: "BOX_LIST:landmarks"
//   output_stream: "RENDER_DATA:render_data"
//   options {
//     [TimedBoxListToRenderDataCalculatorOptions.ext] {
//       box_color { r: 0 g: 255 b: 0 }
//       thickness: 4.0
//     }
//   }
// }
class TimedBoxListToRenderDataCalculator : public CalculatorBase {
 public:
  TimedBoxListToRenderDataCalculator() {}
  ~TimedBoxListToRenderDataCalculator() override {}
  TimedBoxListToRenderDataCalculator(
      const TimedBoxListToRenderDataCalculator&) = delete;
  TimedBoxListToRenderDataCalculator& operator=(
      const TimedBoxListToRenderDataCalculator&) = delete;

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;

  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  TimedBoxListToRenderDataCalculatorOptions options_;
};
REGISTER_CALCULATOR(TimedBoxListToRenderDataCalculator);

::mediapipe::Status TimedBoxListToRenderDataCalculator::GetContract(
    CalculatorContract* cc) {
  if (cc->Inputs().HasTag(kTimedBoxListTag)) {
    cc->Inputs().Tag(kTimedBoxListTag).Set<TimedBoxProtoList>();
  }
  cc->Outputs().Tag(kRenderDataTag).Set<RenderData>();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TimedBoxListToRenderDataCalculator::Open(
    CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  options_ = cc->Options<TimedBoxListToRenderDataCalculatorOptions>();

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TimedBoxListToRenderDataCalculator::Process(
    CalculatorContext* cc) {
  auto render_data = absl::make_unique<RenderData>();

  if (cc->Inputs().HasTag(kTimedBoxListTag)) {
    const auto& box_list =
        cc->Inputs().Tag(kTimedBoxListTag).Get<TimedBoxProtoList>();

    for (const auto& box : box_list.box()) {
      AddTimedBoxProtoToRenderData(box, options_, render_data.get());
    }
  }

  cc->Outputs()
      .Tag(kRenderDataTag)
      .Add(render_data.release(), cc->InputTimestamp());
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
