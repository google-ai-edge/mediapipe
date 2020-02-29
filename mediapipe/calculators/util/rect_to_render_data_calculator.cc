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

#include "mediapipe/calculators/util/rect_to_render_data_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"

namespace mediapipe {

namespace {

constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kRectTag[] = "RECT";
constexpr char kNormRectsTag[] = "NORM_RECTS";
constexpr char kRectsTag[] = "RECTS";
constexpr char kRenderDataTag[] = "RENDER_DATA";

RenderAnnotation::Rectangle* NewRect(
    const RectToRenderDataCalculatorOptions& options, RenderData* render_data) {
  auto* annotation = render_data->add_render_annotations();
  annotation->mutable_color()->set_r(options.color().r());
  annotation->mutable_color()->set_g(options.color().g());
  annotation->mutable_color()->set_b(options.color().b());
  annotation->set_thickness(options.thickness());

  return options.filled()
             ? annotation->mutable_filled_rectangle()->mutable_rectangle()
             : annotation->mutable_rectangle();
}

void SetRect(bool normalized, double xmin, double ymin, double width,
             double height, double rotation,
             RenderAnnotation::Rectangle* rect) {
  if (rotation == 0.0) {
    if (xmin + width < 0.0 || ymin + height < 0.0) return;
    if (normalized) {
      if (xmin > 1.0 || ymin > 1.0) return;
    }
  }
  rect->set_normalized(normalized);
  rect->set_left(xmin);
  rect->set_top(ymin);
  rect->set_right(xmin + width);
  rect->set_bottom(ymin + height);
  rect->set_rotation(rotation);
}

}  // namespace

// Generates render data needed to render a rectangle in
// AnnotationOverlayCalculator.
//
// Input:
//   One of the following:
//   NORM_RECT: A NormalizedRect
//   RECT: A Rect
//   NORM_RECTS: An std::vector<NormalizedRect>
//   RECTS: An std::vector<Rect>
//
// Output:
//   RENDER_DATA: A RenderData
//
// Example config:
// node {
//   calculator: "RectToRenderDataCalculator"
//   input_stream: "NORM_RECT:rect"
//   output_stream: "RENDER_DATA:rect_render_data"
//   options: {
//     [mediapipe.RectToRenderDataCalculatorOptions.ext] {
//       filled: true
//       color { r: 255 g: 0 b: 0 }
//       thickness: 4.0
//     }
//   }
// }
class RectToRenderDataCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;

  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  RectToRenderDataCalculatorOptions options_;
};
REGISTER_CALCULATOR(RectToRenderDataCalculator);

::mediapipe::Status RectToRenderDataCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK_EQ((cc->Inputs().HasTag(kNormRectTag) ? 1 : 0) +
                   (cc->Inputs().HasTag(kRectTag) ? 1 : 0) +
                   (cc->Inputs().HasTag(kNormRectsTag) ? 1 : 0) +
                   (cc->Inputs().HasTag(kRectsTag) ? 1 : 0),
               1)
      << "Exactly one of NORM_RECT, RECT, NORM_RECTS or RECTS input stream "
         "should be provided.";
  RET_CHECK(cc->Outputs().HasTag(kRenderDataTag));

  if (cc->Inputs().HasTag(kNormRectTag)) {
    cc->Inputs().Tag(kNormRectTag).Set<NormalizedRect>();
  }
  if (cc->Inputs().HasTag(kRectTag)) {
    cc->Inputs().Tag(kRectTag).Set<Rect>();
  }
  if (cc->Inputs().HasTag(kNormRectsTag)) {
    cc->Inputs().Tag(kNormRectsTag).Set<std::vector<NormalizedRect>>();
  }
  if (cc->Inputs().HasTag(kRectsTag)) {
    cc->Inputs().Tag(kRectsTag).Set<std::vector<Rect>>();
  }
  cc->Outputs().Tag(kRenderDataTag).Set<RenderData>();

  return ::mediapipe::OkStatus();
}

::mediapipe::Status RectToRenderDataCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  options_ = cc->Options<RectToRenderDataCalculatorOptions>();

  return ::mediapipe::OkStatus();
}

::mediapipe::Status RectToRenderDataCalculator::Process(CalculatorContext* cc) {
  auto render_data = absl::make_unique<RenderData>();

  if (cc->Inputs().HasTag(kNormRectTag) &&
      !cc->Inputs().Tag(kNormRectTag).IsEmpty()) {
    const auto& rect = cc->Inputs().Tag(kNormRectTag).Get<NormalizedRect>();
    auto* rectangle = NewRect(options_, render_data.get());
    SetRect(/*normalized=*/true, rect.x_center() - rect.width() / 2.f,
            rect.y_center() - rect.height() / 2.f, rect.width(), rect.height(),
            rect.rotation(), rectangle);
  }
  if (cc->Inputs().HasTag(kRectTag) && !cc->Inputs().Tag(kRectTag).IsEmpty()) {
    const auto& rect = cc->Inputs().Tag(kRectTag).Get<Rect>();
    auto* rectangle = NewRect(options_, render_data.get());
    SetRect(/*normalized=*/false, rect.x_center() - rect.width() / 2.f,
            rect.y_center() - rect.height() / 2.f, rect.width(), rect.height(),
            rect.rotation(), rectangle);
  }
  if (cc->Inputs().HasTag(kNormRectsTag) &&
      !cc->Inputs().Tag(kNormRectsTag).IsEmpty()) {
    const auto& rects =
        cc->Inputs().Tag(kNormRectsTag).Get<std::vector<NormalizedRect>>();
    for (auto& rect : rects) {
      auto* rectangle = NewRect(options_, render_data.get());
      SetRect(/*normalized=*/true, rect.x_center() - rect.width() / 2.f,
              rect.y_center() - rect.height() / 2.f, rect.width(),
              rect.height(), rect.rotation(), rectangle);
    }
  }
  if (cc->Inputs().HasTag(kRectsTag) &&
      !cc->Inputs().Tag(kRectsTag).IsEmpty()) {
    const auto& rects = cc->Inputs().Tag(kRectsTag).Get<std::vector<Rect>>();
    for (auto& rect : rects) {
      auto* rectangle = NewRect(options_, render_data.get());
      SetRect(/*normalized=*/false, rect.x_center() - rect.width() / 2.f,
              rect.y_center() - rect.height() / 2.f, rect.width(),
              rect.height(), rect.rotation(), rectangle);
    }
  }

  cc->Outputs()
      .Tag(kRenderDataTag)
      .Add(render_data.release(), cc->InputTimestamp());

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
