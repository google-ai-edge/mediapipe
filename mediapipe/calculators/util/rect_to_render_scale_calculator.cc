// Copyright 2020 The MediaPipe Authors.
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

#include "mediapipe/calculators/util/rect_to_render_scale_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe {

namespace {

constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kRenderScaleTag[] = "RENDER_SCALE";

using ::mediapipe::NormalizedRect;

}  // namespace

// A calculator to get scale for RenderData primitives.
//
// This calculator allows you to make RenderData primitives size (configured via
// `thickness`) to depend on actual size of the object they should highlight
// (e.g. pose, hand or face). It will give you bigger rendered primitives for
// bigger/closer objects and smaller primitives for smaller/far objects.
//
// IMPORTANT NOTE: RenderData primitives are rendered via OpenCV, which accepts
// only integer thickness. So when object goes further/closer you'll see 1 pixel
// jumps.
//
// Check `mediapipe/util/render_data.proto` for details on
// RenderData primitives and `thickness` parameter.
//
// Inputs:
//   NORM_RECT: Normalized rectangle to compute object size from as maximum of
//     width and height.
//   IMAGE_SIZE: A std::pair<int, int> represention of image width and height to
//     transform normalized object width and height to absolute pixel
//     coordinates.
//
// Outputs:
//   RENDER_SCALE: Float value that should be used to scale RenderData
//     primitives calculated as `rect_size * multiplier`.
//
// Example config:
//   node {
//     calculator: "RectToRenderScaleCalculator"
//     input_stream: "NORM_RECT:pose_landmarks_rect"
//     input_stream: "IMAGE_SIZE:image_size"
//     output_stream: "RENDER_SCALE:render_scale"
//     options: {
//       [mediapipe.RectToRenderScaleCalculatorOptions.ext] {
//         multiplier: 0.001
//       }
//     }
//   }
class RectToRenderScaleCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  RectToRenderScaleCalculatorOptions options_;
};
REGISTER_CALCULATOR(RectToRenderScaleCalculator);

absl::Status RectToRenderScaleCalculator::GetContract(CalculatorContract* cc) {
  cc->Inputs().Tag(kNormRectTag).Set<NormalizedRect>();
  cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
  cc->Outputs().Tag(kRenderScaleTag).Set<float>();
  cc->SetProcessTimestampBounds(
      cc->Options<RectToRenderScaleCalculatorOptions>()
          .process_timestamp_bounds());
  return absl::OkStatus();
}

absl::Status RectToRenderScaleCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  options_ = cc->Options<RectToRenderScaleCalculatorOptions>();

  return absl::OkStatus();
}

absl::Status RectToRenderScaleCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().Tag(kNormRectTag).IsEmpty()) {
    cc->Outputs()
        .Tag(kRenderScaleTag)
        .AddPacket(
            MakePacket<float>(options_.multiplier()).At(cc->InputTimestamp()));
    return absl::OkStatus();
  }

  // Get image size.
  int image_width;
  int image_height;
  std::tie(image_width, image_height) =
      cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();

  // Get rect size in absolute pixel coordinates.
  const auto& rect = cc->Inputs().Tag(kNormRectTag).Get<NormalizedRect>();
  const float rect_width = rect.width() * image_width;
  const float rect_height = rect.height() * image_height;

  // Calculate render scale.
  const float rect_size = std::max(rect_width, rect_height);
  const float render_scale = rect_size * options_.multiplier();

  cc->Outputs()
      .Tag(kRenderScaleTag)
      .AddPacket(MakePacket<float>(render_scale).At(cc->InputTimestamp()));

  return absl::OkStatus();
}

}  // namespace mediapipe
