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
#include <cmath>

#include "mediapipe/calculators/util/rect_transformation_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe {

namespace {

constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kNormRectsTag[] = "NORM_RECTS";
constexpr char kRectTag[] = "RECT";
constexpr char kRectsTag[] = "RECTS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";

using ::mediapipe::NormalizedRect;
using ::mediapipe::Rect;

// Wraps around an angle in radians to within -M_PI and M_PI.
inline float NormalizeRadians(float angle) {
  return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}

}  // namespace

// Performs geometric transformation to the input Rect or NormalizedRect,
// corresponding to input stream RECT or NORM_RECT respectively. When the input
// is NORM_RECT, an addition input stream IMAGE_SIZE is required, which is a
// std::pair<int, int> representing the image width and height.
//
// Example config:
// node {
//   calculator: "RectTransformationCalculator"
//   input_stream: "NORM_RECT:rect"
//   input_stream: "IMAGE_SIZE:image_size"
//   output_stream: "output_rect"
//   options: {
//     [mediapipe.RectTransformationCalculatorOptions.ext] {
//       scale_x: 2.6
//       scale_y: 2.6
//       shift_y: -0.5
//       square_long: true
//     }
//   }
// }
class RectTransformationCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  RectTransformationCalculatorOptions options_;

  float ComputeNewRotation(float rotation);
  void TransformRect(Rect* rect);
  void TransformNormalizedRect(NormalizedRect* rect, int image_width,
                               int image_height);
};
REGISTER_CALCULATOR(RectTransformationCalculator);

absl::Status RectTransformationCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK_EQ((cc->Inputs().HasTag(kNormRectTag) ? 1 : 0) +
                   (cc->Inputs().HasTag(kNormRectsTag) ? 1 : 0) +
                   (cc->Inputs().HasTag(kRectTag) ? 1 : 0) +
                   (cc->Inputs().HasTag(kRectsTag) ? 1 : 0),
               1);
  if (cc->Inputs().HasTag(kRectTag)) {
    cc->Inputs().Tag(kRectTag).Set<Rect>();
    cc->Outputs().Index(0).Set<Rect>();
  }
  if (cc->Inputs().HasTag(kRectsTag)) {
    cc->Inputs().Tag(kRectsTag).Set<std::vector<Rect>>();
    cc->Outputs().Index(0).Set<std::vector<Rect>>();
  }
  if (cc->Inputs().HasTag(kNormRectTag)) {
    RET_CHECK(cc->Inputs().HasTag(kImageSizeTag));
    cc->Inputs().Tag(kNormRectTag).Set<NormalizedRect>();
    cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
    cc->Outputs().Index(0).Set<NormalizedRect>();
  }
  if (cc->Inputs().HasTag(kNormRectsTag)) {
    RET_CHECK(cc->Inputs().HasTag(kImageSizeTag));
    cc->Inputs().Tag(kNormRectsTag).Set<std::vector<NormalizedRect>>();
    cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
    cc->Outputs().Index(0).Set<std::vector<NormalizedRect>>();
  }

  return absl::OkStatus();
}

absl::Status RectTransformationCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  options_ = cc->Options<RectTransformationCalculatorOptions>();
  RET_CHECK(!(options_.has_rotation() && options_.has_rotation_degrees()));
  RET_CHECK(!(options_.has_square_long() && options_.has_square_short()));

  return absl::OkStatus();
}

absl::Status RectTransformationCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kRectTag) && !cc->Inputs().Tag(kRectTag).IsEmpty()) {
    auto rect = cc->Inputs().Tag(kRectTag).Get<Rect>();
    TransformRect(&rect);
    cc->Outputs().Index(0).AddPacket(
        MakePacket<Rect>(rect).At(cc->InputTimestamp()));
  }
  if (cc->Inputs().HasTag(kRectsTag) &&
      !cc->Inputs().Tag(kRectsTag).IsEmpty()) {
    auto rects = cc->Inputs().Tag(kRectsTag).Get<std::vector<Rect>>();
    auto output_rects = absl::make_unique<std::vector<Rect>>(rects.size());
    for (int i = 0; i < rects.size(); ++i) {
      output_rects->at(i) = rects[i];
      auto it = output_rects->begin() + i;
      TransformRect(&(*it));
    }
    cc->Outputs().Index(0).Add(output_rects.release(), cc->InputTimestamp());
  }
  if (HasTagValue(cc->Inputs(), kNormRectTag) &&
      HasTagValue(cc->Inputs(), kImageSizeTag)) {
    auto rect = cc->Inputs().Tag(kNormRectTag).Get<NormalizedRect>();
    const auto& image_size =
        cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();
    TransformNormalizedRect(&rect, image_size.first, image_size.second);
    cc->Outputs().Index(0).AddPacket(
        MakePacket<NormalizedRect>(rect).At(cc->InputTimestamp()));
  }
  if (HasTagValue(cc->Inputs(), kNormRectsTag) &&
      HasTagValue(cc->Inputs(), kImageSizeTag)) {
    auto rects =
        cc->Inputs().Tag(kNormRectsTag).Get<std::vector<NormalizedRect>>();
    const auto& image_size =
        cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();
    auto output_rects =
        absl::make_unique<std::vector<NormalizedRect>>(rects.size());
    for (int i = 0; i < rects.size(); ++i) {
      output_rects->at(i) = rects[i];
      auto it = output_rects->begin() + i;
      TransformNormalizedRect(&(*it), image_size.first, image_size.second);
    }
    cc->Outputs().Index(0).Add(output_rects.release(), cc->InputTimestamp());
  }

  return absl::OkStatus();
}

float RectTransformationCalculator::ComputeNewRotation(float rotation) {
  if (options_.has_rotation()) {
    rotation += options_.rotation();
  } else if (options_.has_rotation_degrees()) {
    rotation += M_PI * options_.rotation_degrees() / 180.f;
  }
  return NormalizeRadians(rotation);
}

void RectTransformationCalculator::TransformRect(Rect* rect) {
  float width = rect->width();
  float height = rect->height();
  float rotation = rect->rotation();

  if (options_.has_rotation() || options_.has_rotation_degrees()) {
    rotation = ComputeNewRotation(rotation);
  }
  if (rotation == 0.f) {
    rect->set_x_center(rect->x_center() + width * options_.shift_x());
    rect->set_y_center(rect->y_center() + height * options_.shift_y());
  } else {
    const float x_shift = width * options_.shift_x() * std::cos(rotation) -
                          height * options_.shift_y() * std::sin(rotation);
    const float y_shift = width * options_.shift_x() * std::sin(rotation) +
                          height * options_.shift_y() * std::cos(rotation);
    rect->set_x_center(rect->x_center() + x_shift);
    rect->set_y_center(rect->y_center() + y_shift);
  }

  if (options_.square_long()) {
    const float long_side = std::max(width, height);
    width = long_side;
    height = long_side;
  } else if (options_.square_short()) {
    const float short_side = std::min(width, height);
    width = short_side;
    height = short_side;
  }
  rect->set_width(width * options_.scale_x());
  rect->set_height(height * options_.scale_y());
}

void RectTransformationCalculator::TransformNormalizedRect(NormalizedRect* rect,
                                                           int image_width,
                                                           int image_height) {
  float width = rect->width();
  float height = rect->height();
  float rotation = rect->rotation();

  if (options_.has_rotation() || options_.has_rotation_degrees()) {
    rotation = ComputeNewRotation(rotation);
  }
  if (rotation == 0.f) {
    rect->set_x_center(rect->x_center() + width * options_.shift_x());
    rect->set_y_center(rect->y_center() + height * options_.shift_y());
  } else {
    const float x_shift =
        (image_width * width * options_.shift_x() * std::cos(rotation) -
         image_height * height * options_.shift_y() * std::sin(rotation)) /
        image_width;
    const float y_shift =
        (image_width * width * options_.shift_x() * std::sin(rotation) +
         image_height * height * options_.shift_y() * std::cos(rotation)) /
        image_height;
    rect->set_x_center(rect->x_center() + x_shift);
    rect->set_y_center(rect->y_center() + y_shift);
  }

  if (options_.square_long()) {
    const float long_side =
        std::max(width * image_width, height * image_height);
    width = long_side / image_width;
    height = long_side / image_height;
  } else if (options_.square_short()) {
    const float short_side =
        std::min(width * image_width, height * image_height);
    width = short_side / image_width;
    height = short_side / image_height;
  }
  rect->set_width(width * options_.scale_x());
  rect->set_height(height * options_.scale_y());
}

}  // namespace mediapipe
