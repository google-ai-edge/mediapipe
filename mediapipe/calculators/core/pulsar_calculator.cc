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

#include "mediapipe/calculators/core/pulsar_calculator.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

// Example use-case:
// Suppose we are debugging a pipeline where a vector of NormalizedRect has to
// be passed to a ImageCroppingCalculator somehow, which only takes non-vectors.
//
// Example config:
// node {
//   calculator: "NormalizedRectsPulsarCalculator"
//   input_stream: "head_rects"
//   output_stream: "head_rect"
// }
//
// node {
//   calculator: "ImageCroppingCalculator"
//   input_stream: "IMAGE:throttled_input_video"
//   input_stream: "NORM_RECT:head_rect"
//   output_stream: "IMAGE:cropped_head"
// }
typedef PulsarCalculator<NormalizedRect> NormalizedRectsPulsarCalculator;
REGISTER_CALCULATOR(NormalizedRectsPulsarCalculator);

typedef PulsarCalculator<Rect> RectsPulsarCalculator;
REGISTER_CALCULATOR(RectsPulsarCalculator);

}  // namespace mediapipe
