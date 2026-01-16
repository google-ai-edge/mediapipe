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

#include "mediapipe/calculators/util/world_landmark_projection_calculator.h"

#include <cmath>
#include <functional>
#include <utility>

#include "absl/status/status.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe::api3 {

class WorldLandmarkProjectionNodeImpl
    : public Calculator<WorldLandmarkProjectionNode,
                        WorldLandmarkProjectionNodeImpl> {
 public:
  absl::Status Process(
      CalculatorContext<WorldLandmarkProjectionNode>& cc) final {
    // Check that landmarks and rect (if connected) are not empty.
    if (!cc.input_landmarks ||
        (cc.input_rect.IsConnected() && !cc.input_rect)) {
      return absl::OkStatus();
    }

    const auto& in_landmarks = cc.input_landmarks.GetOrDie();
    std::function<void(const Landmark&, Landmark*)> rotate_fn;
    if (cc.input_rect) {
      const auto& in_rect = cc.input_rect.GetOrDie();
      const float cosa = std::cos(in_rect.rotation());
      const float sina = std::sin(in_rect.rotation());
      rotate_fn = [cosa, sina](const Landmark& in_landmark,
                               Landmark* out_landmark) {
        out_landmark->set_x(cosa * in_landmark.x() - sina * in_landmark.y());
        out_landmark->set_y(sina * in_landmark.x() + cosa * in_landmark.y());
      };
    }

    LandmarkList out_landmarks;
    for (int i = 0; i < in_landmarks.landmark_size(); ++i) {
      const auto& in_landmark = in_landmarks.landmark(i);

      Landmark* out_landmark = out_landmarks.add_landmark();
      *out_landmark = in_landmark;

      if (rotate_fn) {
        rotate_fn(in_landmark, out_landmark);
      }
    }

    cc.output_landmarks.Send(std::move(out_landmarks));
    return absl::OkStatus();
  }
};

}  // namespace mediapipe::api3
