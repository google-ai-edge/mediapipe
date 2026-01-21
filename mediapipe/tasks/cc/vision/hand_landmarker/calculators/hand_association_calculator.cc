/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/hand_association_calculator.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/hand_association_calculator.pb.h"
#include "mediapipe/util/rectangle_util.h"

namespace mediapipe {
namespace tasks {
namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api3::Calculator;
using ::mediapipe::api3::CalculatorContext;

}  // namespace

class HandAssociationNodeImpl
    : public Calculator<HandAssociationNode, HandAssociationNodeImpl> {
 public:
  absl::Status Open(CalculatorContext<HandAssociationNode>& cc) override {
    options_ = cc.options.Get();
    ABSL_CHECK_GT(options_.min_similarity_threshold(), 0.0);
    ABSL_CHECK_LE(options_.min_similarity_threshold(), 1.0);

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext<HandAssociationNode>& cc) override {
    MP_ASSIGN_OR_RETURN(auto result, GetNonOverlappingElements(cc));

    auto output =
        std::make_unique<std::vector<NormalizedRect>>(std::move(result));
    cc.output_rects.Send(std::move(output));

    return absl::OkStatus();
  }

 private:
  // HandAssociationCalculatorOptions from the calculator options.
  HandAssociationCalculatorOptions options_;

  // Each NormalizedRect processed by the calculator will be assigned
  // an unique id, if it does not already have an ID. The starting ID will be
  // 1. Note: This rect_id_ is local to an instance of this calculator. And it
  // is expected that the hand tracking graph to have only one instance of this
  // association calculator.
  int64_t rect_id_ = 1;

  inline int GetNextRectId() { return rect_id_++; }

  // Return a list of non-overlapping elements from all input streams, with
  // decreasing order of priority based on input stream index and indices
  // within an input stream.
  absl::StatusOr<std::vector<NormalizedRect>> GetNonOverlappingElements(
      CalculatorContext<HandAssociationNode>& cc) {
    std::vector<NormalizedRect> result;

    for (int i = 0; i < cc.base_rects.Count(); ++i) {
      const auto& base_rects_input_stream = cc.base_rects.At(i);
      if (!base_rects_input_stream) {
        continue;
      }
      for (auto rect : base_rects_input_stream.GetOrDie()) {
        if (!rect.has_rect_id()) {
          rect.set_rect_id(GetNextRectId());
        }
        result.push_back(rect);
      }
    }

    for (int i = 0; i < cc.rects.Count(); ++i) {
      const auto& rects_input_stream = cc.rects.At(i);
      if (!rects_input_stream) {
        continue;
      }

      for (auto rect : rects_input_stream.GetOrDie()) {
        MP_ASSIGN_OR_RETURN(
            bool is_overlapping,
            mediapipe::DoesRectOverlap(rect, result,
                                       options_.min_similarity_threshold()));
        if (!is_overlapping) {
          if (!rect.has_rect_id()) {
            rect.set_rect_id(GetNextRectId());
          }
          result.push_back(rect);
        }
      }
    }

    return result;
  }
};

}  // namespace tasks
}  // namespace mediapipe
