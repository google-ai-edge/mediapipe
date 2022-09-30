/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

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

#include <memory>
#include <utility>
#include <vector>

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/rectangle.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/hand_association_calculator.pb.h"
#include "mediapipe/util/rectangle_util.h"

namespace mediapipe::api2 {

// HandAssociationCalculator accepts multiple inputs of vectors of
// NormalizedRect. The output is a vector of NormalizedRect that contains
// rects from the input vectors that don't overlap with each other. When two
// rects overlap, the rect that comes in from an earlier input stream is
// kept in the output. If a rect has no ID (i.e. from detection stream),
// then a unique rect ID is assigned for it.

// The rects in multiple input streams are effectively flattened to a single
// list.  For example:
// Stream1 : rect 1, rect 2
// Stream2:  rect 3, rect 4
// Stream3: rect 5, rect 6
// (Conceptually) flattened list : rect 1, 2, 3, 4, 5, 6
// In the flattened list, if a rect with a higher index overlaps with a rect a
// lower index, beyond a specified IOU threshold, the rect with the lower
// index will be in the output, and the rect with higher index will be
// discarded.
// TODO: Upgrade this to latest API for calculators
class HandAssociationCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    // Initialize input and output streams.
    for (auto& input_stream : cc->Inputs()) {
      input_stream.Set<std::vector<NormalizedRect>>();
    }
    cc->Outputs().Index(0).Set<std::vector<NormalizedRect>>();

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));

    options_ = cc->Options<HandAssociationCalculatorOptions>();
    CHECK_GT(options_.min_similarity_threshold(), 0.0);
    CHECK_LE(options_.min_similarity_threshold(), 1.0);

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    ASSIGN_OR_RETURN(auto result, GetNonOverlappingElements(cc));

    auto output =
        std::make_unique<std::vector<NormalizedRect>>(std::move(result));
    cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());

    return absl::OkStatus();
  }

 private:
  HandAssociationCalculatorOptions options_;

  // Return a list of non-overlapping elements from all input streams, with
  // decreasing order of priority based on input stream index and indices
  // within an input stream.
  absl::StatusOr<std::vector<NormalizedRect>> GetNonOverlappingElements(
      CalculatorContext* cc) {
    std::vector<NormalizedRect> result;

    for (const auto& input_stream : cc->Inputs()) {
      if (input_stream.IsEmpty()) {
        continue;
      }

      for (auto rect : input_stream.Get<std::vector<NormalizedRect>>()) {
        ASSIGN_OR_RETURN(
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

 private:
  // Each NormalizedRect processed by the calculator will be assigned
  // an unique id, if it does not already have an ID. The starting ID will be 1.
  // Note: This rect_id_ is local to an instance of this calculator. And it is
  // expected that the hand tracking graph to have only one instance of
  // this association calculator.
  int64 rect_id_ = 1;

  inline int GetNextRectId() { return rect_id_++; }
};

MEDIAPIPE_REGISTER_NODE(HandAssociationCalculator);

}  // namespace mediapipe::api2
