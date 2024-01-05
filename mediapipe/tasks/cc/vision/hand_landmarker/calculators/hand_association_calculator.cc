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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/rectangle.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/hand_association_calculator.pb.h"
#include "mediapipe/util/rectangle_util.h"

namespace mediapipe::api2 {

using ::mediapipe::NormalizedRect;

// Input:
//  BASE_RECTS - Vector of NormalizedRect.
//  RECTS - Vector of NormalizedRect.
//
// Output:
//  No tag - Vector of NormalizedRect.
//
// Example use:
// node {
//   calculator: "HandAssociationCalculator"
//   input_stream: "BASE_RECTS:base_rects"
//   input_stream: "RECTS:0:rects0"
//   input_stream: "RECTS:1:rects1"
//   input_stream: "RECTS:2:rects2"
//   output_stream: "output_rects"
//   options {
//     [mediapipe.HandAssociationCalculatorOptions.ext] {
//       min_similarity_threshold: 0.1
//   }
// }
//
// IMPORTANT Notes:
//  - Rects from input streams tagged with "BASE_RECTS" are always preserved.
//  - This calculator checks for overlap among rects from input streams tagged
//    with "RECTS". Rects are prioritized based on their index in the vector and
//    input streams to the calculator. When two rects overlap, the rect that
//    comes from an input stream with lower tag-index is kept in the output.
//  - Example of inputs for the node above:
//      "base_rects": rect 0, rect 1
//      "rects0": rect 2, rect 3
//      "rects1": rect 4, rect 5
//      "rects2": rect 6, rect 7
//    (Conceptually) flattened list: 0, 1, 2, 3, 4, 5, 6, 7.
//    Rects 0, 1 will be preserved. Rects 2, 3, 4, 5, 6, 7 will be checked for
//    overlap. If a rect with a higher index overlaps with a rect with lower
//    index, beyond a specified IOU threshold, the rect with the lower index
//    will be in the output, and the rect with higher index will be discarded.
// TODO: Upgrade this to latest API for calculators
class HandAssociationCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    // Initialize input and output streams.
    for (CollectionItemId id = cc->Inputs().BeginId("BASE_RECTS");
         id != cc->Inputs().EndId("BASE_RECTS"); ++id) {
      cc->Inputs().Get(id).Set<std::vector<NormalizedRect>>();
    }
    for (CollectionItemId id = cc->Inputs().BeginId("RECTS");
         id != cc->Inputs().EndId("RECTS"); ++id) {
      cc->Inputs().Get(id).Set<std::vector<NormalizedRect>>();
    }
    cc->Outputs().Index(0).Set<std::vector<NormalizedRect>>();

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));

    options_ = cc->Options<HandAssociationCalculatorOptions>();
    ABSL_CHECK_GT(options_.min_similarity_threshold(), 0.0);
    ABSL_CHECK_LE(options_.min_similarity_threshold(), 1.0);

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    MP_ASSIGN_OR_RETURN(auto result, GetNonOverlappingElements(cc));

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

    for (CollectionItemId id = cc->Inputs().BeginId("BASE_RECTS");
         id != cc->Inputs().EndId("BASE_RECTS"); ++id) {
      const auto& input_stream = cc->Inputs().Get(id);
      if (input_stream.IsEmpty()) {
        continue;
      }

      for (auto rect : input_stream.Get<std::vector<NormalizedRect>>()) {
        if (!rect.has_rect_id()) {
          rect.set_rect_id(GetNextRectId());
        }
        result.push_back(rect);
      }
    }

    for (CollectionItemId id = cc->Inputs().BeginId("RECTS");
         id != cc->Inputs().EndId("RECTS"); ++id) {
      const auto& input_stream = cc->Inputs().Get(id);
      if (input_stream.IsEmpty()) {
        continue;
      }

      for (auto rect : input_stream.Get<std::vector<NormalizedRect>>()) {
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

 private:
  // Each NormalizedRect processed by the calculator will be assigned
  // an unique id, if it does not already have an ID. The starting ID will be 1.
  // Note: This rect_id_ is local to an instance of this calculator. And it is
  // expected that the hand tracking graph to have only one instance of
  // this association calculator.
  int64_t rect_id_ = 1;

  inline int GetNextRectId() { return rect_id_++; }
};

MEDIAPIPE_REGISTER_NODE(HandAssociationCalculator);

}  // namespace mediapipe::api2
