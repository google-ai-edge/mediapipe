// Copyright 2020 Google LLC
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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/graphs/instant_motion_tracking/calculators/transformations.h"
#include "mediapipe/util/tracking/box_tracker.pb.h"

namespace mediapipe {

constexpr char kSentinelTag[] = "SENTINEL";
constexpr char kAnchorsTag[] = "ANCHORS";
constexpr char kBoxesInputTag[] = "BOXES";
constexpr char kBoxesOutputTag[] = "START_POS";
constexpr char kCancelTag[] = "CANCEL_ID";
// TODO: Find optimal Height/Width (0.1-0.3)
constexpr float kBoxEdgeSize =
    0.2f;  // Used to establish tracking box dimensions
constexpr float kUsToMs =
    1000.0f;  // Used to convert from microseconds to millis

// This calculator manages the regions being tracked for each individual sticker
// and adjusts the regions being tracked if a change is detected in a sticker's
// initial anchor placement. Regions being tracked that have no associated
// sticker will be automatically removed upon the next iteration of the graph to
// optimize performance and remove all sticker artifacts
//
// Input:
//  SENTINEL - ID of sticker which has an anchor that must be reset (-1 when no
//  anchor must be reset) [REQUIRED]
//  ANCHORS - Initial anchor data (tracks changes and where to re/position)
//  [REQUIRED] BOXES - Used in cycle, boxes being tracked meant to update
//  positions [OPTIONAL
//  - provided by subgraph]
// Output:
//  START_POS - Positions of boxes being tracked (can be overwritten with ID)
//  [REQUIRED] CANCEL_ID - Single integer ID of tracking box to remove from
//  tracker subgraph [OPTIONAL] ANCHORS - Updated set of anchors with tracked
//  and normalized X,Y,Z [REQUIRED]
//
// Example config:
// node {
//   calculator: "TrackedAnchorManagerCalculator"
//   input_stream: "SENTINEL:sticker_sentinel"
//   input_stream: "ANCHORS:initial_anchor_data"
//   input_stream: "BOXES:boxes"
//   input_stream_info: {
//     tag_index: 'BOXES'
//     back_edge: true
//   }
//   output_stream: "START_POS:start_pos"
//   output_stream: "CANCEL_ID:cancel_object_id"
//   output_stream: "ANCHORS:tracked_scaled_anchor_data"
// }

class TrackedAnchorManagerCalculator : public CalculatorBase {
 private:
  // Previous graph iteration anchor data
  std::vector<Anchor> previous_anchor_data_;

 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().HasTag(kAnchorsTag) &&
              cc->Inputs().HasTag(kSentinelTag));
    RET_CHECK(cc->Outputs().HasTag(kAnchorsTag) &&
              cc->Outputs().HasTag(kBoxesOutputTag));

    cc->Inputs().Tag(kAnchorsTag).Set<std::vector<Anchor>>();
    cc->Inputs().Tag(kSentinelTag).Set<int>();

    if (cc->Inputs().HasTag(kBoxesInputTag)) {
      cc->Inputs().Tag(kBoxesInputTag).Set<mediapipe::TimedBoxProtoList>();
    }

    cc->Outputs().Tag(kAnchorsTag).Set<std::vector<Anchor>>();
    cc->Outputs().Tag(kBoxesOutputTag).Set<mediapipe::TimedBoxProtoList>();

    if (cc->Outputs().HasTag(kCancelTag)) {
      cc->Outputs().Tag(kCancelTag).Set<int>();
    }

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override;
};
REGISTER_CALCULATOR(TrackedAnchorManagerCalculator);

::mediapipe::Status TrackedAnchorManagerCalculator::Process(
    CalculatorContext* cc) {
  mediapipe::Timestamp timestamp = cc->InputTimestamp();
  const int sticker_sentinel = cc->Inputs().Tag(kSentinelTag).Get<int>();
  std::vector<Anchor> current_anchor_data =
      cc->Inputs().Tag(kAnchorsTag).Get<std::vector<Anchor>>();
  auto pos_boxes = absl::make_unique<mediapipe::TimedBoxProtoList>();
  std::vector<Anchor> tracked_scaled_anchor_data;

  // Delete any boxes being tracked without an associated anchor
  for (const mediapipe::TimedBoxProto& box :
       cc->Inputs()
           .Tag(kBoxesInputTag)
           .Get<mediapipe::TimedBoxProtoList>()
           .box()) {
    bool anchor_exists = false;
    for (Anchor anchor : current_anchor_data) {
      if (box.id() == anchor.sticker_id) {
        anchor_exists = true;
        break;
      }
    }
    if (!anchor_exists) {
      cc->Outputs()
          .Tag(kCancelTag)
          .AddPacket(MakePacket<int>(box.id()).At(timestamp++));
    }
  }

  // Perform tracking or updating for each anchor position
  for (const Anchor& anchor : current_anchor_data) {
    Anchor output_anchor = anchor;
    // Check if anchor position is being reset by user in this graph iteration
    if (sticker_sentinel == anchor.sticker_id) {
      // Delete associated tracking box
      // TODO: BoxTrackingSubgraph should accept vector to avoid breaking
      // timestamp rules
      cc->Outputs()
          .Tag(kCancelTag)
          .AddPacket(MakePacket<int>(anchor.sticker_id).At(timestamp++));
      // Add a tracking box
      mediapipe::TimedBoxProto* box = pos_boxes->add_box();
      box->set_left(anchor.x - kBoxEdgeSize * 0.5f);
      box->set_right(anchor.x + kBoxEdgeSize * 0.5f);
      box->set_top(anchor.y - kBoxEdgeSize * 0.5f);
      box->set_bottom(anchor.y + kBoxEdgeSize * 0.5f);
      box->set_id(anchor.sticker_id);
      box->set_time_msec((timestamp++).Microseconds() / kUsToMs);
      // Default value for normalized z (scale factor)
      output_anchor.z = 1.0f;
    } else {
      // Anchor position was not reset by user
      // Attempt to update anchor position from tracking subgraph
      // (TimedBoxProto)
      bool updated_from_tracker = false;
      const mediapipe::TimedBoxProtoList box_list =
          cc->Inputs().Tag(kBoxesInputTag).Get<mediapipe::TimedBoxProtoList>();
      for (const auto& box : box_list.box()) {
        if (box.id() == anchor.sticker_id) {
          // Get center x normalized coordinate [0.0-1.0]
          output_anchor.x = (box.left() + box.right()) * 0.5f;
          // Get center y normalized coordinate [0.0-1.0]
          output_anchor.y = (box.top() + box.bottom()) * 0.5f;
          // Get center z coordinate [z starts at normalized 1.0 and scales
          // inversely with box-width]
          // TODO: Look into issues with uniform scaling on x-axis and y-axis
          output_anchor.z = kBoxEdgeSize / (box.right() - box.left());
          updated_from_tracker = true;
          break;
        }
      }
      // If anchor position was not updated from tracker, create new tracking
      // box at last recorded anchor coordinates. This will allow all current
      // stickers to be tracked at approximately last location even if
      // re-acquisitioning in the BoxTrackingSubgraph encounters errors
      if (!updated_from_tracker) {
        for (const Anchor& prev_anchor : previous_anchor_data_) {
          if (anchor.sticker_id == prev_anchor.sticker_id) {
            mediapipe::TimedBoxProto* box = pos_boxes->add_box();
            box->set_left(prev_anchor.x - kBoxEdgeSize * 0.5f);
            box->set_right(prev_anchor.x + kBoxEdgeSize * 0.5f);
            box->set_top(prev_anchor.y - kBoxEdgeSize * 0.5f);
            box->set_bottom(prev_anchor.y + kBoxEdgeSize * 0.5f);
            box->set_id(prev_anchor.sticker_id);
            box->set_time_msec(cc->InputTimestamp().Microseconds() / kUsToMs);
            output_anchor = prev_anchor;
            // Default value for normalized z (scale factor)
            output_anchor.z = 1.0f;
            break;
          }
        }
      }
    }
    tracked_scaled_anchor_data.emplace_back(output_anchor);
  }
  // Set anchor data for next iteration
  previous_anchor_data_ = tracked_scaled_anchor_data;

  cc->Outputs()
      .Tag(kAnchorsTag)
      .AddPacket(MakePacket<std::vector<Anchor>>(tracked_scaled_anchor_data)
                     .At(cc->InputTimestamp()));
  cc->Outputs()
      .Tag(kBoxesOutputTag)
      .Add(pos_boxes.release(), cc->InputTimestamp());

  return ::mediapipe::OkStatus();
}
}  // namespace mediapipe
