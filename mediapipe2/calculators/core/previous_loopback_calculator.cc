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

#include <deque>

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace api2 {

// PreviousLoopbackCalculator is useful when a graph needs to process an input
// together with some previous output.
//
// For the first packet that arrives on the MAIN input, the timestamp bound is
// advanced on the PREV_LOOP. Downstream calculators will see this as an empty
// packet. This way they are not kept waiting for the previous output, which
// for the first iteration does not exist.
//
// Thereafter,
// - Each non-empty MAIN packet results in:
//   a) a PREV_LOOP packet with contents of the LOOP packet received at the
//      timestamp of the previous non-empty MAIN packet
//   b) or in a PREV_LOOP timestamp bound update if the LOOP packet was empty.
// - Each empty MAIN packet indicating timestamp bound update results in a
//   PREV_LOOP timestamp bound update.
//
// Example config:
// node {
//   calculator: "PreviousLoopbackCalculator"
//   input_stream: "MAIN:input"
//   input_stream: "LOOP:output"
//   input_stream_info: { tag_index: 'LOOP' back_edge: true }
//   output_stream: "PREV_LOOP:prev_output"
// }
// node {
//   calculator: "FaceTracker"
//   input_stream: "VIDEO:input"
//   input_stream: "PREV_TRACK:prev_output"
//   output_stream: "TRACK:output"
// }
class PreviousLoopbackCalculator : public Node {
 public:
  static constexpr Input<AnyType> kMain{"MAIN"};
  static constexpr Input<AnyType> kLoop{"LOOP"};
  static constexpr Output<SameType<kLoop>> kPrevLoop{"PREV_LOOP"};
  // TODO: an optional PREV_TIMESTAMP output could be added to
  // carry the original timestamp of the packet on PREV_LOOP.

  MEDIAPIPE_NODE_CONTRACT(kMain, kLoop, kPrevLoop,
                          StreamHandler("ImmediateInputStreamHandler"),
                          TimestampChange::Arbitrary());

  static absl::Status UpdateContract(CalculatorContract* cc) {
    // Process() function is invoked in response to MAIN/LOOP stream timestamp
    // bound updates.
    cc->SetProcessTimestampBounds(true);
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    kPrevLoop(cc).SetHeader(kLoop(cc).Header());
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    // Non-empty packets and empty packets indicating timestamp bound updates
    // are guaranteed to have timestamps greater than timestamps of previous
    // packets within the same stream. Calculator tracks and operates on such
    // packets.

    const PacketBase& main_packet = kMain(cc).packet();
    if (prev_main_ts_ < main_packet.timestamp()) {
      Timestamp loop_timestamp;
      if (!main_packet.IsEmpty()) {
        loop_timestamp = prev_non_empty_main_ts_;
        prev_non_empty_main_ts_ = main_packet.timestamp();
      } else {
        // Calculator advances PREV_LOOP timestamp bound in response to empty
        // MAIN packet, hence not caring about corresponding loop packet.
        loop_timestamp = Timestamp::Unset();
      }
      main_packet_specs_.push_back({main_packet.timestamp(), loop_timestamp});
      prev_main_ts_ = main_packet.timestamp();
    }

    const PacketBase& loop_packet = kLoop(cc).packet();
    if (prev_loop_ts_ < loop_packet.timestamp()) {
      loop_packets_.push_back(loop_packet);
      prev_loop_ts_ = loop_packet.timestamp();
    }

    while (!main_packet_specs_.empty() && !loop_packets_.empty()) {
      // The earliest MAIN packet.
      MainPacketSpec main_spec = main_packet_specs_.front();
      // The earliest LOOP packet.
      const PacketBase& loop_candidate = loop_packets_.front();
      // Match LOOP and MAIN packets.
      if (main_spec.loop_timestamp < loop_candidate.timestamp()) {
        // No LOOP packet can match the MAIN packet under review.
        kPrevLoop(cc).SetNextTimestampBound(main_spec.timestamp + 1);
        main_packet_specs_.pop_front();
      } else if (main_spec.loop_timestamp > loop_candidate.timestamp()) {
        // No MAIN packet can match the LOOP packet under review.
        loop_packets_.pop_front();
      } else {
        // Exact match found.
        if (loop_candidate.IsEmpty()) {
          // However, LOOP packet is empty.
          kPrevLoop(cc).SetNextTimestampBound(main_spec.timestamp + 1);
        } else {
          kPrevLoop(cc).Send(loop_candidate.At(main_spec.timestamp));
        }
        loop_packets_.pop_front();
        main_packet_specs_.pop_front();
      }

      // We can close PREV_LOOP output stream as soon as we processed last
      // possible MAIN packet. That can happen in two cases:
      // a) Non-empty MAIN packet has been received with Timestamp::Max()
      // b) Empty MAIN packet has been received with Timestamp::Max() indicating
      //    MAIN is done.
      if (main_spec.timestamp == Timestamp::Done().PreviousAllowedInStream()) {
        kPrevLoop(cc).Close();
      }
    }

    return absl::OkStatus();
  }

 private:
  struct MainPacketSpec {
    Timestamp timestamp;
    // Expected timestamp of the packet from LOOP stream that corresponds to the
    // packet from MAIN stream descirbed by this spec.
    Timestamp loop_timestamp;
  };

  // Contains specs for MAIN packets which only can be:
  // - non-empty packets
  // - empty packets indicating timestamp bound updates
  //
  // Sorted according to packet timestamps.
  std::deque<MainPacketSpec> main_packet_specs_;
  Timestamp prev_main_ts_ = Timestamp::Unstarted();
  Timestamp prev_non_empty_main_ts_ = Timestamp::Unstarted();

  // Contains LOOP packets which only can be:
  // - the very first empty packet
  // - non empty packets
  // - empty packets indicating timestamp bound updates
  //
  // Sorted according to packet timestamps.
  std::deque<PacketBase> loop_packets_;
  // Using "Timestamp::Unset" instead of "Timestamp::Unstarted" in order to
  // allow addition of the very first empty packet (which doesn't indicate
  // timestamp bound change necessarily).
  Timestamp prev_loop_ts_ = Timestamp::Unset();
};
MEDIAPIPE_REGISTER_NODE(PreviousLoopbackCalculator);

}  // namespace api2
}  // namespace mediapipe
