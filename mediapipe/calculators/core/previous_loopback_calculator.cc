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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {

// PreviousLoopbackCalculator is useful when a graph needs to process an input
// together with some previous output.
//
// For the first packet that arrives on the MAIN input, the timestamp bound is
// advanced on the output. Downstream calculators will see this as an empty
// packet. This way they are not kept waiting for the previous output, which
// for the first iteration does not exist.
//
// Thereafter, each packet received on MAIN is matched with a packet received
// on LOOP; the LOOP packet's timestamp is changed to that of the MAIN packet,
// and it is output on PREV_LOOP.
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
class PreviousLoopbackCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Get("MAIN", 0).SetAny();
    cc->Inputs().Get("LOOP", 0).SetAny();
    cc->Outputs().Get("PREV_LOOP", 0).SetSameAs(&(cc->Inputs().Get("LOOP", 0)));
    // TODO: an optional PREV_TIMESTAMP output could be added to
    // carry the original timestamp of the packet on PREV_LOOP.
    cc->SetInputStreamHandler("ImmediateInputStreamHandler");
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    main_id_ = cc->Inputs().GetId("MAIN", 0);
    loop_id_ = cc->Inputs().GetId("LOOP", 0);
    loop_out_id_ = cc->Outputs().GetId("PREV_LOOP", 0);
    cc->Outputs()
        .Get(loop_out_id_)
        .SetHeader(cc->Inputs().Get(loop_id_).Header());

    // Use an empty packet for the first round, since there is no previous
    // output.
    loopback_packets_.push_back({});

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    Packet& main_packet = cc->Inputs().Get(main_id_).Value();
    if (!main_packet.IsEmpty()) {
      main_ts_.push_back(main_packet.Timestamp());
    }
    Packet& loopback_packet = cc->Inputs().Get(loop_id_).Value();
    if (!loopback_packet.IsEmpty()) {
      loopback_packets_.push_back(loopback_packet);
      while (!main_ts_.empty() &&
             main_ts_.front() <= loopback_packets_.front().Timestamp()) {
        main_ts_.pop_front();
      }
    }
    auto& loop_out = cc->Outputs().Get(loop_out_id_);

    while (!main_ts_.empty() && !loopback_packets_.empty()) {
      Timestamp main_timestamp = main_ts_.front();
      main_ts_.pop_front();
      Packet previous_loopback = loopback_packets_.front().At(main_timestamp);
      loopback_packets_.pop_front();

      if (previous_loopback.IsEmpty()) {
        // TODO: SetCompleteTimestampBound would be more useful.
        loop_out.SetNextTimestampBound(main_timestamp + 1);
      } else {
        loop_out.AddPacket(std::move(previous_loopback));
      }
    }

    // In case of an empty loopback input, the next timestamp bound for
    // loopback input is the loopback timestamp + 1.  The next timestamp bound
    // for output is set and the main_ts_ vector is truncated accordingly.
    if (loopback_packet.IsEmpty() &&
        loopback_packet.Timestamp() != Timestamp::Unstarted()) {
      Timestamp loopback_bound =
          loopback_packet.Timestamp().NextAllowedInStream();
      while (!main_ts_.empty() && main_ts_.front() <= loopback_bound) {
        main_ts_.pop_front();
      }
      if (main_ts_.empty()) {
        loop_out.SetNextTimestampBound(loopback_bound.NextAllowedInStream());
      }
    }
    if (!main_ts_.empty()) {
      loop_out.SetNextTimestampBound(main_ts_.front());
    }
    if (cc->Inputs().Get(main_id_).IsDone() && main_ts_.empty()) {
      loop_out.Close();
    }
    return ::mediapipe::OkStatus();
  }

 private:
  CollectionItemId main_id_;
  CollectionItemId loop_id_;
  CollectionItemId loop_out_id_;

  std::deque<Timestamp> main_ts_;
  std::deque<Packet> loopback_packets_;
};
REGISTER_CALCULATOR(PreviousLoopbackCalculator);

}  // namespace mediapipe
