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

#include <algorithm>
#include <utility>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/header_util.h"

namespace mediapipe {

constexpr char kAllowTag[] = "ALLOW";
constexpr char kMaxInFlightTag[] = "MAX_IN_FLIGHT";

// RealTimeFlowLimiterCalculator is used to limit the number of pipelined
// processing operations in a section of the graph.
//
// Typical topology:
//
// in ->-[FLC]-[foo]-...-[bar]-+->- out
//         ^_____________________|
//      FINISHED
//
// By connecting the output of the graph section to this calculator's FINISHED
// input with a backwards edge, this allows FLC to keep track of how many
// timestamps are currently being processed.
//
// The limit defaults to 1, and can be overridden with the MAX_IN_FLIGHT side
// packet.
//
// As long as the number of timestamps being processed ("in flight") is below
// the limit, FLC allows input to pass through. When the limit is reached,
// FLC starts dropping input packets, keeping only the most recent. When the
// processing count decreases again, as signaled by the receipt of a packet on
// FINISHED, FLC allows packets to flow again, releasing the most recently
// queued packet, if any.
//
// If there are multiple input streams, packet dropping is synchronized.
//
// IMPORTANT: for each timestamp where FLC forwards a packet (or a set of
// packets, if using multiple data streams), a packet must eventually arrive on
// the FINISHED stream. Dropping packets in the section between FLC and
// FINISHED will make the in-flight count incorrect.
//
// TODO: Remove this comment when graph-level ISH has been removed.
// NOTE: this calculator should always use the ImmediateInputStreamHandler and
// uses it by default. However, if the graph specifies a graph-level
// InputStreamHandler, to override that setting, the InputStreamHandler must
// be explicitly specified as shown below.
//
// Example config:
// node {
//   calculator: "RealTimeFlowLimiterCalculator"
//   input_stream: "raw_frames"
//   input_stream: "FINISHED:finished"
//   input_stream_info: {
//     tag_index: 'FINISHED'
//     back_edge: true
//   }
//   input_stream_handler {
//     input_stream_handler: 'ImmediateInputStreamHandler'
//   }
//   output_stream: "gated_frames"
// }
class RealTimeFlowLimiterCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    int num_data_streams = cc->Inputs().NumEntries("");
    RET_CHECK_GE(num_data_streams, 1);
    RET_CHECK_EQ(cc->Outputs().NumEntries(""), num_data_streams)
        << "Output streams must correspond input streams except for the "
           "finish indicator input stream.";
    for (int i = 0; i < num_data_streams; ++i) {
      cc->Inputs().Get("", i).SetAny();
      cc->Outputs().Get("", i).SetSameAs(&(cc->Inputs().Get("", i)));
    }
    cc->Inputs().Get("FINISHED", 0).SetAny();
    if (cc->InputSidePackets().HasTag(kMaxInFlightTag)) {
      cc->InputSidePackets().Tag(kMaxInFlightTag).Set<int>();
    }
    if (cc->Outputs().HasTag(kAllowTag)) {
      cc->Outputs().Tag(kAllowTag).Set<bool>();
    }

    cc->SetInputStreamHandler("ImmediateInputStreamHandler");

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    finished_id_ = cc->Inputs().GetId("FINISHED", 0);
    max_in_flight_ = 1;
    if (cc->InputSidePackets().HasTag(kMaxInFlightTag)) {
      max_in_flight_ = cc->InputSidePackets().Tag(kMaxInFlightTag).Get<int>();
    }
    RET_CHECK_GE(max_in_flight_, 1);
    num_in_flight_ = 0;

    allowed_id_ = cc->Outputs().GetId("ALLOW", 0);
    allow_ctr_ts_ = Timestamp(0);

    num_data_streams_ = cc->Inputs().NumEntries("");
    data_stream_bound_ts_.resize(num_data_streams_);
    RET_CHECK_OK(CopyInputHeadersToOutputs(cc->Inputs(), &(cc->Outputs())));
    return absl::OkStatus();
  }

  bool Allow() { return num_in_flight_ < max_in_flight_; }

  absl::Status Process(CalculatorContext* cc) final {
    bool old_allow = Allow();
    Timestamp lowest_incomplete_ts = Timestamp::Done();

    // Process FINISHED stream.
    if (!cc->Inputs().Get(finished_id_).Value().IsEmpty()) {
      RET_CHECK_GT(num_in_flight_, 0)
          << "Received a FINISHED packet, but we had none in flight.";
      --num_in_flight_;
    }

    // Process data streams.
    for (int i = 0; i < num_data_streams_; ++i) {
      auto& stream = cc->Inputs().Get("", i);
      auto& out = cc->Outputs().Get("", i);
      Packet& packet = stream.Value();
      auto ts = packet.Timestamp();
      if (ts.IsRangeValue() && data_stream_bound_ts_[i] <= ts) {
        data_stream_bound_ts_[i] = ts + 1;
        // Note: it's ok to update the output bound here, before sending the
        // packet, because updates are batched during the Process function.
        out.SetNextTimestampBound(data_stream_bound_ts_[i]);
      }
      lowest_incomplete_ts =
          std::min(lowest_incomplete_ts, data_stream_bound_ts_[i]);

      if (packet.IsEmpty()) {
        // If the input stream is closed, close the corresponding output.
        if (stream.IsDone() && !out.IsClosed()) {
          out.Close();
        }
        // TODO: if the packet is empty, the ts is unset, and we
        // cannot read the timestamp bound, even though we'd like to propagate
        // it.
      } else if (mediapipe::ContainsKey(pending_ts_, ts)) {
        // If we have already sent this timestamp (on another stream), send it
        // on this stream too.
        out.AddPacket(std::move(packet));
      } else if (Allow() && (ts > last_dropped_ts_)) {
        // If the in-flight is under the limit, and if we have not already
        // dropped this or a later timestamp on another stream, then send
        // the packet and add an in-flight timestamp.
        out.AddPacket(std::move(packet));
        pending_ts_.insert(ts);
        ++num_in_flight_;
      } else {
        // Otherwise, we'll drop the packet.
        last_dropped_ts_ = std::max(last_dropped_ts_, ts);
      }
    }

    // Remove old pending_ts_ entries.
    auto it = std::lower_bound(pending_ts_.begin(), pending_ts_.end(),
                               lowest_incomplete_ts);
    pending_ts_.erase(pending_ts_.begin(), it);

    // Update ALLOW signal.
    if ((old_allow != Allow()) && allowed_id_.IsValid()) {
      cc->Outputs()
          .Get(allowed_id_)
          .AddPacket(MakePacket<bool>(Allow()).At(++allow_ctr_ts_));
    }
    return absl::OkStatus();
  }

 private:
  std::set<Timestamp> pending_ts_;
  Timestamp last_dropped_ts_;
  int num_data_streams_;
  int num_in_flight_;
  int max_in_flight_;
  CollectionItemId finished_id_;
  CollectionItemId allowed_id_;
  Timestamp allow_ctr_ts_;
  std::vector<Timestamp> data_stream_bound_ts_;
};
REGISTER_CALCULATOR(RealTimeFlowLimiterCalculator);

}  // namespace mediapipe
