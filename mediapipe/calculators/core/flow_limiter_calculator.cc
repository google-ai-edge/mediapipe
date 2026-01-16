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

#include "mediapipe/calculators/core/flow_limiter_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/header_util.h"

namespace mediapipe {

constexpr char kFinishedTag[] = "FINISHED";
constexpr char kAllowTag[] = "ALLOW";
constexpr char kMaxInFlightTag[] = "MAX_IN_FLIGHT";
constexpr char kOptionsTag[] = "OPTIONS";

// FlowLimiterCalculator is used to limit the number of frames in flight
// by dropping input frames when necessary.
//
// The input stream "FINISHED" is used to signal the FlowLimiterCalculator
// when a frame is finished processing.  Either a non-empty "FINISHED" packet
// or a timestamp bound should be received for each processed frame.
//
// The combination of `max_in_flight: 1` and `max_in_queue: 1` generally gives
// best throughput/latency balance.  Throughput is nearly optimal as the
// graph is never idle as there is always something in the queue.  Latency is
// nearly optimal latency as the queue always stores the latest available frame.
//
// Increasing `max_in_flight` to 2 or more can yield the better throughput
// when the graph exhibits a high degree of pipeline parallelism.  Decreasing
// `max_in_queue` to 0 can yield a better average latency, but at the cost of
// lower throughput (lower framerate) due to the time during which the graph
// is idle awaiting the next input frame.
//
// Example config:
// node {
//   calculator: "FlowLimiterCalculator"
//   input_stream: "raw_frames"
//   input_stream: "FINISHED:finished"
//   input_stream_info: {
//     tag_index: 'FINISHED'
//     back_edge: true
//   }
//   output_stream: "sampled_frames"
//   output_stream: "ALLOW:allowed_timestamps"
// }
//
// The "ALLOW" stream indicates the transition between accepting frames and
// dropping frames.  "ALLOW = true" indicates the start of accepting frames
// including the current timestamp, and "ALLOW = false" indicates the start of
// dropping frames including the current timestamp.
//
// FlowLimiterCalculator provides limited support for multiple input streams.
// The first input stream is treated as the main input stream and successive
// input streams are treated as auxiliary input streams.  The auxiliary input
// streams are limited to timestamps allowed by the "ALLOW" stream.
//
class FlowLimiterCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    auto& side_inputs = cc->InputSidePackets();
    side_inputs.Tag(kOptionsTag).Set<FlowLimiterCalculatorOptions>().Optional();
    cc->Inputs()
        .Tag(kOptionsTag)
        .Set<FlowLimiterCalculatorOptions>()
        .Optional();
    RET_CHECK_GE(cc->Inputs().NumEntries(""), 1);
    for (int i = 0; i < cc->Inputs().NumEntries(""); ++i) {
      cc->Inputs().Get("", i).SetAny();
      cc->Outputs().Get("", i).SetSameAs(&(cc->Inputs().Get("", i)));
    }
    cc->Inputs().Get("FINISHED", 0).SetAny();
    cc->InputSidePackets().Tag(kMaxInFlightTag).Set<int>().Optional();
    cc->Outputs().Tag(kAllowTag).Set<bool>().Optional();
    cc->SetInputStreamHandler("ImmediateInputStreamHandler");
    cc->SetProcessTimestampBounds(true);
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    options_ = cc->Options<FlowLimiterCalculatorOptions>();
    options_ = tool::RetrieveOptions(options_, cc->InputSidePackets());
    if (cc->InputSidePackets().HasTag(kMaxInFlightTag)) {
      options_.set_max_in_flight(
          cc->InputSidePackets().Tag(kMaxInFlightTag).Get<int>());
    }
    input_queues_.resize(cc->Inputs().NumEntries(""));
    allowed_[Timestamp::Unset()] = true;
    RET_CHECK_OK(CopyInputHeadersToOutputs(cc->Inputs(), &(cc->Outputs())));
    return absl::OkStatus();
  }

  // Releases input packets allowed by the max_in_flight constraint.
  absl::Status Process(CalculatorContext* cc) final {
    options_ = tool::RetrieveOptions(options_, cc->Inputs());

    // Process the FINISHED input stream.
    Packet finished_packet = cc->Inputs().Tag(kFinishedTag).Value();
    if (finished_packet.Timestamp() == cc->InputTimestamp()) {
      while (!frames_in_flight_.empty() &&
             frames_in_flight_.front() <= finished_packet.Timestamp()) {
        frames_in_flight_.pop_front();
      }
    }

    // Process the frame input streams.
    for (int i = 0; i < cc->Inputs().NumEntries(""); ++i) {
      Packet packet = cc->Inputs().Get("", i).Value();
      if (!packet.IsEmpty()) {
        input_queues_[i].push_back(packet);
      }
    }

    // Abandon expired frames in flight.  Note that old frames are abandoned
    // when much newer frame timestamps arrive regardless of elapsed time.
    TimestampDiff timeout = options_.in_flight_timeout();
    Timestamp latest_ts = cc->Inputs().Get("", 0).Value().Timestamp();
    if (timeout > 0 && latest_ts == cc->InputTimestamp() &&
        latest_ts < Timestamp::Max()) {
      while (!frames_in_flight_.empty() &&
             (latest_ts - frames_in_flight_.front()) > timeout) {
        frames_in_flight_.pop_front();
      }
    }

    // Release allowed frames from the main input queue.
    auto& input_queue = input_queues_[0];
    while (ProcessingAllowed() && !input_queue.empty()) {
      Packet packet = input_queue.front();
      input_queue.pop_front();
      cc->Outputs().Get("", 0).AddPacket(packet);
      SendAllow(true, packet.Timestamp(), cc);
      frames_in_flight_.push_back(packet.Timestamp());
    }

    // Limit the number of queued frames.
    // Note that frames can be dropped after frames are released because
    // frame-packets and FINISH-packets never arrive in the same Process call.
    while (input_queue.size() > options_.max_in_queue()) {
      Packet packet = input_queue.front();
      input_queue.pop_front();
      SendAllow(false, packet.Timestamp(), cc);
    }

    // Propagate the input timestamp bound.
    if (!input_queue.empty()) {
      Timestamp bound = input_queue.front().Timestamp();
      SetNextTimestampBound(bound, &cc->Outputs().Get("", 0));
    } else {
      Timestamp bound =
          cc->Inputs().Get("", 0).Value().Timestamp().NextAllowedInStream();
      SetNextTimestampBound(bound, &cc->Outputs().Get("", 0));
      if (cc->Outputs().HasTag(kAllowTag)) {
        SetNextTimestampBound(bound, &cc->Outputs().Tag(kAllowTag));
      }
    }

    ProcessAuxiliaryInputs(cc);

    // Discard old ALLOW ranges.
    Timestamp input_bound = InputTimestampBound(cc);
    auto first_range = std::prev(allowed_.upper_bound(input_bound));
    allowed_.erase(allowed_.begin(), first_range);
    return absl::OkStatus();
  }

  int LedgerSize() {
    int result = frames_in_flight_.size() + allowed_.size();
    for (const auto& queue : input_queues_) {
      result += queue.size();
    }
    return result;
  }

 private:
  // Returns true if an additional frame can be released for processing.
  // The "ALLOW" output stream indicates this condition at each input frame.
  bool ProcessingAllowed() {
    return frames_in_flight_.size() < options_.max_in_flight();
  }

  // Outputs a packet indicating whether a frame was sent or dropped.
  void SendAllow(bool allow, Timestamp ts, CalculatorContext* cc) {
    if (cc->Outputs().HasTag(kAllowTag)) {
      cc->Outputs().Tag(kAllowTag).AddPacket(MakePacket<bool>(allow).At(ts));
    }
    allowed_[ts] = allow;
  }

  // Returns true if a timestamp falls within a range of allowed timestamps.
  bool IsAllowed(Timestamp timestamp) {
    auto it = allowed_.upper_bound(timestamp);
    return std::prev(it)->second;
  }

  // Sets the timestamp bound or closes an output stream.
  void SetNextTimestampBound(Timestamp bound, OutputStream* stream) {
    if (bound > Timestamp::Max()) {
      stream->Close();
    } else {
      stream->SetNextTimestampBound(bound);
    }
  }

  // Returns the lowest unprocessed input Timestamp.
  Timestamp InputTimestampBound(CalculatorContext* cc) {
    Timestamp result = Timestamp::Done();
    for (int i = 0; i < input_queues_.size(); ++i) {
      auto& queue = input_queues_[i];
      auto& stream = cc->Inputs().Get("", i);
      Timestamp bound = queue.empty()
                            ? stream.Value().Timestamp().NextAllowedInStream()
                            : queue.front().Timestamp();
      result = std::min(result, bound);
    }
    return result;
  }

  // Releases input packets up to the latest settled input timestamp.
  void ProcessAuxiliaryInputs(CalculatorContext* cc) {
    Timestamp settled_bound = cc->Outputs().Get("", 0).NextTimestampBound();
    for (int i = 1; i < cc->Inputs().NumEntries(""); ++i) {
      // Release settled frames from each input queue.
      while (!input_queues_[i].empty() &&
             input_queues_[i].front().Timestamp() < settled_bound) {
        Packet packet = input_queues_[i].front();
        input_queues_[i].pop_front();
        if (IsAllowed(packet.Timestamp())) {
          cc->Outputs().Get("", i).AddPacket(packet);
        }
      }

      // Propagate each input timestamp bound.
      if (!input_queues_[i].empty()) {
        Timestamp bound = input_queues_[i].front().Timestamp();
        SetNextTimestampBound(bound, &cc->Outputs().Get("", i));
      } else {
        Timestamp bound =
            cc->Inputs().Get("", i).Value().Timestamp().NextAllowedInStream();
        SetNextTimestampBound(bound, &cc->Outputs().Get("", i));
      }
    }
  }

 private:
  FlowLimiterCalculatorOptions options_;
  std::vector<std::deque<Packet>> input_queues_;
  std::deque<Timestamp> frames_in_flight_;
  std::map<Timestamp, bool> allowed_;
};
REGISTER_CALCULATOR(FlowLimiterCalculator);

}  // namespace mediapipe
