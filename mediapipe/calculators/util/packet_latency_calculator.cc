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

#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "mediapipe/calculators/util/latency.pb.h"
#include "mediapipe/calculators/util/packet_latency_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/deps/clock.h"
#include "mediapipe/framework/deps/monotonic_clock.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {

namespace {

// Tag name for clock side packet.
constexpr char kClockTag[] = "CLOCK";

// Tag name for reference signal.
constexpr char kReferenceSignalTag[] = "REFERENCE_SIGNAL";
}  // namespace

// A MediaPipe calculator that computes latency of incoming packet streams with
// respect to a reference signal (e.g image, audio frames).
//
// The latency of a packet wrt a reference packet is defined as the difference
// between arrival times of the two. A latency of X microseconds implies that
// the packet arrived X microseconds after its corresponding reference packet.
// For each packet stream, the calculator outputs the current latency, average,
// and a histogram of observed latencies so far.
//
// NOTE:
// 1) This calculator is meant to be used ONLY with an
// ImmediateInputStreamHandler.
// 2) This calculator is meant to be used only for real-time or simulated real-
// time applications. For example, the reference signal could be audio/video
// frames coming from a calculator that reads microphone/webcam data or some
// calculator that simulates real-time input.
// 3) If the packet labels are provided through options, then the number of
// labels should be exactly same as number of output_streams. If no packet
// label is defined in the node options, the calculator uses the input stream
// names.
//
// InputSidePacket (Optional):
// CLOCK: A clock for knowing current time.
//
// Inputs:
// 0- Packet stream 0 (e.g image feature 0):
// 1- Packet stream 1 (e.g image features 1):
// ...
// N- Packet stream N (e.g image features N):
// REFERENCE_SIGNAL: The reference signal from which the above packets were
//                   extracted (e.g image frames).
//
// Outputs:
// 0- Latency of packet stream 0.
// 1- Latency of packet stream 1.
// ...
// N- Latency of packet stream N.
//
// Example config:
// node {
//   calculator: "PacketLatencyCalculator"
//   input_side_packet: "monotonic_clock"
//   input_stream: "packet_stream_0"
//   input_stream: "packet_stream_1"
//   ...
//   input_stream: "packet_stream_N"
//   input_stream: "REFERENCE_SIGNAL:camera_frames"
//   output_stream: "packet_latency_0"
//   output_stream: "packet_latency_1"
//   ...
//   output_stream: "packet_latency_N"
//   options {
//     [soapbox.PacketLatencyCalculatorOptions.ext] {
//       num_intervals: 10
//       interval_size_usec: 10000
//     }
//   }
//   input_stream_handler {
//     input_stream_handler: 'ImmediateInputStreamHandler'
//   }
// }
class PacketLatencyCalculator : public CalculatorBase {
 public:
  PacketLatencyCalculator() {}

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  // Resets the histogram and running average variables by initializing them to
  // zero.
  void ResetStatistics();

  // Calculator options.
  PacketLatencyCalculatorOptions options_;

  // Clock object.
  std::shared_ptr<::mediapipe::Clock> clock_;

  // Clock time when the first reference packet was received.
  int64 first_process_time_usec_ = -1;

  // Timestamp of the first reference packet received.
  int64 first_reference_timestamp_usec_ = -1;

  // Number of packet streams.
  int64 num_packet_streams_ = -1;

  // Latency output for each packet stream.
  std::vector<PacketLatency> packet_latencies_;

  // Running sum and count of latencies for each packet stream. This is required
  // to compute the average latency.
  std::vector<int64> sum_latencies_usec_;
  std::vector<int64> num_latencies_;

  // Clock time when last reset was done for histogram and running average.
  int64 last_reset_time_usec_ = -1;
};
REGISTER_CALCULATOR(PacketLatencyCalculator);

::mediapipe::Status PacketLatencyCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK_GT(cc->Inputs().NumEntries(), 1);

  // Input and output streams.
  int64 num_packet_streams = cc->Inputs().NumEntries() - 1;
  RET_CHECK_EQ(cc->Outputs().NumEntries(), num_packet_streams);
  for (int64 i = 0; i < num_packet_streams; ++i) {
    cc->Inputs().Index(i).SetAny();
    cc->Outputs().Index(i).Set<PacketLatency>();
  }

  // Reference signal.
  cc->Inputs().Tag(kReferenceSignalTag).SetAny();

  // Clock side packet.
  if (cc->InputSidePackets().HasTag(kClockTag)) {
    cc->InputSidePackets()
        .Tag(kClockTag)
        .Set<std::shared_ptr<::mediapipe::Clock>>();
  }

  return ::mediapipe::OkStatus();
}

void PacketLatencyCalculator::ResetStatistics() {
  // Initialize histogram with zero counts and set running average to zero.
  for (int64 i = 0; i < num_packet_streams_; ++i) {
    for (int64 interval_index = 0; interval_index < options_.num_intervals();
         ++interval_index) {
      packet_latencies_[i].set_counts(interval_index, 0);
    }

    // Initialize the running sum and count to 0.
    sum_latencies_usec_[i] = 0;
    num_latencies_[i] = 0;
  }
}

::mediapipe::Status PacketLatencyCalculator::Open(CalculatorContext* cc) {
  options_ = cc->Options<PacketLatencyCalculatorOptions>();
  num_packet_streams_ = cc->Inputs().NumEntries() - 1;

  // Check if provided labels are of correct size.
  bool labels_provided = !options_.packet_labels().empty();
  if (labels_provided) {
    RET_CHECK_EQ(options_.packet_labels_size(), num_packet_streams_)
        << "Input packet stream count different from output stream count.";
  }

  // Check that histogram params are valid.
  RET_CHECK_GT(options_.num_intervals(), 0);
  RET_CHECK_GT(options_.interval_size_usec(), 0);

  // Initialize latency outputs for all streams.
  packet_latencies_.resize(num_packet_streams_);
  sum_latencies_usec_.resize(num_packet_streams_);
  num_latencies_.resize(num_packet_streams_);
  for (int64 i = 0; i < num_packet_streams_; ++i) {
    // Initialize latency histograms with zero counts.
    packet_latencies_[i].set_num_intervals(options_.num_intervals());
    packet_latencies_[i].set_interval_size_usec(options_.interval_size_usec());
    packet_latencies_[i].mutable_counts()->Resize(options_.num_intervals(), 0);

    // Set the label for the stream. The packet labels are taken from options
    // (if provided). If not, default labels are created using the input/output
    // stream names.
    if (labels_provided) {
      packet_latencies_[i].set_label(options_.packet_labels(i));
    } else {
      int64 input_stream_index = cc->Inputs().TagMap()->GetId("", i).value();
      packet_latencies_[i].set_label(
          cc->Inputs().TagMap()->Names()[input_stream_index]);
    }
  }

  // Initialize the clock.
  if (cc->InputSidePackets().HasTag(kClockTag)) {
    clock_ = cc->InputSidePackets()
                 .Tag("CLOCK")
                 .Get<std::shared_ptr<::mediapipe::Clock>>();
  } else {
    clock_ = std::shared_ptr<::mediapipe::Clock>(
        ::mediapipe::MonotonicClock::CreateSynchronizedMonotonicClock());
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status PacketLatencyCalculator::Process(CalculatorContext* cc) {
  // Record first process timestamp if this is the first call.
  if (first_process_time_usec_ < 0 &&
      !cc->Inputs().Tag(kReferenceSignalTag).IsEmpty()) {
    first_process_time_usec_ = absl::ToUnixMicros(clock_->TimeNow());
    first_reference_timestamp_usec_ = cc->InputTimestamp().Value();
    last_reset_time_usec_ = first_process_time_usec_;
  }

  if (first_process_time_usec_ < 0) {
    LOG(WARNING) << "No reference packet received.";
    return ::mediapipe::OkStatus();
  }

  if (options_.reset_duration_usec() > 0) {
    const int64 time_now_usec = absl::ToUnixMicros(clock_->TimeNow());
    if (time_now_usec - last_reset_time_usec_ >=
        options_.reset_duration_usec()) {
      ResetStatistics();
      last_reset_time_usec_ = time_now_usec;
    }
  }

  // Update latency info if there is any incoming packet.
  for (int64 i = 0; i < num_packet_streams_; ++i) {
    if (!cc->Inputs().Index(i).IsEmpty()) {
      const auto& packet_timestamp_usec = cc->InputTimestamp().Value();

      // Update latency statistics for this stream.
      int64 current_clock_time_usec = absl::ToUnixMicros(clock_->TimeNow());
      int64 current_calibrated_timestamp_usec =
          (current_clock_time_usec - first_process_time_usec_) +
          first_reference_timestamp_usec_;
      int64 packet_latency_usec =
          current_calibrated_timestamp_usec - packet_timestamp_usec;

      // Invalid timestamps in input signals could result in negative latencies.
      if (packet_latency_usec < 0) {
        continue;
      }

      // Update the latency, running average and histogram for this stream.
      packet_latencies_[i].set_current_latency_usec(packet_latency_usec);
      int64 interval_index =
          packet_latency_usec / packet_latencies_[i].interval_size_usec();
      if (interval_index >= packet_latencies_[i].num_intervals()) {
        interval_index = packet_latencies_[i].num_intervals() - 1;
      }
      packet_latencies_[i].set_counts(
          interval_index, packet_latencies_[i].counts(interval_index) + 1);
      sum_latencies_usec_[i] += packet_latency_usec;
      num_latencies_[i] += 1;
      packet_latencies_[i].set_avg_latency_usec(sum_latencies_usec_[i] /
                                                num_latencies_[i]);

      packet_latencies_[i].set_sum_latency_usec(sum_latencies_usec_[i]);

      // Push the latency packet to output.
      auto packet_latency =
          absl::make_unique<PacketLatency>(packet_latencies_[i]);
      cc->Outputs().Index(i).Add(packet_latency.release(),
                                 cc->InputTimestamp());
    }
  }

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
