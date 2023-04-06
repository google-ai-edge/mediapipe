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

#include "absl/time/time.h"
#include "mediapipe/calculators/util/packet_frequency.pb.h"
#include "mediapipe/calculators/util/packet_frequency_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/timestamp.h"

namespace {
constexpr int kSecondsToMicroseconds = 1000000;

}  // namespace

namespace mediapipe {
// A MediaPipe calculator that computes the frequency (in Hertz) of incoming
// packet streams. The frequency of packets is computed over a time window
// that is configured in options. There must be one output stream corresponding
// to every input packet stream. The frequency is output as a PacketFrequency
// proto.
//
// NOTE:
// 1. For computing frequency, packet timestamps are used and not the wall
// timestamp. Hence, the calculator is best-suited for real-time applications.
// 2. When multiple input/output streams are present, the calculator must be
// used with an ImmediateInputStreamHandler.
//
// Example config:
// node {
//   calculator: "PacketFrequencyCalculator"
//   input_stream: "input_stream_0"
//   input_stream: "input_stream_1"
//   .
//   .
//   input_stream: "input_stream_N"
//   output_stream: "packet_frequency_0"
//   output_stream: "packet_frequency_1"
//   .
//   .
//   output_stream: "packet_frequency_N"
//   input_stream_handler {
//     input_stream_handler: "ImmediateInputStreamHandler"
//   }
//   options {
//     [soapbox.PacketFrequencyCalculatorOptions.ext] {
//       time_window_sec: 3.0
//       label: "stream_name_0"
//       label: "stream_name_1"
//       .
//       .
//       label: "stream_name_N"
//     }
//   }
// }
class PacketFrequencyCalculator : public CalculatorBase {
 public:
  PacketFrequencyCalculator() {}

  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  // Outputs the given framerate on the specified output stream as a
  // PacketFrequency proto.
  absl::Status OutputPacketFrequency(CalculatorContext* cc, int stream_id,
                                     double framerate_hz,
                                     const std::string& label,
                                     const Timestamp& input_timestamp);

  // Adds the input timestamp in the particular stream's timestamp buffer.
  absl::Status AddPacketTimestampForStream(int stream_id, int64_t timestamp);

  // For the specified input stream, clears timestamps from buffer that are
  // older than the configured time_window_sec.
  absl::Status ClearOldpacketTimestamps(int stream_id,
                                        int64_t current_timestamp);

  // Options for the calculator.
  PacketFrequencyCalculatorOptions options_;

  // Map where key is the input stream ID and value is the timestamp of the
  // first packet received on that stream.
  std::map<int, int64_t> first_timestamp_for_stream_id_usec_;

  // Map where key is the input stream ID and value is a vector that stores
  // timestamps of recently received packets on the stream. Timestamps older
  // than the time_window_sec are continuously deleted for all the streams.
  std::map<int, std::vector<int64_t>> previous_timestamps_for_stream_id_;
};
REGISTER_CALCULATOR(PacketFrequencyCalculator);

absl::Status PacketFrequencyCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK_EQ(cc->Outputs().NumEntries(), cc->Inputs().NumEntries());
  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    cc->Inputs().Index(i).SetAny();
    cc->Outputs().Index(i).Set<PacketFrequency>();
  }
  return absl::OkStatus();
}

absl::Status PacketFrequencyCalculator::Open(CalculatorContext* cc) {
  options_ = cc->Options<PacketFrequencyCalculatorOptions>();
  RET_CHECK_EQ(options_.label_size(), cc->Inputs().NumEntries());
  RET_CHECK_GT(options_.time_window_sec(), 0);
  RET_CHECK_LE(options_.time_window_sec(), 100);

  // Initialize the stream-related data structures.
  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    RET_CHECK(!options_.label(i).empty());
    previous_timestamps_for_stream_id_[i] = {};
    first_timestamp_for_stream_id_usec_[i] = -1;
  }
  return absl::OkStatus();
}

absl::Status PacketFrequencyCalculator::Process(CalculatorContext* cc) {
  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    if (cc->Inputs().Index(i).IsEmpty()) {
      continue;
    }
    RET_CHECK_OK(AddPacketTimestampForStream(/*stream_id=*/i,
                                             cc->InputTimestamp().Value()));
    RET_CHECK_OK(ClearOldpacketTimestamps(/*stream_id=*/i,
                                          cc->InputTimestamp().Value()));

    if (first_timestamp_for_stream_id_usec_[i] < 0) {
      first_timestamp_for_stream_id_usec_[i] = cc->InputTimestamp().Value();

      // Since this is the very first packet on this stream, we don't have a
      // window of time over which we can compute the packet frequency. So
      // outputting packet frequency for this stream as 0 Hz.
      return OutputPacketFrequency(cc, /*stream_id=*/i, /*framerate_hz=*/0.0,
                                   options_.label(i), cc->InputTimestamp());
    }

    // If the time elapsed is less that the configured time window, then use
    // that time duration instead, else use the configured time window.
    double time_window_usec =
        std::min(static_cast<double>(cc->InputTimestamp().Value() -
                                     first_timestamp_for_stream_id_usec_[i]),
                 options_.time_window_sec() * kSecondsToMicroseconds);

    double framerate_hz = (previous_timestamps_for_stream_id_[i].size() * 1.0) /
                          (time_window_usec / kSecondsToMicroseconds);

    return OutputPacketFrequency(cc, /*stream_id=*/i, framerate_hz,
                                 options_.label(i), cc->InputTimestamp());
  }

  return absl::OkStatus();
}

absl::Status PacketFrequencyCalculator::AddPacketTimestampForStream(
    int stream_id, int64_t timestamp_usec) {
  if (previous_timestamps_for_stream_id_.find(stream_id) ==
      previous_timestamps_for_stream_id_.end()) {
    return absl::InvalidArgumentError("Input stream id is invalid");
  }

  previous_timestamps_for_stream_id_[stream_id].push_back(timestamp_usec);

  return absl::OkStatus();
}

absl::Status PacketFrequencyCalculator::ClearOldpacketTimestamps(
    int stream_id, int64_t current_timestamp_usec) {
  if (previous_timestamps_for_stream_id_.find(stream_id) ==
      previous_timestamps_for_stream_id_.end()) {
    return absl::InvalidArgumentError("Input stream id is invalid");
  }

  auto& timestamps_buffer = previous_timestamps_for_stream_id_[stream_id];
  int64_t time_window_usec =
      options_.time_window_sec() * kSecondsToMicroseconds;

  timestamps_buffer.erase(
      std::remove_if(timestamps_buffer.begin(), timestamps_buffer.end(),
                     [&time_window_usec,
                      &current_timestamp_usec](const int64_t timestamp_usec) {
                       return current_timestamp_usec - timestamp_usec >
                              time_window_usec;
                     }),
      timestamps_buffer.end());

  return absl::OkStatus();
}

absl::Status PacketFrequencyCalculator::OutputPacketFrequency(
    CalculatorContext* cc, int stream_id, double framerate_hz,
    const std::string& label, const Timestamp& input_timestamp) {
  auto packet_frequency = absl::make_unique<PacketFrequency>();
  packet_frequency->set_packet_frequency_hz(framerate_hz);
  packet_frequency->set_label(label);

  cc->Outputs().Index(stream_id).Add(packet_frequency.release(),
                                     input_timestamp);

  return absl::OkStatus();
}

}  // namespace mediapipe
