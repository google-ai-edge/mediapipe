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

#include "absl/log/absl_log.h"
#include "mediapipe/calculators/core/sequence_shift_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"

namespace mediapipe {
namespace api2 {

// A Calculator that shifts the timestamps of packets along a stream. Packets on
// the input stream are output with a timestamp of the packet given by packet
// offset. That is, packet[i] is output with the timestamp of
// packet[i + packet_offset]. Packet offset can be either positive or negative.
// If packet_offset is -n, the first n packets will be dropped. If packet offset
// is n, the final n packets will be dropped. For example, with a packet_offset
// of -1, the first packet on the stream will be dropped, the second will be
// output with the timestamp of the first, the third with the timestamp of the
// second, and so on.
class SequenceShiftCalculator : public Node {
 public:
  static constexpr Input<AnyType> kIn{""};
  static constexpr SideInput<int>::Optional kOffset{"PACKET_OFFSET"};
  static constexpr Output<SameType<kIn>> kOut{""};

  MEDIAPIPE_NODE_CONTRACT(kIn, kOffset, kOut, TimestampChange::Arbitrary());

  // Reads from options to set cache_size_ and packet_offset_.
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  // A positive offset means we want a packet to be output with the timestamp of
  // a later packet. Stores packets waiting for their output timestamps and
  // outputs a single packet when the cache fills.
  void ProcessPositiveOffset(CalculatorContext* cc);

  // A negative offset means we want a packet to be output with the timestamp of
  // an earlier packet. Stores timestamps waiting for the corresponding input
  // packet and outputs a single packet when the cache fills.
  void ProcessNegativeOffset(CalculatorContext* cc);

  // Storage for packets waiting to be output when packet_offset > 0. When cache
  // is full, oldest packet is output with current timestamp.
  std::deque<PacketBase> packet_cache_;

  // Storage for previous timestamps used when packet_offset < 0. When cache is
  // full, oldest timestamp is used for current packet.
  std::deque<Timestamp> timestamp_cache_;

  // Copied from corresponding field in options.
  int packet_offset_;
  // The number of packets or timestamps we need to store to output packet[i] at
  // the timestamp of packet[i + packet_offset]; equal to abs(packet_offset).
  int cache_size_;
  bool emit_empty_packets_before_first_packet_ = false;
};
MEDIAPIPE_REGISTER_NODE(SequenceShiftCalculator);

absl::Status SequenceShiftCalculator::Open(CalculatorContext* cc) {
  packet_offset_ = kOffset(cc).GetOr(
      cc->Options<mediapipe::SequenceShiftCalculatorOptions>().packet_offset());
  emit_empty_packets_before_first_packet_ =
      cc->Options<mediapipe::SequenceShiftCalculatorOptions>()
          .emit_empty_packets_before_first_packet();
  cache_size_ = abs(packet_offset_);
  // An offset of zero is a no-op, but someone might still request it.
  if (packet_offset_ == 0) {
    cc->Outputs().Index(0).SetOffset(0);
  }
  return absl::OkStatus();
}

absl::Status SequenceShiftCalculator::Process(CalculatorContext* cc) {
  if (packet_offset_ > 0) {
    ProcessPositiveOffset(cc);
  } else if (packet_offset_ < 0) {
    ProcessNegativeOffset(cc);
  } else {
    kOut(cc).Send(kIn(cc).packet());
  }
  return absl::OkStatus();
}

void SequenceShiftCalculator::ProcessPositiveOffset(CalculatorContext* cc) {
  if (packet_cache_.size() >= cache_size_) {
    // Ready to output oldest packet with current timestamp.
    kOut(cc).Send(packet_cache_.front().At(cc->InputTimestamp()));
    packet_cache_.pop_front();
  } else if (emit_empty_packets_before_first_packet_) {
    ABSL_LOG(FATAL) << "Not supported yet";
  }
  // Store current packet for later output.
  packet_cache_.push_back(kIn(cc).packet());
}

void SequenceShiftCalculator::ProcessNegativeOffset(CalculatorContext* cc) {
  if (timestamp_cache_.size() >= cache_size_) {
    // Ready to output current packet with oldest timestamp.
    kOut(cc).Send(kIn(cc).packet().At(timestamp_cache_.front()));
    timestamp_cache_.pop_front();
  }
  // Store current timestamp for use by a future packet.
  timestamp_cache_.push_back(cc->InputTimestamp());
}

}  // namespace api2
}  // namespace mediapipe
