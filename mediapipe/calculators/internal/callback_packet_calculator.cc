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
#include "mediapipe/calculators/internal/callback_packet_calculator.h"

#include <functional>
#include <string>

#include "absl/status/status.h"
#include "mediapipe/calculators/internal/callback_packet_calculator.pb.h"  // NOLINT
#include "mediapipe/framework/calculator_base.h"
#include "mediapipe/framework/calculator_registry.h"
#include "mediapipe/framework/output_side_packet.h"

namespace mediapipe {

namespace {

// Callback function for writing a packet to a vector.  The output is before the
// input since std::bind fills arguments from left to right (and only
// dumped_data is filled by std::bind).
void DumpToVector(std::vector<Packet>* dumped_data, const Packet& packet) {
  dumped_data->push_back(packet);
}

// Callback function for saving the Timestamp::PostStream() packet.
// The output is before the input since std::bind fills arguments from left to
// right (and only post_stream_packet is filled by std::bind).
void DumpPostStreamPacket(Packet* post_stream_packet, const Packet& packet) {
  if (packet.Timestamp() == Timestamp::PostStream()) {
    *post_stream_packet = packet;
  }
}

}  // namespace

absl::Status CallbackPacketCalculator::GetContract(CalculatorContract* cc) {
  const auto& options = cc->Options<CallbackPacketCalculatorOptions>();
  switch (options.type()) {
    case CallbackPacketCalculatorOptions::VECTOR_PACKET:
    case CallbackPacketCalculatorOptions::POST_STREAM_PACKET:
      cc->OutputSidePackets()
          .Index(0)
          .Set<std::function<void(const Packet&)>>();
      break;
    default:
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Invalid type of callback to produce.";
  }
  return absl::OkStatus();
}

absl::Status CallbackPacketCalculator::Open(CalculatorContext* cc) {
  const auto& options = cc->Options<CallbackPacketCalculatorOptions>();
  void* ptr;
  if (sscanf(options.pointer().c_str(), "%p", &ptr) != 1) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Stored pointer value in options is invalid.";
  }
  switch (options.type()) {
    case CallbackPacketCalculatorOptions::VECTOR_PACKET:
      cc->OutputSidePackets().Index(0).Set(
          MakePacket<std::function<void(const Packet&)>>(std::bind(
              &DumpToVector, reinterpret_cast<std::vector<Packet>*>(ptr),
              std::placeholders::_1)));
      break;
    case CallbackPacketCalculatorOptions::POST_STREAM_PACKET:
      cc->OutputSidePackets().Index(0).Set(
          MakePacket<std::function<void(const Packet&)>>(
              std::bind(&DumpPostStreamPacket, reinterpret_cast<Packet*>(ptr),
                        std::placeholders::_1)));
      break;
    default:
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Invalid type to dump into.";
  }
  return absl::OkStatus();
}

absl::Status CallbackPacketCalculator::Process(CalculatorContext* cc) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(CallbackPacketCalculator);

}  // namespace mediapipe
