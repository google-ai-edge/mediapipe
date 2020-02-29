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

#include <map>
#include <memory>
#include <set>
#include <string>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

using mediapipe::PacketTypeSet;
using mediapipe::Timestamp;

namespace {

constexpr char kTagAtPreStream[] = "AT_PRESTREAM";
constexpr char kTagAtPostStream[] = "AT_POSTSTREAM";
constexpr char kTagAtZero[] = "AT_ZERO";
constexpr char kTagAtTick[] = "AT_TICK";
constexpr char kTagTick[] = "TICK";

static std::map<std::string, Timestamp>* kTimestampMap = []() {
  auto* res = new std::map<std::string, Timestamp>();
  res->emplace(kTagAtPreStream, Timestamp::PreStream());
  res->emplace(kTagAtPostStream, Timestamp::PostStream());
  res->emplace(kTagAtZero, Timestamp(0));
  res->emplace(kTagAtTick, Timestamp::Unset());
  return res;
}();

template <typename CC>
std::string GetOutputTag(const CC& cc) {
  // Single output tag only is required by contract.
  return *cc.Outputs().GetTags().begin();
}

}  // namespace

// Outputs side packet(s) in corresponding output stream(s) with a particular
// timestamp, depending on the tag used to define output stream(s). (One tag can
// be used only.)
//
// Valid tags are AT_PRESTREAM, AT_POSTSTREAM, AT_ZERO and AT_TICK and
// corresponding timestamps are Timestamp::PreStream(), Timestamp::PostStream(),
// Timestamp(0) and timestamp of a packet received in TICK input.
//
// Examples:
// node {
//   calculator: "SidePacketToStreamCalculator"
//   input_side_packet: "side_packet"
//   output_stream: "AT_PRESTREAM:packet"
// }
//
// node {
//   calculator: "SidePacketToStreamCalculator"
//   input_stream: "TICK:tick"
//   input_side_packet: "side_packet"
//   output_stream: "AT_TICK:packet"
// }
class SidePacketToStreamCalculator : public CalculatorBase {
 public:
  SidePacketToStreamCalculator() = default;
  ~SidePacketToStreamCalculator() override = default;

  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  bool is_tick_processing_ = false;
  std::string output_tag_;
};
REGISTER_CALCULATOR(SidePacketToStreamCalculator);

::mediapipe::Status SidePacketToStreamCalculator::GetContract(
    CalculatorContract* cc) {
  const auto& tags = cc->Outputs().GetTags();
  RET_CHECK(tags.size() == 1 && kTimestampMap->count(*tags.begin()) == 1)
      << "Only one of AT_PRESTREAM, AT_POSTSTREAM, AT_ZERO and AT_TICK tags is "
         "allowed and required to specify output stream(s).";
  RET_CHECK(
      (cc->Outputs().HasTag(kTagAtTick) && cc->Inputs().HasTag(kTagTick)) ||
      (!cc->Outputs().HasTag(kTagAtTick) && !cc->Inputs().HasTag(kTagTick)))
      << "Either both of TICK and AT_TICK should be used or none of them.";
  const std::string output_tag = GetOutputTag(*cc);
  const int num_entries = cc->Outputs().NumEntries(output_tag);
  RET_CHECK_EQ(num_entries, cc->InputSidePackets().NumEntries())
      << "Same number of input side packets and output streams is required.";
  for (int i = 0; i < num_entries; ++i) {
    cc->InputSidePackets().Index(i).SetAny();
    cc->Outputs()
        .Get(output_tag, i)
        .SetSameAs(cc->InputSidePackets().Index(i).GetSameAs());
  }

  if (cc->Inputs().HasTag(kTagTick)) {
    cc->Inputs().Tag(kTagTick).SetAny();
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status SidePacketToStreamCalculator::Open(CalculatorContext* cc) {
  output_tag_ = GetOutputTag(*cc);
  if (cc->Inputs().HasTag(kTagTick)) {
    is_tick_processing_ = true;
    // Set offset, so output timestamp bounds are updated in response to TICK
    // timestamp bound update.
    cc->SetOffset(TimestampDiff(0));
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status SidePacketToStreamCalculator::Process(
    CalculatorContext* cc) {
  if (is_tick_processing_) {
    // TICK input is guaranteed to be non-empty, as it's the only input stream
    // for this calculator.
    const auto& timestamp = cc->Inputs().Tag(kTagTick).Value().Timestamp();
    for (int i = 0; i < cc->Outputs().NumEntries(output_tag_); ++i) {
      cc->Outputs()
          .Get(output_tag_, i)
          .AddPacket(cc->InputSidePackets().Index(i).At(timestamp));
    }

    return ::mediapipe::OkStatus();
  }

  return ::mediapipe::tool::StatusStop();
}

::mediapipe::Status SidePacketToStreamCalculator::Close(CalculatorContext* cc) {
  if (!cc->Outputs().HasTag(kTagAtTick)) {
    const auto& timestamp = kTimestampMap->at(output_tag_);
    for (int i = 0; i < cc->Outputs().NumEntries(output_tag_); ++i) {
      cc->Outputs()
          .Get(output_tag_, i)
          .AddPacket(cc->InputSidePackets().Index(i).At(timestamp));
    }
  }
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
