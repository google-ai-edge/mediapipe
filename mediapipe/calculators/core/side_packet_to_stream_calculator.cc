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
static std::map<std::string, Timestamp>* kTimestampMap = []() {
  auto* res = new std::map<std::string, Timestamp>();
  res->emplace("AT_PRESTREAM", Timestamp::PreStream());
  res->emplace("AT_POSTSTREAM", Timestamp::PostStream());
  res->emplace("AT_ZERO", Timestamp(0));
  return res;
}();

}  // namespace

// Outputs the single input_side_packet at the timestamp specified in the
// output_stream tag. Valid tags are AT_PRESTREAM, AT_POSTSTREAM and AT_ZERO.
class SidePacketToStreamCalculator : public CalculatorBase {
 public:
  SidePacketToStreamCalculator() = default;
  ~SidePacketToStreamCalculator() override = default;

  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;
};
REGISTER_CALCULATOR(SidePacketToStreamCalculator);

::mediapipe::Status SidePacketToStreamCalculator::GetContract(
    CalculatorContract* cc) {
  cc->InputSidePackets().Index(0).SetAny();

  std::set<std::string> tags = cc->Outputs().GetTags();
  RET_CHECK_EQ(tags.size(), 1);

  RET_CHECK_EQ(kTimestampMap->count(*tags.begin()), 1);
  cc->Outputs().Tag(*tags.begin()).SetAny();

  return ::mediapipe::OkStatus();
}

::mediapipe::Status SidePacketToStreamCalculator::Process(
    CalculatorContext* cc) {
  return mediapipe::tool::StatusStop();
}

::mediapipe::Status SidePacketToStreamCalculator::Close(CalculatorContext* cc) {
  std::set<std::string> tags = cc->Outputs().GetTags();
  RET_CHECK_EQ(tags.size(), 1);
  const std::string& tag = *tags.begin();
  RET_CHECK_EQ(kTimestampMap->count(tag), 1);
  cc->Outputs().Tag(tag).AddPacket(
      cc->InputSidePackets().Index(0).At(kTimestampMap->at(tag)));

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
