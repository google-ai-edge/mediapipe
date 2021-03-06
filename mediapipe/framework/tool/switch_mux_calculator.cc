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
#include <memory>
#include <set>
#include <string>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/input_stream_shard.h"
#include "mediapipe/framework/output_stream_shard.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/tool/container_util.h"

namespace mediapipe {

// A calculator to join several sets of input streams into one
// output channel, consisting of corresponding output streams.
// Each channel is distinguished by a tag-prefix such as "C1__".
// For example:
//
//         node {
//           calculator: "SwitchMuxCalculator"
//           input_stream: "ENABLE:enable"
//           input_stream: "C0__FUNC_INPUT:foo_0"
//           input_stream: "C0__FUNC_INPUT:bar_0"
//           input_stream: "C1__FUNC_INPUT:foo_1"
//           input_stream: "C1__FUNC_INPUT:bar_1"
//           output_stream: "FUNC_INPUT:foo"
//           output_stream: "FUNC_INPUT:bar"
//         }
//
// Input stream "ENABLE" specifies routing of packets from either channel 0
// or channel 1, given "ENABLE:false" or "ENABLE:true" respectively.
// Input-side-packet "ENABLE" and input-stream "SELECT" can also be used
// similarly to specify the active channel.
//
// SwitchMuxCalculator is used by SwitchContainer to enable one of several
// contained subgraph or calculator nodes.
//
class SwitchMuxCalculator : public CalculatorBase {
  static constexpr char kSelectTag[] = "SELECT";
  static constexpr char kEnableTag[] = "ENABLE";

 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  int channel_index_;
  std::set<std::string> channel_tags_;
};
REGISTER_CALCULATOR(SwitchMuxCalculator);

absl::Status SwitchMuxCalculator::GetContract(CalculatorContract* cc) {
  // Allow any one of kSelectTag, kEnableTag.
  if (cc->Inputs().HasTag(kSelectTag)) {
    cc->Inputs().Tag(kSelectTag).Set<int>();
  } else if (cc->Inputs().HasTag(kEnableTag)) {
    cc->Inputs().Tag(kEnableTag).Set<bool>();
  }
  // Allow any one of kSelectTag, kEnableTag.
  if (cc->InputSidePackets().HasTag(kSelectTag)) {
    cc->InputSidePackets().Tag(kSelectTag).Set<int>();
  } else if (cc->InputSidePackets().HasTag(kEnableTag)) {
    cc->InputSidePackets().Tag(kEnableTag).Set<bool>();
  }

  // Set the types for all input channels to corresponding output types.
  std::set<std::string> channel_tags = ChannelTags(cc->Inputs().TagMap());
  int channel_count = ChannelCount(cc->Inputs().TagMap());
  for (const std::string& tag : channel_tags) {
    for (int index = 0; index < cc->Outputs().NumEntries(tag); ++index) {
      cc->Outputs().Get(tag, index).SetAny();
      auto output_id = cc->Outputs().GetId(tag, index);
      if (output_id.IsValid()) {
        for (int channel = 0; channel < channel_count; ++channel) {
          auto input_id =
              cc->Inputs().GetId(tool::ChannelTag(tag, channel), index);
          if (input_id.IsValid()) {
            cc->Inputs().Get(input_id).SetSameAs(&cc->Outputs().Get(output_id));
          }
        }
      }
    }
  }
  channel_tags = ChannelTags(cc->InputSidePackets().TagMap());
  channel_count = ChannelCount(cc->InputSidePackets().TagMap());
  for (const std::string& tag : channel_tags) {
    int num_entries = cc->OutputSidePackets().NumEntries(tag);
    for (int index = 0; index < num_entries; ++index) {
      cc->OutputSidePackets().Get(tag, index).SetAny();
      auto output_id = cc->OutputSidePackets().GetId(tag, index);
      if (output_id.IsValid()) {
        for (int channel = 0; channel < channel_count; ++channel) {
          auto input_id = cc->InputSidePackets().GetId(
              tool::ChannelTag(tag, channel), index);
          if (input_id.IsValid()) {
            cc->InputSidePackets().Get(input_id).SetSameAs(
                &cc->OutputSidePackets().Get(output_id));
          }
        }
      }
    }
  }
  cc->SetInputStreamHandler("ImmediateInputStreamHandler");
  cc->SetProcessTimestampBounds(true);
  return absl::OkStatus();
}

absl::Status SwitchMuxCalculator::Open(CalculatorContext* cc) {
  channel_index_ = tool::GetChannelIndex(*cc, channel_index_);
  channel_tags_ = ChannelTags(cc->Inputs().TagMap());

  // Relay side packets only from channel_index_.
  for (const std::string& tag : ChannelTags(cc->InputSidePackets().TagMap())) {
    int num_outputs = cc->OutputSidePackets().NumEntries(tag);
    for (int index = 0; index < num_outputs; ++index) {
      std::string input_tag = tool::ChannelTag(tag, channel_index_);
      Packet input = cc->InputSidePackets().Get(input_tag, index);
      cc->OutputSidePackets().Get(tag, index).Set(input);
    }
  }
  return absl::OkStatus();
}

absl::Status SwitchMuxCalculator::Process(CalculatorContext* cc) {
  // Update the input channel index if specified.
  channel_index_ = tool::GetChannelIndex(*cc, channel_index_);

  // Relay packets and timestamps only from channel_index_.
  for (const std::string& tag : channel_tags_) {
    for (int index = 0; index < cc->Outputs().NumEntries(tag); ++index) {
      auto& output = cc->Outputs().Get(tag, index);
      std::string input_tag = tool::ChannelTag(tag, channel_index_);
      auto& input = cc->Inputs().Get(input_tag, index);
      tool::Relay(input, &output);
    }
  }
  return absl::OkStatus();
}

}  // namespace mediapipe
