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
#include "mediapipe/framework/tool/switch_container.pb.h"

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
  mediapipe::SwitchContainerOptions options_;
  // This is used to keep around packets that we've received but not
  // relayed yet (because we may not know which channel we should yet be using
  // when synchronized_io flag is set).
  std::map<Timestamp, std::map<CollectionItemId, Packet>> packet_history_;
  // Historical channel index values for timestamps where we don't have all
  // packets available yet (when synchronized_io flag is set).
  std::map<Timestamp, int> channel_history_;
  // Number of output steams that we already processed for the current output
  // timestamp.
  int current_processed_stream_count_ = 0;
};
REGISTER_CALCULATOR(SwitchMuxCalculator);

absl::Status SwitchMuxCalculator::GetContract(CalculatorContract* cc) {
  // Allow any one of kSelectTag, kEnableTag.
  cc->Inputs().Tag(kSelectTag).Set<int>().Optional();
  cc->Inputs().Tag(kEnableTag).Set<bool>().Optional();
  // Allow any one of kSelectTag, kEnableTag.
  cc->InputSidePackets().Tag(kSelectTag).Set<int>().Optional();
  cc->InputSidePackets().Tag(kEnableTag).Set<bool>().Optional();

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
  options_ = cc->Options<mediapipe::SwitchContainerOptions>();
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

  if (options_.synchronize_io()) {
    // Start with adding input signals into channel_history_ and packet_history_
    if (cc->Inputs().HasTag("ENABLE") &&
        !cc->Inputs().Tag("ENABLE").IsEmpty()) {
      channel_history_[cc->Inputs().Tag("ENABLE").Value().Timestamp()] =
          channel_index_;
    }
    if (cc->Inputs().HasTag("SELECT") &&
        !cc->Inputs().Tag("SELECT").IsEmpty()) {
      channel_history_[cc->Inputs().Tag("SELECT").Value().Timestamp()] =
          channel_index_;
    }
    for (auto input_id = cc->Inputs().BeginId();
         input_id < cc->Inputs().EndId(); ++input_id) {
      auto& entry = cc->Inputs().Get(input_id);
      if (entry.IsEmpty()) {
        continue;
      }
      packet_history_[entry.Value().Timestamp()][input_id] = entry.Value();
    }
    // Now check if we have enough information to produce any outputs.
    while (!channel_history_.empty()) {
      // Look at the oldest unprocessed timestamp.
      auto it = channel_history_.begin();
      auto& packets = packet_history_[it->first];
      int total_streams = 0;
      // Loop over all outputs to see if we have anything new that we can relay.
      for (const std::string& tag : channel_tags_) {
        for (int index = 0; index < cc->Outputs().NumEntries(tag); ++index) {
          ++total_streams;
          auto input_id =
              cc->Inputs().GetId(tool::ChannelTag(tag, it->second), index);
          auto packet_it = packets.find(input_id);
          if (packet_it != packets.end()) {
            cc->Outputs().Get(tag, index).AddPacket(packet_it->second);
            ++current_processed_stream_count_;
          } else if (it->first <
                     cc->Inputs().Get(input_id).Value().Timestamp()) {
            // Getting here means that input stream that corresponds to this
            // output at the timestamp we're trying to process right now has
            // already advanced beyond this timestamp. This means that we will
            // shouldn't expect a packet for this timestamp anymore, and we can
            // safely advance timestamp on the output.
            cc->Outputs()
                .Get(tag, index)
                .SetNextTimestampBound(it->first.NextAllowedInStream());
            ++current_processed_stream_count_;
          }
        }
      }
      if (current_processed_stream_count_ == total_streams) {
        // There's nothing else to wait for at the current timestamp, do the
        // cleanup and move on to the next one.
        packet_history_.erase(it->first);
        channel_history_.erase(it);
        current_processed_stream_count_ = 0;
      } else {
        // We're still missing some packets for the current timestamp. Clean up
        // those that we just relayed and let the rest wait until the next
        // Process() call.
        packets.clear();
        break;
      }
    }
  } else {
    // Relay packets and timestamps only from channel_index_.
    for (const std::string& tag : channel_tags_) {
      for (int index = 0; index < cc->Outputs().NumEntries(tag); ++index) {
        auto& output = cc->Outputs().Get(tag, index);
        std::string input_tag = tool::ChannelTag(tag, channel_index_);
        auto& input = cc->Inputs().Get(input_tag, index);
        tool::Relay(input, &output);
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace mediapipe
