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
#include <functional>
#include <iterator>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <type_traits>
#include <utility>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/input_stream_shard.h"
#include "mediapipe/framework/output_stream_shard.h"
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
  // Stores any new input channel history.
  void RecordChannel(CalculatorContext* cc);

  // Temporarily enqueues every new packet or timestamp bounds.
  void RecordPackets(CalculatorContext* cc);

  // Immediately sends any packets or timestamp bounds for settled timestamps.
  void SendActivePackets(CalculatorContext* cc);

 private:
  int channel_index_;
  std::set<std::string> channel_tags_;
  mediapipe::SwitchContainerOptions options_;
  // This is used to keep around packets that we've received but not
  // relayed yet (because we may not know which channel we should yet be using).
  std::map<CollectionItemId, std::queue<Packet>> packet_queue_;
  // Historical channel index values for timestamps where we don't have all
  // packets available yet.
  std::map<Timestamp, int> channel_history_;
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

// Returns the last delivered timestamp for an input stream.
Timestamp SettledTimestamp(const InputStreamShard& input) {
  return input.Value().Timestamp();
}

// Returns the last delivered timestamp for channel selection.
Timestamp ChannelSettledTimestamp(CalculatorContext* cc) {
  Timestamp result = Timestamp::Done();
  if (cc->Inputs().HasTag("ENABLE")) {
    result = SettledTimestamp(cc->Inputs().Tag("ENABLE"));
  } else if (cc->Inputs().HasTag("SELECT")) {
    result = SettledTimestamp(cc->Inputs().Tag("SELECT"));
  }
  return result;
}

absl::Status SwitchMuxCalculator::Open(CalculatorContext* cc) {
  // Initialize channel_index_ and channel_history_.
  options_ = cc->Options<mediapipe::SwitchContainerOptions>();
  channel_index_ = tool::GetChannelIndex(*cc, channel_index_);
  channel_tags_ = ChannelTags(cc->Inputs().TagMap());
  channel_history_[Timestamp::Unstarted()] = channel_index_;

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

void SwitchMuxCalculator::RecordChannel(CalculatorContext* cc) {
  Timestamp channel_settled = ChannelSettledTimestamp(cc);
  int new_channel_index = tool::GetChannelIndex(*cc, channel_index_);

  // Enque any new input channel and its activation timestamp.
  if (channel_settled == cc->InputTimestamp() &&
      new_channel_index != channel_index_) {
    channel_index_ = new_channel_index;
    channel_history_[channel_settled] = channel_index_;
  }
}

void SwitchMuxCalculator::RecordPackets(CalculatorContext* cc) {
  auto select_id = cc->Inputs().GetId("SELECT", 0);
  auto enable_id = cc->Inputs().GetId("ENABLE", 0);
  for (auto id = cc->Inputs().BeginId(); id < cc->Inputs().EndId(); ++id) {
    if (id == select_id || id == enable_id) continue;
    Packet packet = cc->Inputs().Get(id).Value();
    // Enque any new packet or timestamp bound.
    if (packet.Timestamp() == cc->InputTimestamp()) {
      packet_queue_[id].push(packet);
    }
  }
}

void SwitchMuxCalculator::SendActivePackets(CalculatorContext* cc) {
  Timestamp expired_history;
  // Iterate through the recent active input channels.
  for (auto it = channel_history_.begin(); it != channel_history_.end(); ++it) {
    int channel = it->second;
    Timestamp channel_start = it->first;
    Timestamp channel_end =
        (std::next(it) == channel_history_.end())
            ? ChannelSettledTimestamp(cc).NextAllowedInStream()
            : std::next(it)->first;
    Timestamp stream_settled = Timestamp::Done();
    for (const std::string& tag : channel_tags_) {
      std::string input_tag = tool::ChannelTag(tag, channel);
      for (int index = 0; index < cc->Inputs().NumEntries(input_tag); ++index) {
        CollectionItemId input_id = cc->Inputs().GetId(input_tag, index);
        OutputStreamShard& output = cc->Outputs().Get(tag, index);
        std::queue<Packet>& q = packet_queue_[input_id];
        // Send any packets or bounds from a recent active input channel.
        while (!q.empty() && q.front().Timestamp() < channel_end) {
          if (q.front().Timestamp() >= channel_start) {
            output.AddPacket(q.front());
          }
          q.pop();
        }
        stream_settled = std::min(stream_settled,
                                  SettledTimestamp(cc->Inputs().Get(input_id)));
      }
    }

    // A history entry is expired only if all streams have advanced past it.
    if (stream_settled.NextAllowedInStream() < channel_end ||
        std::next(it) == channel_history_.end()) {
      break;
    }
    expired_history = channel_start;

    // Discard any packets or bounds from recent inactive input channels.
    for (auto id = cc->Inputs().BeginId(); id < cc->Inputs().EndId(); ++id) {
      std::queue<Packet>& q = packet_queue_[id];
      while (!q.empty() && q.front().Timestamp() < channel_end) {
        q.pop();
      }
    }
  }

  // Discard any expired channel history entries.
  if (expired_history != Timestamp::Unset()) {
    channel_history_.erase(channel_history_.begin(),
                           std::next(channel_history_.find(expired_history)));
  }
}

absl::Status SwitchMuxCalculator::Process(CalculatorContext* cc) {
  // Normally packets will arrive on the active channel and will be passed
  // through immediately.  In the less common case in which the active input
  // channel is not known for an input packet timestamp, the input packet is
  // queued until the active channel becomes known.
  RecordChannel(cc);
  RecordPackets(cc);
  SendActivePackets(cc);
  return absl::OkStatus();
}

}  // namespace mediapipe
