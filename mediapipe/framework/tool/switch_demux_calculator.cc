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
#include <iterator>
#include <memory>
#include <queue>
#include <set>
#include <string>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/tool/container_util.h"
#include "mediapipe/framework/tool/switch_container.pb.h"

namespace mediapipe {

// A calculator to redirect a set of input streams to one of several output
// channels, each consisting of corresponding output streams.  Each channel
// is distinguished by a tag-prefix such as "C1__".  For example:
//
//         node {
//           calculator: "SwitchDemuxCalculator"
//           input_stream: "ENABLE:enable"
//           input_stream: "FUNC_INPUT:foo"
//           input_stream: "FUNC_INPUT:bar"
//           output_stream: "C0__FUNC_INPUT:foo_0"
//           output_stream: "C0__FUNC_INPUT:bar_0"
//           output_stream: "C1__FUNC_INPUT:foo_1"
//           output_stream: "C1__FUNC_INPUT:bar_1"
//         }
//
// Input stream "ENABLE" specifies routing of packets to either channel 0
// or channel 1, given "ENABLE:false" or "ENABLE:true" respectively.
// Input-side-packet "ENABLE" and input-stream "SELECT" can also be used
// similarly to specify the active channel.
//
// SwitchDemuxCalculator is used by SwitchContainer to enable one of several
// contained subgraph or calculator nodes.
//
class SwitchDemuxCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  absl::Status RecordPackets(CalculatorContext* cc);
  int ChannelIndex(Timestamp timestamp);
  absl::Status SendActivePackets(CalculatorContext* cc);

 private:
  int channel_index_;
  std::set<std::string> channel_tags_;
  using PacketQueue = std::map<CollectionItemId, std::queue<Packet>>;
  PacketQueue input_queue_;
  std::map<Timestamp, int> channel_history_;
};
REGISTER_CALCULATOR(SwitchDemuxCalculator);

namespace {
static constexpr char kSelectTag[] = "SELECT";
static constexpr char kEnableTag[] = "ENABLE";

// Returns the last received timestamp for an input stream.
inline Timestamp SettledTimestamp(const InputStreamShard& input) {
  return input.Value().Timestamp();
}

// Returns the last received timestamp for channel selection.
inline Timestamp ChannelSettledTimestamp(CalculatorContext* cc) {
  Timestamp result = Timestamp::Done();
  if (cc->Inputs().HasTag(kEnableTag)) {
    result = SettledTimestamp(cc->Inputs().Tag(kEnableTag));
  } else if (cc->Inputs().HasTag(kSelectTag)) {
    result = SettledTimestamp(cc->Inputs().Tag(kSelectTag));
  }
  return result;
}
}  // namespace

absl::Status SwitchDemuxCalculator::GetContract(CalculatorContract* cc) {
  // Allow any one of kSelectTag, kEnableTag.
  cc->Inputs().Tag(kSelectTag).Set<int>().Optional();
  cc->Inputs().Tag(kEnableTag).Set<bool>().Optional();
  // Allow any one of kSelectTag, kEnableTag.
  cc->InputSidePackets().Tag(kSelectTag).Set<int>().Optional();
  cc->InputSidePackets().Tag(kEnableTag).Set<bool>().Optional();

  // Set the types for all output channels to corresponding input types.
  std::set<std::string> channel_tags = ChannelTags(cc->Outputs().TagMap());
  int channel_count = ChannelCount(cc->Outputs().TagMap());
  for (const std::string& tag : channel_tags) {
    for (int index = 0; index < cc->Inputs().NumEntries(tag); ++index) {
      auto input_id = cc->Inputs().GetId(tag, index);
      if (input_id.IsValid()) {
        cc->Inputs().Get(tag, index).SetAny();
        for (int channel = 0; channel < channel_count; ++channel) {
          auto output_id =
              cc->Outputs().GetId(tool::ChannelTag(tag, channel), index);
          if (output_id.IsValid()) {
            cc->Outputs().Get(output_id).SetSameAs(&cc->Inputs().Get(input_id));
          }
        }
      }
    }
  }
  channel_tags = ChannelTags(cc->OutputSidePackets().TagMap());
  channel_count = ChannelCount(cc->OutputSidePackets().TagMap());
  for (const std::string& tag : channel_tags) {
    int num_entries = cc->InputSidePackets().NumEntries(tag);
    for (int index = 0; index < num_entries; ++index) {
      auto input_id = cc->InputSidePackets().GetId(tag, index);
      if (input_id.IsValid()) {
        cc->InputSidePackets().Get(tag, index).SetAny();
        for (int channel = 0; channel < channel_count; ++channel) {
          auto output_id = cc->OutputSidePackets().GetId(
              tool::ChannelTag(tag, channel), index);
          if (output_id.IsValid()) {
            cc->OutputSidePackets().Get(output_id).SetSameAs(
                &cc->InputSidePackets().Get(input_id));
          }
        }
      }
    }
  }
  auto& options = cc->Options<mediapipe::SwitchContainerOptions>();
  if (!options.synchronize_io()) {
    cc->SetInputStreamHandler("ImmediateInputStreamHandler");
  }
  cc->SetProcessTimestampBounds(true);
  return absl::OkStatus();
}

absl::Status SwitchDemuxCalculator::Open(CalculatorContext* cc) {
  channel_index_ = tool::GetChannelIndex(*cc, channel_index_);
  channel_tags_ = ChannelTags(cc->Outputs().TagMap());
  channel_history_[Timestamp::Unstarted()] = channel_index_;

  // Relay side packets to all channels.
  // Note: This is necessary because Calculator::Open only proceeds when every
  // anticipated side-packet arrives.
  int side_channel_count = tool::ChannelCount(cc->OutputSidePackets().TagMap());
  for (const std::string& tag : ChannelTags(cc->OutputSidePackets().TagMap())) {
    int num_entries = cc->InputSidePackets().NumEntries(tag);
    for (int index = 0; index < num_entries; ++index) {
      Packet input = cc->InputSidePackets().Get(tag, index);
      for (int channel = 0; channel < side_channel_count; ++channel) {
        std::string output_tag = tool::ChannelTag(tag, channel);
        auto output_id = cc->OutputSidePackets().GetId(output_tag, index);
        if (output_id.IsValid()) {
          cc->OutputSidePackets().Get(output_tag, index).Set(input);
        }
      }
    }
  }

  // Relay headers to all channels.
  int output_channel_count = tool::ChannelCount(cc->Outputs().TagMap());
  for (const std::string& tag : ChannelTags(cc->Outputs().TagMap())) {
    int num_entries = cc->Inputs().NumEntries(tag);
    for (int index = 0; index < num_entries; ++index) {
      auto& input = cc->Inputs().Get(tag, index);
      if (input.Header().IsEmpty()) continue;
      for (int channel = 0; channel < output_channel_count; ++channel) {
        std::string output_tag = tool::ChannelTag(tag, channel);
        auto output_id = cc->Outputs().GetId(output_tag, index);
        if (output_id.IsValid()) {
          cc->Outputs().Get(output_tag, index).SetHeader(input.Header());
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status SwitchDemuxCalculator::Process(CalculatorContext* cc) {
  MP_RETURN_IF_ERROR(RecordPackets(cc));
  MP_RETURN_IF_ERROR(SendActivePackets(cc));
  return absl::OkStatus();
}

// Enqueue all arriving packets and bounds.
absl::Status SwitchDemuxCalculator::RecordPackets(CalculatorContext* cc) {
  // Enqueue any new arriving packets.
  for (const std::string& tag : channel_tags_) {
    for (int index = 0; index < cc->Inputs().NumEntries(tag); ++index) {
      auto input_id = cc->Inputs().GetId(tag, index);
      Packet packet = cc->Inputs().Get(input_id).Value();
      if (packet.Timestamp() == cc->InputTimestamp()) {
        input_queue_[input_id].push(packet);
      }
    }
  }

  // Enque any new input channel and its activation timestamp.
  Timestamp channel_settled = ChannelSettledTimestamp(cc);
  int new_channel_index = tool::GetChannelIndex(*cc, channel_index_);
  if (channel_settled == cc->InputTimestamp() &&
      new_channel_index != channel_index_) {
    channel_index_ = new_channel_index;
    channel_history_[channel_settled] = channel_index_;
  }
  return absl::OkStatus();
}

// Returns the channel index for a Timestamp.
int SwitchDemuxCalculator::ChannelIndex(Timestamp timestamp) {
  auto it = channel_history_.upper_bound(timestamp);
  return it == channel_history_.begin() ? -1 : std::prev(it)->second;
}

// Dispatches all queued input packets with known channels.
absl::Status SwitchDemuxCalculator::SendActivePackets(CalculatorContext* cc) {
  // Dispatch any queued input packets with a defined channel_index.
  Timestamp channel_settled = ChannelSettledTimestamp(cc);
  for (const std::string& tag : channel_tags_) {
    for (int index = 0; index < cc->Inputs().NumEntries(tag); ++index) {
      auto input_id = cc->Inputs().GetId(tag, index);
      auto& queue = input_queue_[input_id];
      while (!queue.empty() && queue.front().Timestamp() <= channel_settled) {
        int channel_index = ChannelIndex(queue.front().Timestamp());
        if (channel_index != -1) {
          std::string output_tag = tool::ChannelTag(tag, channel_index);
          auto output_id = cc->Outputs().GetId(output_tag, index);
          if (output_id.IsValid()) {
            cc->Outputs().Get(output_id).AddPacket(queue.front());
          }
        }
        queue.pop();
      }
    }
  }

  // Discard all select packets not needed for any remaining input packets.
  Timestamp input_settled = Timestamp::Done();
  for (const std::string& tag : channel_tags_) {
    for (int index = 0; index < cc->Inputs().NumEntries(tag); ++index) {
      auto input_id = cc->Inputs().GetId(tag, index);
      Timestamp stream_settled = SettledTimestamp(cc->Inputs().Get(input_id));
      if (!input_queue_[input_id].empty()) {
        Timestamp stream_bound = input_queue_[input_id].front().Timestamp();
        stream_settled =
            std::min(stream_settled, stream_bound.PreviousAllowedInStream());
      }
    }
  }
  Timestamp input_bound = input_settled.NextAllowedInStream();
  auto history_bound = std::prev(channel_history_.upper_bound(input_bound));
  channel_history_.erase(channel_history_.begin(), history_bound);
  return absl::OkStatus();
}

}  // namespace mediapipe
