#include "mediapipe/framework/tool/container_util.h"

#include "mediapipe/framework/tool/switch_container.pb.h"

namespace mediapipe {
namespace tool {

std::string ChannelTag(const std::string& tag, int channel) {
  return absl::StrCat("C", channel, "__", tag);
}

// Parses a tag name starting with a channel prefix, like "C2__".
bool ParseChannelTag(const std::string& channel_name, std::string* name,
                     std::string* num) {
  int pos = channel_name.find('C');
  int sep = channel_name.find("__");
  if (pos != 0 || sep == std::string::npos) {
    return false;
  }
  *num = channel_name.substr(pos + 1, sep - (pos + 1));
  *name = channel_name.substr(sep + 2);
  return true;
}

std::set<std::string> ChannelTags(const std::shared_ptr<tool::TagMap>& map) {
  std::set<std::string> result;
  for (const std::string& tag : map->GetTags()) {
    std::string name, num;
    if (ParseChannelTag(tag, &name, &num)) {
      result.insert(name);
    }
  }
  return result;
}

int ChannelCount(const std::shared_ptr<tool::TagMap>& map) {
  int count = 0;
  for (const std::string& tag : map->GetTags()) {
    std::string name, num;
    int channel = -1;
    if (ParseChannelTag(tag, &name, &num)) {
      if (absl::SimpleAtoi(num, &channel)) {
        count = std::max(count, channel + 1);
      }
    }
  }
  return count;
}

void Relay(const InputStreamShard& input, OutputStreamShard* output) {
  if (input.IsEmpty()) {
    Timestamp input_bound = input.Value().Timestamp().NextAllowedInStream();
    if (output->NextTimestampBound() < input_bound) {
      output->SetNextTimestampBound(input_bound);
    }
  } else {
    output->AddPacket(input.Value());
  }
}

int GetChannelIndex(const CalculatorContext& cc, int previous_index) {
  int result = previous_index;
  Packet select_packet;
  Packet enable_packet;
  if (cc.InputTimestamp() == Timestamp::Unstarted()) {
    auto& options = cc.Options<mediapipe::SwitchContainerOptions>();
    if (options.has_enable()) {
      result = options.enable() ? 1 : 0;
    }
    if (options.has_select()) {
      result = options.select();
    }
    if (cc.InputSidePackets().HasTag("ENABLE")) {
      enable_packet = cc.InputSidePackets().Tag("ENABLE");
    }
    if (cc.InputSidePackets().HasTag("SELECT")) {
      select_packet = cc.InputSidePackets().Tag("SELECT");
    }
  } else {
    if (cc.Inputs().HasTag("ENABLE")) {
      enable_packet = cc.Inputs().Tag("ENABLE").Value();
    }
    if (cc.Inputs().HasTag("SELECT")) {
      select_packet = cc.Inputs().Tag("SELECT").Value();
    }
  }
  if (!enable_packet.IsEmpty()) {
    result = enable_packet.Get<bool>() ? 1 : 0;
  }
  if (!select_packet.IsEmpty()) {
    result = select_packet.Get<int>();
  }
  return result;
}

}  // namespace tool
}  // namespace mediapipe
