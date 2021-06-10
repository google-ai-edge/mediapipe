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

#include "mediapipe/framework/input_stream_shard.h"

namespace mediapipe {

void InputStreamShard::AddPacket(Packet&& value, bool is_done) {
  // A packet can be added if the shard is still active or the packet being
  // added is empty. An empty packet corresponds to absence of a packet.
  CHECK(!is_done_ || value.IsEmpty());
  packet_queue_.emplace(std::move(value));
  is_done_ = is_done;
}

}  // namespace mediapipe
