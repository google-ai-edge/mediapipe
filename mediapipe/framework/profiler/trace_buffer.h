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

#ifndef MEDIAPIPE_FRAMEWORK_PROFILER_TRACE_BUFFER_H_
#define MEDIAPIPE_FRAMEWORK_PROFILER_TRACE_BUFFER_H_

#include "absl/time/time.h"
#include "mediapipe/framework/calculator_profile.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/profiler/circular_buffer.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {

// Packet content identifier.
using PacketDataId = const void*;

namespace packet_internal {
// Returns the packet data address for a packet data holder.
inline const void* GetPacketDataId(const HolderBase* holder) {
  return (holder == nullptr)
             ? nullptr
             : &(static_cast<const Holder<int>*>(holder)->data());
}
}  // namespace packet_internal

// Packet trace log event.
struct TraceEvent {
  using EventType = GraphTrace::EventType;
  // GraphTrace::EventType constants, repeated here to match GraphProfilerStub.
  static constexpr EventType UNKNOWN = GraphTrace::UNKNOWN;
  static constexpr EventType OPEN = GraphTrace::OPEN;
  static constexpr EventType PROCESS = GraphTrace::PROCESS;
  static constexpr EventType CLOSE = GraphTrace::CLOSE;
  static constexpr EventType NOT_READY = GraphTrace::NOT_READY;
  static constexpr EventType READY_FOR_PROCESS = GraphTrace::READY_FOR_PROCESS;
  static constexpr EventType READY_FOR_CLOSE = GraphTrace::READY_FOR_CLOSE;
  static constexpr EventType THROTTLED = GraphTrace::THROTTLED;
  static constexpr EventType UNTHROTTLED = GraphTrace::UNTHROTTLED;
  static constexpr EventType CPU_TASK_USER = GraphTrace::CPU_TASK_USER;
  static constexpr EventType CPU_TASK_SYSTEM = GraphTrace::CPU_TASK_SYSTEM;
  static constexpr EventType GPU_TASK = GraphTrace::GPU_TASK;
  static constexpr EventType DSP_TASK = GraphTrace::DSP_TASK;
  static constexpr EventType TPU_TASK = GraphTrace::TPU_TASK;
  absl::Time event_time;
  EventType event_type = UNKNOWN;
  bool is_finish = false;
  Timestamp input_ts = Timestamp::Unset();
  Timestamp packet_ts = Timestamp::Unset();
  int node_id = -1;
  const std::string* stream_id = nullptr;
  PacketDataId packet_data_id = 0;
  int thread_id = 0;

  TraceEvent(const EventType& event_type) : event_type(event_type) {}
  TraceEvent() {}

  inline TraceEvent& set_event_time(absl::Time event_time) {
    this->event_time = event_time;
    return *this;
  }
  inline TraceEvent& set_event_type(const EventType& event_type) {
    this->event_type = event_type;
    return *this;
  }
  inline TraceEvent& set_node_id(int node_id) {
    this->node_id = node_id;
    return *this;
  }
  inline TraceEvent& set_stream_id(const std::string* stream_id) {
    this->stream_id = stream_id;
    return *this;
  }
  inline TraceEvent& set_input_ts(Timestamp input_ts) {
    this->input_ts = input_ts;
    return *this;
  }
  inline TraceEvent& set_packet_ts(Timestamp packet_ts) {
    this->packet_ts = packet_ts;
    return *this;
  }
  inline TraceEvent& set_packet_data_id(const Packet* packet) {
    this->packet_data_id =
        packet_internal::GetPacketDataId(packet_internal::GetHolder(*packet));
    return *this;
  }
  inline TraceEvent& set_thread_id(int thread_id) {
    this->thread_id = thread_id;
    return *this;
  }
  inline TraceEvent& set_is_finish(bool is_finish) {
    this->is_finish = is_finish;
    return *this;
  }
};

// Packet trace log buffer.
using TraceBuffer = CircularBuffer<TraceEvent>;

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_PROFILER_TRACE_BUFFER_H_
