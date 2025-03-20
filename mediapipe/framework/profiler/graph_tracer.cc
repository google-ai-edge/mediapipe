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

#include "mediapipe/framework/profiler/graph_tracer.h"

#include <atomic>

#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/input_stream_shard.h"
#include "mediapipe/framework/output_stream_shard.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/profiler/trace_builder.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {

namespace {
using EventType = GraphTrace::EventType;

const absl::Duration kDefaultTraceLogInterval = absl::Milliseconds(500);

// Returns a unique identifier for the current thread.
inline int GetCurrentThreadId() {
  static std::atomic<int> next_thread_id = 0;
  static thread_local int thread_id = next_thread_id++;
  return thread_id;
}

}  // namespace

absl::Duration GraphTracer::GetTraceLogInterval() {
  return profiler_config_.trace_log_interval_usec()
             ? absl::Microseconds(profiler_config_.trace_log_interval_usec())
             : kDefaultTraceLogInterval;
}

int64_t GraphTracer::GetTraceLogCapacity() {
  return profiler_config_.trace_log_capacity()
             ? profiler_config_.trace_log_capacity()
             : 20000;
}

GraphTracer::GraphTracer(const ProfilerConfig& profiler_config)
    : profiler_config_(profiler_config), trace_buffer_(GetTraceLogCapacity()) {
  for (int disabled : profiler_config_.trace_event_types_disabled()) {
    EventType event_type = static_cast<EventType>(disabled);
    (*trace_event_registry())[event_type].set_enabled(false);
  }
}

TraceEventRegistry* GraphTracer::trace_event_registry() {
  return trace_builder_.trace_event_registry();
}

void GraphTracer::LogEvent(TraceEvent event) {
  if (!(*trace_event_registry())[event.event_type].enabled()) {
    return;
  }
  event.set_thread_id(GetCurrentThreadId());
  trace_buffer_.push_back(event);
}

void GraphTracer::LogInputEvents(GraphTrace::EventType event_type,
                                 const CalculatorContext* context,
                                 absl::Time event_time) {
  Timestamp input_ts = context->InputTimestamp();
  for (const InputStreamShard& in_stream : context->Inputs()) {
    const Packet& packet = in_stream.Value();
    if (!packet.IsEmpty()) {
      const std::string* stream_id = &in_stream.Name();
      LogEvent(TraceEvent(event_type)
                   .set_event_time(event_time)
                   .set_is_finish(false)
                   .set_input_ts(input_ts)
                   .set_node_id(context->NodeId())
                   .set_stream_id(stream_id)
                   .set_packet_ts(packet.Timestamp())
                   .set_packet_data_id(&packet));
    }
  }
}

void GraphTracer::LogOutputEvents(GraphTrace::EventType event_type,
                                  const CalculatorContext* context,
                                  absl::Time event_time) {
  // For source nodes, the first output timestamp is used as the input_ts.
  Timestamp input_ts = (context->Inputs().NumEntries() > 0)
                           ? context->InputTimestamp()
                           : GetOutputTimestamp(context);
  for (const OutputStreamShard& out_stream : context->Outputs()) {
    const std::string* stream_id = &out_stream.Name();
    for (const Packet& packet : *out_stream.OutputQueue()) {
      LogEvent(TraceEvent(event_type)
                   .set_event_time(event_time)
                   .set_is_finish(true)
                   .set_input_ts(input_ts)
                   .set_node_id(context->NodeId())
                   .set_stream_id(stream_id)
                   .set_packet_ts(packet.Timestamp())
                   .set_packet_data_id(&packet));
    }
  }
}

Timestamp GraphTracer::TimestampAfter(absl::Time begin_time) {
  return TraceBuilder::TimestampAfter(trace_buffer_, begin_time);
}

// The mutex to guard GraphTracer::trace_builder_.
absl::Mutex* trace_builder_mutex() {
  static absl::Mutex trace_builder_mutex(absl::kConstInit);
  return &trace_builder_mutex;
}

void GraphTracer::GetTrace(absl::Time begin_time, absl::Time end_time,
                           GraphTrace* result) {
  absl::MutexLock lock(trace_builder_mutex());
  trace_builder_.CreateTrace(trace_buffer_, begin_time, end_time, result);
  trace_builder_.Clear();
}

void GraphTracer::GetLog(absl::Time begin_time, absl::Time end_time,
                         GraphTrace* result) {
  absl::MutexLock lock(trace_builder_mutex());
  trace_builder_.CreateLog(trace_buffer_, begin_time, end_time, result);
  trace_builder_.Clear();
}

const TraceBuffer& GraphTracer::GetTraceBuffer() { return trace_buffer_; }

Timestamp GraphTracer::GetOutputTimestamp(const CalculatorContext* context) {
  for (const OutputStreamShard& out_stream : context->Outputs()) {
    for (const Packet& packet : *out_stream.OutputQueue()) {
      if (packet.Timestamp() != Timestamp::Unset()) {
        return packet.Timestamp();
      }
    }
  }
  return Timestamp();
}

}  // namespace mediapipe
