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

#include "mediapipe/framework/graph_output_stream.h"

namespace mediapipe {

namespace internal {

::mediapipe::Status GraphOutputStream::Initialize(
    const std::string& stream_name, const PacketType* packet_type,
    OutputStreamManager* output_stream_manager) {
  RET_CHECK(output_stream_manager);

  // Initializes input_stream_handler_ with one input stream as the observer.
  proto_ns::RepeatedPtrField<ProtoString> input_stream_field;
  input_stream_field.Add()->assign(stream_name);
  std::shared_ptr<tool::TagMap> tag_map =
      tool::TagMap::Create(input_stream_field).ValueOrDie();
  input_stream_handler_ = absl::make_unique<GraphOutputStreamHandler>(
      tag_map, /*cc_manager=*/nullptr, MediaPipeOptions(),
      /*calculator_run_in_parallel=*/false);
  const CollectionItemId& id = tag_map->BeginId();
  input_stream_ = absl::make_unique<InputStreamManager>();
  MP_RETURN_IF_ERROR(
      input_stream_->Initialize(stream_name, packet_type, /*back_edge=*/false));
  MP_RETURN_IF_ERROR(input_stream_handler_->InitializeInputStreamManagers(
      input_stream_.get()));
  output_stream_manager->AddMirror(input_stream_handler_.get(), id);
  return ::mediapipe::OkStatus();
}

void GraphOutputStream::PrepareForRun(
    std::function<void()> notification_callback,
    std::function<void(::mediapipe::Status)> error_callback) {
  input_stream_handler_->PrepareForRun(
      /*headers_ready_callback=*/[] {}, std::move(notification_callback),
      /*schedule_callback=*/nullptr, std::move(error_callback));
}

::mediapipe::Status OutputStreamObserver::Initialize(
    const std::string& stream_name, const PacketType* packet_type,
    std::function<::mediapipe::Status(const Packet&)> packet_callback,
    OutputStreamManager* output_stream_manager) {
  RET_CHECK(output_stream_manager);

  packet_callback_ = std::move(packet_callback);
  return GraphOutputStream::Initialize(stream_name, packet_type,
                                       output_stream_manager);
}

::mediapipe::Status OutputStreamObserver::Notify() {
  while (true) {
    bool empty;
    Timestamp min_timestamp = input_stream_->MinTimestampOrBound(&empty);
    if (empty) {
      break;
    }
    int num_packets_dropped = 0;
    bool stream_is_done = false;
    Packet packet = input_stream_->PopPacketAtTimestamp(
        min_timestamp, &num_packets_dropped, &stream_is_done);
    RET_CHECK_EQ(num_packets_dropped, 0).SetNoLogging()
        << absl::Substitute("Dropped $0 packet(s) on input stream \"$1\".",
                            num_packets_dropped, input_stream_->Name());
    MP_RETURN_IF_ERROR(packet_callback_(packet));
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status OutputStreamPollerImpl::Initialize(
    const std::string& stream_name, const PacketType* packet_type,
    std::function<void(InputStreamManager*, bool*)> queue_size_callback,
    OutputStreamManager* output_stream_manager) {
  MP_RETURN_IF_ERROR(GraphOutputStream::Initialize(stream_name, packet_type,
                                                   output_stream_manager));
  input_stream_handler_->SetQueueSizeCallbacks(queue_size_callback,
                                               queue_size_callback);
  return ::mediapipe::OkStatus();
}

void OutputStreamPollerImpl::PrepareForRun(
    std::function<void()> notification_callback,
    std::function<void(::mediapipe::Status)> error_callback) {
  input_stream_handler_->PrepareForRun(
      /*headers_ready_callback=*/[] {}, std::move(notification_callback),
      /*schedule_callback=*/nullptr, std::move(error_callback));
  mutex_.Lock();
  graph_has_error_ = false;
  mutex_.Unlock();
}

void OutputStreamPollerImpl::Reset() {
  mutex_.Lock();
  graph_has_error_ = false;
  input_stream_->PrepareForRun();
  mutex_.Unlock();
}

void OutputStreamPollerImpl::SetMaxQueueSize(int queue_size) {
  CHECK(queue_size >= -1)
      << "Max queue size must be either -1 or non-negative.";
  input_stream_handler_->SetMaxQueueSize(queue_size);
}

int OutputStreamPollerImpl::QueueSize() { return input_stream_->QueueSize(); }

::mediapipe::Status OutputStreamPollerImpl::Notify() {
  mutex_.Lock();
  handler_condvar_.Signal();
  mutex_.Unlock();
  return ::mediapipe::OkStatus();
}

void OutputStreamPollerImpl::NotifyError() {
  mutex_.Lock();
  graph_has_error_ = true;
  handler_condvar_.Signal();
  mutex_.Unlock();
}

bool OutputStreamPollerImpl::Next(Packet* packet) {
  CHECK(packet);
  bool empty_queue = true;
  Timestamp min_timestamp = Timestamp::Unset();
  mutex_.Lock();
  while (true) {
    min_timestamp = input_stream_->MinTimestampOrBound(&empty_queue);
    if (graph_has_error_ || !empty_queue ||
        min_timestamp == Timestamp::Done()) {
      break;
    } else {
      handler_condvar_.Wait(&mutex_);
    }
  }
  if (graph_has_error_ && empty_queue) {
    mutex_.Unlock();
    return false;
  }
  mutex_.Unlock();
  if (min_timestamp == Timestamp::Done()) {
    return false;
  }
  int num_packets_dropped = 0;
  bool stream_is_done = false;
  *packet = input_stream_->PopPacketAtTimestamp(
      min_timestamp, &num_packets_dropped, &stream_is_done);
  CHECK_EQ(num_packets_dropped, 0)
      << absl::Substitute("Dropped $0 packet(s) on input stream \"$1\".",
                          num_packets_dropped, input_stream_->Name());
  return true;
}

}  // namespace internal
}  // namespace mediapipe
