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

#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

namespace internal {

absl::Status GraphOutputStream::Initialize(
    const std::string& stream_name, const PacketType* packet_type,
    OutputStreamManager* output_stream_manager, bool observe_timestamp_bounds) {
  RET_CHECK(output_stream_manager);

  // Initializes input_stream_handler_ with one input stream as the observer.
  proto_ns::RepeatedPtrField<ProtoString> input_stream_field;
  input_stream_field.Add()->assign(stream_name);
  std::shared_ptr<tool::TagMap> tag_map =
      tool::TagMap::Create(input_stream_field).value();
  input_stream_handler_ = absl::make_unique<GraphOutputStreamHandler>(
      tag_map, /*cc_manager=*/nullptr, MediaPipeOptions(),
      /*calculator_run_in_parallel=*/false);
  input_stream_handler_->SetProcessTimestampBounds(observe_timestamp_bounds);
  const CollectionItemId& id = tag_map->BeginId();
  input_stream_ = absl::make_unique<InputStreamManager>();
  MP_RETURN_IF_ERROR(
      input_stream_->Initialize(stream_name, packet_type, /*back_edge=*/false));
  MP_RETURN_IF_ERROR(input_stream_handler_->InitializeInputStreamManagers(
      input_stream_.get()));
  output_stream_manager->AddMirror(input_stream_handler_.get(), id);
  return absl::OkStatus();
}

void GraphOutputStream::PrepareForRun(
    std::function<void()> notification_callback,
    std::function<void(absl::Status)> error_callback) {
  input_stream_handler_->PrepareForRun(
      /*headers_ready_callback=*/[] {}, std::move(notification_callback),
      /*schedule_callback=*/nullptr, std::move(error_callback));
}

absl::Status OutputStreamObserver::Initialize(
    const std::string& stream_name, const PacketType* packet_type,
    std::function<absl::Status(const Packet&)> packet_callback,
    OutputStreamManager* output_stream_manager, bool observe_timestamp_bounds) {
  RET_CHECK(output_stream_manager);

  packet_callback_ = std::move(packet_callback);
  observe_timestamp_bounds_ = observe_timestamp_bounds;
  return GraphOutputStream::Initialize(stream_name, packet_type,
                                       output_stream_manager,
                                       observe_timestamp_bounds);
}

absl::Status OutputStreamObserver::Notify() {
  // Lets one thread perform packets notification as much as possible.
  // Other threads should quit if a thread is already performing notification.
  {
    absl::MutexLock l(&mutex_);

    if (notifying_ == false) {
      notifying_ = true;
    } else {
      return absl::OkStatus();
    }
  }
  while (true) {
    bool empty;
    Timestamp min_timestamp = input_stream_->MinTimestampOrBound(&empty);
    if (empty) {
      // Emits an empty packet at timestamp_bound.PreviousAllowedInStream().
      if (observe_timestamp_bounds_ && min_timestamp < Timestamp::Done()) {
        Timestamp settled = (min_timestamp == Timestamp::PostStream()
                                 ? Timestamp::PostStream()
                                 : min_timestamp.PreviousAllowedInStream());
        if (last_processed_ts_ < settled) {
          MP_RETURN_IF_ERROR(packet_callback_(Packet().At(settled)));
          last_processed_ts_ = settled;
        }
      }
      // Last check to make sure that the min timestamp or bound doesn't change.
      // If so, flips notifying_ to false to allow any other threads to perform
      // notification when new packets/timestamp bounds arrive. Otherwise, in
      // case of the min timestamp or bound getting updated, jumps to the
      // beginning of the notification loop for a new iteration.
      {
        absl::MutexLock l(&mutex_);
        Timestamp new_min_timestamp =
            input_stream_->MinTimestampOrBound(&empty);
        if (new_min_timestamp == min_timestamp) {
          notifying_ = false;
          break;
        } else {
          continue;
        }
      }
    }
    int num_packets_dropped = 0;
    bool stream_is_done = false;
    Packet packet = input_stream_->PopPacketAtTimestamp(
        min_timestamp, &num_packets_dropped, &stream_is_done);
    RET_CHECK_EQ(num_packets_dropped, 0).SetNoLogging()
        << absl::Substitute("Dropped $0 packet(s) on input stream \"$1\".",
                            num_packets_dropped, input_stream_->Name());
    MP_RETURN_IF_ERROR(packet_callback_(packet));
    last_processed_ts_ = min_timestamp;
  }
  return absl::OkStatus();
}

absl::Status OutputStreamPollerImpl::Initialize(
    const std::string& stream_name, const PacketType* packet_type,
    std::function<void(InputStreamManager*, bool*)> queue_size_callback,
    OutputStreamManager* output_stream_manager, bool observe_timestamp_bounds) {
  MP_RETURN_IF_ERROR(GraphOutputStream::Initialize(stream_name, packet_type,
                                                   output_stream_manager,
                                                   observe_timestamp_bounds));
  input_stream_handler_->SetQueueSizeCallbacks(queue_size_callback,
                                               queue_size_callback);
  return absl::OkStatus();
}

void OutputStreamPollerImpl::PrepareForRun(
    std::function<void()> notification_callback,
    std::function<void(absl::Status)> error_callback) {
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

absl::Status OutputStreamPollerImpl::Notify() {
  mutex_.Lock();
  handler_condvar_.Signal();
  mutex_.Unlock();
  return absl::OkStatus();
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
  bool timestamp_bound_changed = false;
  Timestamp min_timestamp = Timestamp::Unset();
  mutex_.Lock();
  while (true) {
    min_timestamp = input_stream_->MinTimestampOrBound(&empty_queue);
    if (empty_queue) {
      timestamp_bound_changed =
          input_stream_handler_->ProcessTimestampBounds() &&
          output_timestamp_ < min_timestamp.PreviousAllowedInStream();
    }
    if (graph_has_error_ || !empty_queue || timestamp_bound_changed ||
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
  if (empty_queue) {
    output_timestamp_ = min_timestamp.PreviousAllowedInStream();
  } else {
    output_timestamp_ = min_timestamp;
  }
  mutex_.Unlock();
  if (min_timestamp == Timestamp::Done()) {
    return false;
  }
  if (!empty_queue) {
    int num_packets_dropped = 0;
    bool stream_is_done = false;
    *packet = input_stream_->PopPacketAtTimestamp(
        min_timestamp, &num_packets_dropped, &stream_is_done);
    CHECK_EQ(num_packets_dropped, 0)
        << absl::Substitute("Dropped $0 packet(s) on input stream \"$1\".",
                            num_packets_dropped, input_stream_->Name());
  } else if (timestamp_bound_changed) {
    *packet = Packet().At(min_timestamp.PreviousAllowedInStream());
  }
  return true;
}

}  // namespace internal
}  // namespace mediapipe
