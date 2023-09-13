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

#include "mediapipe/framework/input_stream_manager.h"

#include <type_traits>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/source_location.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/framework/tool/status_util.h"

namespace mediapipe {

absl::Status InputStreamManager::Initialize(const std::string& name,
                                            const PacketType* packet_type,
                                            bool back_edge) {
  name_ = name;
  packet_type_ = packet_type;
  back_edge_ = back_edge;
  PrepareForRun();
  return absl::OkStatus();
}

const std::string& InputStreamManager::Name() const { return name_; }

void InputStreamManager::SetQueueSizeCallbacks(
    QueueSizeCallback becomes_full_callback,
    QueueSizeCallback becomes_not_full_callback) {
  becomes_full_callback_ = becomes_full_callback;
  becomes_not_full_callback_ = becomes_not_full_callback;
}

void InputStreamManager::PrepareForRun() {
  absl::MutexLock stream_lock(&stream_mutex_);
  queue_.clear();
  last_reported_stream_full_ = false;
  num_packets_added_ = 0;
  next_timestamp_bound_ = Timestamp::PreStream();
  last_select_timestamp_ = Timestamp::Unstarted();
  closed_ = false;
  header_ = Packet();
}

bool InputStreamManager::IsEmpty() const {
  absl::MutexLock stream_lock(&stream_mutex_);
  return queue_.empty();
}

Packet InputStreamManager::QueueHead() const {
  absl::MutexLock stream_lock(&stream_mutex_);
  if (queue_.empty()) {
    return Packet();
  }
  return queue_.front();
}

absl::Status InputStreamManager::SetHeader(const Packet& header) {
  if (header.Timestamp() != Timestamp::Unset()) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Headers must not have a timestamp.  Stream: \"" << name_
           << "\".";
  }
  header_ = header;
  return absl::OkStatus();
}

absl::Status InputStreamManager::AddPackets(const std::list<Packet>& container,
                                            bool* notify) {
  return AddOrMovePacketsInternal<const std::list<Packet>&>(container, notify);
}

absl::Status InputStreamManager::MovePackets(std::list<Packet>* container,
                                             bool* notify) {
  return AddOrMovePacketsInternal<std::list<Packet>&>(*container, notify);
}

template <typename Container>
absl::Status InputStreamManager::AddOrMovePacketsInternal(Container container,
                                                          bool* notify) {
  *notify = false;
  bool queue_became_non_empty = false;
  bool queue_became_full = false;
  {
    // Scope to prevent locking the stream when notification is called.
    absl::MutexLock stream_lock(&stream_mutex_);
    if (closed_) {
      return absl::OkStatus();
    }
    // Check if the queue was full before packets came in.
    bool was_queue_full =
        (max_queue_size_ != -1 && queue_.size() >= max_queue_size_);
    // Check if the queue becomes non-empty.
    queue_became_non_empty = queue_.empty() && !container.empty();
    for (auto& packet : container) {
      absl::Status result = packet_type_->Validate(packet);
      if (!result.ok()) {
        return tool::AddStatusPrefix(
            absl::StrCat(
                "Packet type mismatch on a calculator receiving from stream \"",
                name_, "\": "),
            result);
      }

      const Timestamp timestamp = packet.Timestamp();
      if (!timestamp.IsAllowedInStream()) {
        return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
               << "In stream \"" << name_
               << "\", timestamp not specified or set to illegal value: "
               << timestamp.DebugString();
      }
      if (enable_timestamps_) {
        // Check that PostStream(), if used, is the only timestamp used.  This
        // is also true for PreStream() but doesn't need to be checked because
        // Timestamp::PreStream().NextAllowedInStream() is
        // Timestamp::OneOverPostStream().
        if (timestamp == Timestamp::PostStream() && num_packets_added_ > 0) {
          return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
                 << "In stream \"" << name_
                 << "\", a packet at Timestamp::PostStream() must be the only "
                    "Packet in an InputStream.";
        }
        if (timestamp < next_timestamp_bound_) {
          return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
                 << "Packet timestamp mismatch on a calculator receiving from "
                    "stream \""
                 << name_ << "\". Current minimum expected timestamp is "
                 << next_timestamp_bound_.DebugString() << " but received "
                 << timestamp.DebugString()
                 << ". Are you using a custom InputStreamHandler? Note that "
                    "some InputStreamHandlers allow timestamps that are not "
                    "strictly monotonically increasing. See for example the "
                    "ImmediateInputStreamHandler class comment.";
        }
      }
      next_timestamp_bound_ = timestamp.NextAllowedInStream();

      // If the caller is MovePackets(), packet's underlying holder should be
      // transferred into queue_. Otherwise, queue_ keeps a copy of the packet.
      ++num_packets_added_;
      VLOG(3) << "Input stream:" << name_
              << " has added packet at time: " << packet.Timestamp();
      if (std::is_const<
              typename std::remove_reference<Container>::type>::value) {
        queue_.emplace_back(packet);
      } else {
        queue_.emplace_back(std::move(packet));
      }
    }
    queue_became_full = (!was_queue_full && max_queue_size_ != -1 &&
                         queue_.size() >= max_queue_size_);
    if (queue_.size() > 1) {
      VLOG(3) << "Queue size greater than 1: stream name: " << name_
              << " queue_size: " << queue_.size();
    }
    VLOG(3) << "Input stream:" << name_
            << " becomes non-empty status:" << queue_became_non_empty
            << " Size: " << queue_.size();
  }
  if (queue_became_full) {
    VLOG(3) << "Queue became full: " << Name();
    becomes_full_callback_(this, &last_reported_stream_full_);
  }
  *notify = queue_became_non_empty;
  return absl::OkStatus();
}

absl::Status InputStreamManager::SetNextTimestampBound(const Timestamp bound,
                                                       bool* notify) {
  *notify = false;
  {
    // Scope to prevent locking the stream when notification is called.
    absl::MutexLock stream_lock(&stream_mutex_);
    if (closed_) {
      return absl::OkStatus();
    }

    if (enable_timestamps_ && bound < next_timestamp_bound_) {
      return mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
             << "SetNextTimestampBound must be called with a timestamp greater "
                "than or equal to the current bound. In stream \""
             << name_ << "\". Current minimum expected timestamp is "
             << next_timestamp_bound_.DebugString() << " but received "
             << bound.DebugString();
    }

    // Even if enable_timestamps_ is false, Timestamp::Done() is used to
    // indicate the end of stream. So this code is common to both timed and
    // untimed scheduling policies.
    if (bound > next_timestamp_bound_) {
      next_timestamp_bound_ = bound;
      VLOG(3) << "Next timestamp bound for input " << name_ << " is "
              << next_timestamp_bound_;
      if (queue_.empty()) {
        // If the queue was not empty then a change to the next_timestamp_bound_
        // is not detectable by the consumer.
        *notify = true;
      }
    }
  }
  return absl::OkStatus();
}

void InputStreamManager::DisableTimestamps() { enable_timestamps_ = false; }

void InputStreamManager::Close() {
  absl::MutexLock stream_lock(&stream_mutex_);
  if (closed_) {
    return;
  }
  next_timestamp_bound_ = Timestamp::Done();
  last_select_timestamp_ = Timestamp::Done();
  closed_ = true;
}

Timestamp InputStreamManager::MinTimestampOrBound(bool* is_empty) const {
  absl::MutexLock stream_lock(&stream_mutex_);
  if (is_empty) {
    *is_empty = queue_.empty();
  }
  return MinTimestampOrBoundHelper();
}

Timestamp InputStreamManager::MinTimestampOrBoundHelper() const
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(stream_mutex_) {
  return queue_.empty() ? next_timestamp_bound_ : queue_.front().Timestamp();
}

Packet InputStreamManager::PopPacketAtTimestamp(Timestamp timestamp,
                                                int* num_packets_dropped,
                                                bool* stream_is_done) {
  ABSL_CHECK(enable_timestamps_);
  *num_packets_dropped = -1;
  *stream_is_done = false;
  bool queue_became_non_full = false;
  Packet packet;
  {
    absl::MutexLock stream_lock(&stream_mutex_);
    // Make sure timestamp didn't decrease from last time.
    ABSL_CHECK_LE(last_select_timestamp_, timestamp);
    last_select_timestamp_ = timestamp;

    // Make sure AddPacket and SetNextTimestampBound are not called with
    // timestamps we have already passed.
    if (next_timestamp_bound_ <= timestamp) {
      next_timestamp_bound_ = timestamp.NextAllowedInStream();
    }

    VLOG(3) << "Input stream " << name_
            << " selecting at timestamp:" << timestamp.Value()
            << " next timestamp bound: " << next_timestamp_bound_;

    // Advances time to timestamp.
    Timestamp current_timestamp = Timestamp::Unset();

    // Checks if queue is full.
    bool was_queue_full =
        (max_queue_size_ != -1 && queue_.size() >= max_queue_size_);

    while (!queue_.empty() && queue_.front().Timestamp() <= timestamp) {
      packet = std::move(queue_.front());
      queue_.pop_front();
      current_timestamp = packet.Timestamp();
      ++(*num_packets_dropped);
    }
    // Clear value_ if it doesn't have exactly the right timestamp.
    if (current_timestamp != timestamp) {
      // The timestamp bound reported when no packet is sent.
      Timestamp bound = MinTimestampOrBoundHelper();
      packet = Packet().At(bound.PreviousAllowedInStream());
      ++(*num_packets_dropped);
    }

    VLOG(3) << "Input stream removed packets:" << name_
            << " Size:" << queue_.size();
    queue_became_non_full = (was_queue_full && queue_.size() < max_queue_size_);
    *stream_is_done = IsDone();
  }
  if (queue_became_non_full) {
    VLOG(3) << "Queue became non-full: " << Name();
    becomes_not_full_callback_(this, &last_reported_stream_full_);
  }
  return packet;
}

Packet InputStreamManager::PopQueueHead(bool* stream_is_done) {
  ABSL_CHECK(!enable_timestamps_);
  *stream_is_done = false;
  bool queue_became_non_full = false;
  Packet packet;
  {
    absl::MutexLock stream_lock(&stream_mutex_);

    VLOG(3) << "Input stream " << name_ << " selecting at queue head";

    // Check if queue is full.
    bool was_queue_full =
        (max_queue_size_ != -1 && queue_.size() >= max_queue_size_);

    if (!queue_.empty()) {
      packet = std::move(queue_.front());
      queue_.pop_front();
    } else {
      packet = Packet();
    }

    VLOG(3) << "Input stream removed a packet:" << name_
            << " Size:" << queue_.size();
    queue_became_non_full = (was_queue_full && queue_.size() < max_queue_size_);
    *stream_is_done = IsDone();
  }
  if (queue_became_non_full) {
    VLOG(3) << "Queue became non-full: " << Name();
    becomes_not_full_callback_(this, &last_reported_stream_full_);
  }
  return packet;
}

int InputStreamManager::NumPacketsAdded() const {
  absl::MutexLock lock(&stream_mutex_);
  return num_packets_added_;
}

int InputStreamManager::QueueSize() const {
  absl::MutexLock lock(&stream_mutex_);
  return static_cast<int>(queue_.size());
}

int InputStreamManager::MaxQueueSize() const {
  absl::MutexLock lock(&stream_mutex_);
  return max_queue_size_;
}

void InputStreamManager::SetMaxQueueSize(int max_queue_size) {
  bool was_full;
  bool is_full;
  {
    absl::MutexLock lock(&stream_mutex_);
    was_full = (max_queue_size_ != -1 && queue_.size() >= max_queue_size_);
    max_queue_size_ = max_queue_size;
    is_full = (max_queue_size_ != -1 && queue_.size() >= max_queue_size_);
  }

  // QueueSizeCallback is called with no mutexes held.
  if (!was_full && is_full) {
    VLOG(3) << "Queue became full: " << Name();
    becomes_full_callback_(this, &last_reported_stream_full_);
  } else if (was_full && !is_full) {
    VLOG(3) << "Queue became non-full: " << Name();
    becomes_not_full_callback_(this, &last_reported_stream_full_);
  }
}

bool InputStreamManager::IsFull() const {
  absl::MutexLock lock(&stream_mutex_);
  return max_queue_size_ != -1 && queue_.size() >= max_queue_size_;
}

Timestamp InputStreamManager::GetMinTimestampAmongNLatest(int n) const {
  absl::MutexLock lock(&stream_mutex_);
  if (queue_.empty()) {
    return Timestamp::Unset();
  }
  return (queue_.cend() - std::min((size_t)n, queue_.size()))->Timestamp();
}

void InputStreamManager::ErasePacketsEarlierThan(Timestamp timestamp) {
  bool queue_became_non_full = false;
  {
    absl::MutexLock lock(&stream_mutex_);
    // Checks if queue is full.
    bool was_queue_full =
        (max_queue_size_ != -1 && queue_.size() >= max_queue_size_);

    while (!queue_.empty() && queue_.front().Timestamp() < timestamp) {
      queue_.pop_front();
    }

    VLOG(3) << "Input stream removed packets:" << name_
            << " Size:" << queue_.size();
    queue_became_non_full = (was_queue_full && queue_.size() < max_queue_size_);
  }
  if (queue_became_non_full) {
    VLOG(3) << "Queue became non-full: " << Name();
    becomes_not_full_callback_(this, &last_reported_stream_full_);
  }
}

bool InputStreamManager::IsDone() const {
  return queue_.empty() && next_timestamp_bound_ == Timestamp::Done();
}

}  // namespace mediapipe
