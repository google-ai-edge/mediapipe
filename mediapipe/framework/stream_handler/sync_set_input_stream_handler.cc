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
#include "mediapipe/framework/stream_handler/sync_set_input_stream_handler.h"

#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/input_stream_handler.h"
#include "mediapipe/framework/packet_set.h"
#include "mediapipe/framework/port/map_util.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/stream_handler/sync_set_input_stream_handler.pb.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {

REGISTER_INPUT_STREAM_HANDLER(SyncSetInputStreamHandler);

void SyncSetInputStreamHandler::PrepareForRun(
    std::function<void()> headers_ready_callback,
    std::function<void()> notification_callback,
    std::function<void(CalculatorContext*)> schedule_callback,
    std::function<void(absl::Status)> error_callback) {
  const auto& handler_options =
      options_.GetExtension(mediapipe::SyncSetInputStreamHandlerOptions::ext);
  {
    absl::MutexLock lock(&mutex_);
    sync_sets_.clear();
    std::set<CollectionItemId> used_ids;
    for (const auto& sync_set : handler_options.sync_set()) {
      std::vector<CollectionItemId> stream_ids;
      ABSL_CHECK_LT(0, sync_set.tag_index_size());
      for (const auto& tag_index : sync_set.tag_index()) {
        std::string tag;
        int index;
        MEDIAPIPE_CHECK_OK(tool::ParseTagIndex(tag_index, &tag, &index));
        CollectionItemId id = input_stream_managers_.GetId(tag, index);
        ABSL_CHECK(id.IsValid())
            << "stream \"" << tag_index << "\" is not found.";
        ABSL_CHECK(!mediapipe::ContainsKey(used_ids, id))
            << "stream \"" << tag_index << "\" is in more than one sync set.";
        used_ids.insert(id);
        stream_ids.push_back(id);
      }
      sync_sets_.emplace_back(this, std::move(stream_ids));
    }
    std::vector<CollectionItemId> remaining_ids;
    for (CollectionItemId id = input_stream_managers_.BeginId();
         id < input_stream_managers_.EndId(); ++id) {
      if (!mediapipe::ContainsKey(used_ids, id)) {
        remaining_ids.push_back(id);
      }
    }
    if (!remaining_ids.empty()) {
      sync_sets_.emplace_back(this, std::move(remaining_ids));
    }
    ready_sync_set_index_ = -1;
    ready_timestamp_ = Timestamp::Done();
  }

  InputStreamHandler::PrepareForRun(
      std::move(headers_ready_callback), std::move(notification_callback),
      std::move(schedule_callback), std::move(error_callback));
}

NodeReadiness SyncSetInputStreamHandler::GetNodeReadiness(
    Timestamp* min_stream_timestamp) {
  ABSL_DCHECK(min_stream_timestamp);
  absl::MutexLock lock(&mutex_);
  if (ready_sync_set_index_ >= 0) {
    *min_stream_timestamp = ready_timestamp_;
    // TODO: Return kNotReady unless a new ready syncset is found.
    return NodeReadiness::kReadyForProcess;
  }
  for (int sync_set_index = 0; sync_set_index < sync_sets_.size();
       ++sync_set_index) {
    NodeReadiness readiness =
        sync_sets_[sync_set_index].GetReadiness(min_stream_timestamp);
    if (readiness == NodeReadiness::kReadyForClose) {
      // This sync set is done, remove it.  Note that this invalidates
      // sync set indexes higher than sync_set_index.  However, we are
      // guaranteed that we were not ready before entering the outer
      // loop, so even if we are ready now, ready_sync_set_index_ must
      // be less than the current value of sync_set_index.
      sync_sets_.erase(sync_sets_.begin() + sync_set_index);
      --sync_set_index;
      continue;
    }

    if (readiness == NodeReadiness::kReadyForProcess) {
      // TODO: Prioritize sync-sets to avoid starvation.
      if (*min_stream_timestamp < ready_timestamp_) {
        // Store the timestamp and corresponding sync set index for the
        // sync set with the earliest arrival timestamp.
        ready_timestamp_ = *min_stream_timestamp;
        ready_sync_set_index_ = sync_set_index;
      }
    }
  }
  if (ready_sync_set_index_ >= 0) {
    *min_stream_timestamp = ready_timestamp_;
    return NodeReadiness::kReadyForProcess;
  }
  if (sync_sets_.empty()) {
    *min_stream_timestamp = Timestamp::Done();
    return NodeReadiness::kReadyForClose;
  }
  // TODO The value of *min_stream_timestamp is undefined in this case.
  return NodeReadiness::kNotReady;
}

void SyncSetInputStreamHandler::FillInputSet(Timestamp input_timestamp,
                                             InputStreamShardSet* input_set) {
  // Assume that all current packets are already cleared.
  absl::MutexLock lock(&mutex_);
  ABSL_CHECK_LE(0, ready_sync_set_index_);
  sync_sets_[ready_sync_set_index_].FillInputSet(input_timestamp, input_set);
  for (int i = 0; i < sync_sets_.size(); ++i) {
    if (i != ready_sync_set_index_) {
      sync_sets_[i].FillInputBounds(input_set);
    }
  }
  ready_sync_set_index_ = -1;
  ready_timestamp_ = Timestamp::Done();
}

int SyncSetInputStreamHandler::SyncSetCount() {
  absl::MutexLock lock(&mutex_);
  return sync_sets_.size();
}

}  // namespace mediapipe
