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

#include "mediapipe/util/tracking/streaming_buffer.h"

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"

namespace mediapipe {

StreamingBuffer::StreamingBuffer(
    const std::vector<TaggedType>& data_configuration, int overlap)
    : overlap_(overlap) {
  ABSL_CHECK_GE(overlap, 0);
  for (auto& item : data_configuration) {
    ABSL_CHECK(data_config_.find(item.first) == data_config_.end())
        << "Tag " << item.first << " already exists";
    data_config_[item.first] = item.second;
    // Init deque.
    data_[item.first].clear();
  }
}

bool StreamingBuffer::HasTag(const std::string& tag) const {
  return data_config_.find(tag) != data_config_.end();
}

bool StreamingBuffer::HasTags(const std::vector<std::string>& tags) const {
  for (const auto& tag : tags) {
    if (!HasTag(tag)) {
      return false;
    }
  }
  return true;
}

int StreamingBuffer::BufferSize(const std::string& tag) const {
  ABSL_CHECK(HasTag(tag));
  return data_.find(tag)->second.size();
}

int StreamingBuffer::MaxBufferSize() const {
  int max_buffer = 0;
  for (const auto& elem : data_) {
    max_buffer = std::max(max_buffer, BufferSize(elem.first));
  }
  return max_buffer;
}

bool StreamingBuffer::HaveEqualSize(
    const std::vector<std::string>& tags) const {
  if (tags.size() < 2) {
    return true;
  }
  int first_size = BufferSize(tags[0]);
  for (int k = 1; k < tags.size(); ++k) {
    if (BufferSize(tags[1]) != first_size) {
      return false;
    }
  }
  return true;
}

std::vector<std::string> StreamingBuffer::AllTags() const {
  std::vector<std::string> all_tags;
  for (auto& item : data_config_) {
    all_tags.push_back(item.first);
  }
  return all_tags;
}

bool StreamingBuffer::TruncateBuffer(bool flush) {
  // Only truncate if sufficient elements have been buffered.
  const int elems_to_clear =
      std::max(0, MaxBufferSize() - (flush ? 0 : overlap_));

  if (elems_to_clear == 0) {
    return true;
  }

  bool is_consistent = true;
  for (auto& item : data_) {
    auto& buffer = item.second;
    const int buffer_elems_to_clear =
        std::min<int>(elems_to_clear, buffer.size());
    if (buffer_elems_to_clear < elems_to_clear) {
      ABSL_LOG(WARNING) << "For tag " << item.first << " got "
                        << elems_to_clear - buffer_elems_to_clear
                        << "fewer elements than buffer can hold.";
      is_consistent = false;
    }
    buffer.erase(buffer.begin(), buffer.begin() + buffer_elems_to_clear);
  }

  first_frame_index_ += elems_to_clear;

  const int remaining_elems = flush ? 0 : overlap_;
  for (const auto& item : data_) {
    const auto& buffer = item.second;
    if (buffer.size() != remaining_elems) {
      ABSL_LOG(WARNING) << "After trunctation, for tag " << item.first << "got "
                        << buffer.size() << " elements, " << "expected "
                        << remaining_elems;
      is_consistent = false;
    }
  }

  return is_consistent;
}

void StreamingBuffer::DiscardDatum(const std::string& tag, int num_frames) {
  ABSL_CHECK(HasTag(tag));
  auto& queue = data_[tag];
  if (queue.empty()) {
    return;
  }
  queue.erase(queue.begin(),
              queue.begin() + std::min<int>(queue.size(), num_frames));
}

void StreamingBuffer::DiscardDatumFromEnd(const std::string& tag,
                                          int num_frames) {
  ABSL_CHECK(HasTag(tag));
  auto& queue = data_[tag];
  if (queue.empty()) {
    return;
  }
  queue.erase(queue.end() - std::min<int>(queue.size(), num_frames),
              queue.end());
}

void StreamingBuffer::DiscardData(const std::vector<std::string>& tags,
                                  int num_frames) {
  for (const std::string& tag : tags) {
    DiscardDatum(tag, num_frames);
  }
}

}  // namespace mediapipe
