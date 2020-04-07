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

#include "mediapipe/framework/stream_handler/default_input_stream_handler.h"

#include <algorithm>

#include "absl/strings/substitute.h"
#include "mediapipe/framework/input_stream_handler.h"

namespace mediapipe {

REGISTER_INPUT_STREAM_HANDLER(DefaultInputStreamHandler);

// Returns all CollectionItemId's for a Collection TagMap.
std::vector<CollectionItemId> GetIds(
    const std::shared_ptr<tool::TagMap>& tag_map) {
  std::vector<CollectionItemId> result;
  for (auto id = tag_map->BeginId(); id < tag_map->EndId(); ++id) {
    result.push_back(id);
  }
  return result;
}

DefaultInputStreamHandler::DefaultInputStreamHandler(
    std::shared_ptr<tool::TagMap> tag_map, CalculatorContextManager* cc_manager,
    const MediaPipeOptions& options, bool calculator_run_in_parallel)
    : InputStreamHandler(std::move(tag_map), cc_manager, options,
                         calculator_run_in_parallel),
      sync_set_(this, GetIds(input_stream_managers_.TagMap())) {
  if (options.HasExtension(DefaultInputStreamHandlerOptions::ext)) {
    SetBatchSize(options.GetExtension(DefaultInputStreamHandlerOptions::ext)
                     .batch_size());
  }
}

void DefaultInputStreamHandler::PrepareForRun(
    std::function<void()> headers_ready_callback,
    std::function<void()> notification_callback,
    std::function<void(CalculatorContext*)> schedule_callback,
    std::function<void(::mediapipe::Status)> error_callback) {
  sync_set_.PrepareForRun();
  InputStreamHandler::PrepareForRun(
      std::move(headers_ready_callback), std::move(notification_callback),
      std::move(schedule_callback), std::move(error_callback));
}

NodeReadiness DefaultInputStreamHandler::GetNodeReadiness(
    Timestamp* min_stream_timestamp) {
  return sync_set_.GetReadiness(min_stream_timestamp);
}

void DefaultInputStreamHandler::FillInputSet(Timestamp input_timestamp,
                                             InputStreamShardSet* input_set) {
  sync_set_.FillInputSet(input_timestamp, input_set);
}

}  // namespace mediapipe
