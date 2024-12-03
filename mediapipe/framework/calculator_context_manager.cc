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

#include "mediapipe/framework/calculator_context_manager.h"

#include <utility>

#include "absl/log/absl_check.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/port/logging.h"

namespace mediapipe {

void CalculatorContextManager::Initialize(
    CalculatorState* calculator_state,
    std::shared_ptr<tool::TagMap> input_tag_map,
    std::shared_ptr<tool::TagMap> output_tag_map,
    bool calculator_run_in_parallel) {
  ABSL_CHECK(calculator_state);
  calculator_state_ = calculator_state;
  input_tag_map_ = std::move(input_tag_map);
  output_tag_map_ = std::move(output_tag_map);
  calculator_run_in_parallel_ = calculator_run_in_parallel;
}

absl::Status CalculatorContextManager::PrepareForRun(
    std::function<absl::Status(CalculatorContext*)> setup_shards_callback) {
  setup_shards_callback_ = std::move(setup_shards_callback);
  default_context_ = absl::make_unique<CalculatorContext>(
      calculator_state_, input_tag_map_, output_tag_map_);
  return setup_shards_callback_(default_context_.get());
}

void CalculatorContextManager::CleanupAfterRun() {
  default_context_ = nullptr;
  absl::MutexLock lock(&contexts_mutex_);
  active_contexts_.clear();
  idle_contexts_.clear();
}

CalculatorContext* CalculatorContextManager::GetDefaultCalculatorContext()
    const {
  ABSL_CHECK(default_context_.get());
  return default_context_.get();
}

CalculatorContext* CalculatorContextManager::GetFrontCalculatorContext(
    Timestamp* context_input_timestamp) {
  ABSL_CHECK(calculator_run_in_parallel_);
  absl::MutexLock lock(&contexts_mutex_);
  ABSL_CHECK(!active_contexts_.empty());
  *context_input_timestamp = active_contexts_.begin()->first;
  return active_contexts_.begin()->second.get();
}

CalculatorContext* CalculatorContextManager::PrepareCalculatorContext(
    Timestamp input_timestamp) {
  if (!calculator_run_in_parallel_) {
    return GetDefaultCalculatorContext();
  }
  absl::MutexLock lock(&contexts_mutex_);
  ABSL_CHECK(!mediapipe::ContainsKey(active_contexts_, input_timestamp))
      << "Multiple invocations with the same timestamps are not allowed with "
         "parallel execution, input_timestamp = "
      << input_timestamp;
  CalculatorContext* calculator_context = nullptr;
  if (idle_contexts_.empty()) {
    auto new_context = absl::make_unique<CalculatorContext>(
        calculator_state_, input_tag_map_, output_tag_map_);
    MEDIAPIPE_CHECK_OK(setup_shards_callback_(new_context.get()));
    calculator_context = new_context.get();
    active_contexts_.emplace(input_timestamp, std::move(new_context));
  } else {
    // Retrieves an inactive calculator context from idle_contexts_.
    calculator_context = idle_contexts_.front().get();
    active_contexts_.emplace(input_timestamp,
                             std::move(idle_contexts_.front()));
    idle_contexts_.pop_front();
  }
  return calculator_context;
}

void CalculatorContextManager::RecycleCalculatorContext() {
  absl::MutexLock lock(&contexts_mutex_);
  // The first element in active_contexts_ will be recycled.
  auto iter = active_contexts_.begin();
  idle_contexts_.push_back(std::move(iter->second));
  active_contexts_.erase(iter);
}

bool CalculatorContextManager::HasActiveContexts() {
  if (!calculator_run_in_parallel_) {
    return false;
  }
  absl::MutexLock lock(&contexts_mutex_);
  return !active_contexts_.empty();
}

}  // namespace mediapipe
