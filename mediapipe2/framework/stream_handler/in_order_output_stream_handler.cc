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

#include "mediapipe/framework/stream_handler/in_order_output_stream_handler.h"

#include "mediapipe/framework/collection.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/output_stream_shard.h"

namespace mediapipe {

REGISTER_OUTPUT_STREAM_HANDLER(InOrderOutputStreamHandler);

void InOrderOutputStreamHandler::PropagationLoop() {
  CHECK_EQ(propagation_state_, kIdle);
  Timestamp context_timestamp;
  CalculatorContext* calculator_context;
  if (!calculator_context_manager_->HasActiveContexts()) {
    propagation_state_ = kPropagatingBound;
  } else {
    calculator_context = calculator_context_manager_->GetFrontCalculatorContext(
        &context_timestamp);
    if (!completed_input_timestamps_.empty()) {
      Timestamp completed_timestamp = *completed_input_timestamps_.begin();
      if (context_timestamp != completed_timestamp) {
        CHECK_LT(context_timestamp, completed_timestamp);
        return;
      }
      propagation_state_ = kPropagatingPackets;
    }
  }

  while (propagation_state_ != kIdle) {
    if (propagation_state_ == kPropagatingPackets) {
      PropagatePackets(&calculator_context, &context_timestamp);
    } else {
      CHECK_EQ(kPropagatingBound, propagation_state_);
      PropagationBound(&calculator_context, &context_timestamp);
    }
  }
}

void InOrderOutputStreamHandler::PropagatePackets(
    CalculatorContext** calculator_context, Timestamp* context_timestamp) {
  timestamp_mutex_.Unlock();
  // Propagates packets without holding timestamp_mutex_.
  PropagateOutputPackets(*context_timestamp, &(*calculator_context)->Outputs());
  calculator_context_manager_->RecycleCalculatorContext();
  timestamp_mutex_.Lock();
  completed_input_timestamps_.erase(completed_input_timestamps_.begin());
  // The first check is for performance reasons (it's cheaper).
  // Note that completed_input_timestamps_ is a subset of the input
  // timestamps of the active contexts. Therefore, the second check
  // covers the first check.
  if (completed_input_timestamps_.empty() &&
      !calculator_context_manager_->HasActiveContexts()) {
    // If task_timestamp_bound_ is not greater than context_timestamp + 1,
    // timestamp propagation isn't necessary since the bound of the
    // downstream input streams has been updated to a larger value
    // already. Timestamp propagation will be skipped, and the
    // propagation process is completed.
    if (task_timestamp_bound_ <= context_timestamp->NextAllowedInStream()) {
      propagation_state_ = kIdle;
      return;
    }
    propagation_state_ = kPropagatingBound;
    return;
  }
  *calculator_context =
      calculator_context_manager_->GetFrontCalculatorContext(context_timestamp);
  if (!completed_input_timestamps_.empty() &&
      *context_timestamp == *completed_input_timestamps_.begin()) {
    // Continues propagating output packets if the smallest completed
    // input timestamp is equal to the input timestamp of the earliest
    // active calculator context.
    return;
  }
  propagation_state_ = kIdle;
}

void InOrderOutputStreamHandler::PropagationBound(
    CalculatorContext** calculator_context, Timestamp* context_timestamp) {
  Timestamp bound_to_propagate = task_timestamp_bound_;
  timestamp_mutex_.Unlock();
  // Timestamp bound propagation without holding timestamp_mutex_.
  TryPropagateTimestampBound(bound_to_propagate);
  timestamp_mutex_.Lock();
  if (propagation_state_ == kPropagatingBound) {
    // There is no invocation completed and no newly arrived timestamp
    // bound during the timestamp bound propagation. So the propagation
    // process is completed.
    propagation_state_ = kIdle;
    return;
  }
  // Some recent changes require the propagation thread to recheck if any
  // new packets can be propagated.
  CHECK_EQ(propagation_state_, kPropagationPending);
  // task_timestamp_bound_ was updated while the propagation thread was
  // doing timestamp propagation. This thread will redo timestamp
  // propagation for the new task_timestamp_bound_.
  if (!calculator_context_manager_->HasActiveContexts()) {
    CHECK_LT(bound_to_propagate, task_timestamp_bound_);
    propagation_state_ = kPropagatingBound;
    return;
  }
  *calculator_context =
      calculator_context_manager_->GetFrontCalculatorContext(context_timestamp);
  if (completed_input_timestamps_.empty() ||
      *context_timestamp != *completed_input_timestamps_.begin()) {
    // If there is no newly completed invocation or the newly arrived packets
    // are not ready for propagation, the propagation process is completed.
    propagation_state_ = kIdle;
    return;
  } else {
    // Found new packets to be propagated, and will redo packets
    // propagation.
    propagation_state_ = kPropagatingPackets;
    return;
  }
}

}  // namespace mediapipe
