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

#include "mediapipe/framework/scheduler_queue.h"

#include <memory>
#include <queue>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/calculator_node.h"
#include "mediapipe/framework/executor.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/status.h"

#ifdef __APPLE__
#define AUTORELEASEPOOL @autoreleasepool
#else
#define AUTORELEASEPOOL
#endif  // __APPLE__

namespace mediapipe {
namespace internal {

SchedulerQueue::Item::Item(CalculatorNode* node, CalculatorContext* cc)
    : node_(node), cc_(cc) {
  ABSL_CHECK(node);
  ABSL_CHECK(cc);
  is_source_ = node->IsSource();
  id_ = node->Id();
  if (is_source_) {
    layer_ = node->source_layer();
    source_process_order_ = node->SourceProcessOrder(cc).Value();
  }
}

SchedulerQueue::Item::Item(CalculatorNode* node)
    : node_(node), cc_(nullptr), is_open_node_(true) {
  ABSL_CHECK(node);
  is_source_ = node->IsSource();
  id_ = node->Id();
  if (is_source_) {
    layer_ = node->source_layer();
    source_process_order_ = Timestamp::Unstarted().Value();
  }
}

// Returning true means "this runs after that".
bool SchedulerQueue::Item::operator<(const SchedulerQueue::Item& that) const {
  if (is_open_node_ || that.is_open_node_) {
    // OpenNode() runs before ProcessNode().
    if (!that.is_open_node_) return false;
    // ProcessNode() runs after OpenNode().
    if (!is_open_node_) return true;
    // If both are OpenNode(), higher ids run after lower ids.
    return id_ > that.id_;
  }
  if (is_source_) {
    // Sources run after non-sources.
    if (!that.is_source_) return true;
    // Higher layer sources run after lower layer sources.
    if (layer_ != that.layer_) return layer_ > that.layer_;
    // Higher SourceProcessOrder values run after lower values.
    if (source_process_order_ != that.source_process_order_) {
      return source_process_order_ > that.source_process_order_;
    }
    // For sources, higher ids run after lower ids.
    return id_ > that.id_;
  } else {
    // Non-sources run before sources.
    if (that.is_source_) return false;
    // For non-sources, higher ids run before lower ids.
    return id_ < that.id_;
  }
}

void SchedulerQueue::Reset() {
  absl::MutexLock lock(&mutex_);
  num_pending_tasks_ = 0;
  num_tasks_to_add_ = 0;
  running_count_ = 0;
}

void SchedulerQueue::SetExecutor(Executor* executor) { executor_ = executor; }

bool SchedulerQueue::IsIdle() {
  VLOG(3) << "Scheduler queue empty: " << queue_.empty()
          << ", # of pending tasks: " << num_pending_tasks_;
  return queue_.empty() && num_pending_tasks_ == 0;
}

void SchedulerQueue::SetRunning(bool running) {
  absl::MutexLock lock(&mutex_);
  running_count_ += running ? 1 : -1;
  ABSL_DCHECK_LE(running_count_, 1);
}

void SchedulerQueue::AddNode(CalculatorNode* node, CalculatorContext* cc) {
  // TODO: If the node isn't successfully scheduled, we must properly
  // handle the pending calculator context.
  if (shared_->has_error) {
    return;
  }
  if (!node->TryToBeginScheduling()) {
    // Only happens when the framework tries to schedule an unthrottled source
    // node while it's running. For non-source nodes, if a calculator context is
    // prepared, it is committed to be scheduled.
    ABSL_CHECK(node->IsSource()) << node->DebugName();
    return;
  }
  AddItemToQueue(Item(node, cc));
}

void SchedulerQueue::AddNodeForOpen(CalculatorNode* node) {
  if (shared_->has_error) {
    return;
  }
  AddItemToQueue(Item(node));
}

void SchedulerQueue::AddItemToQueue(Item&& item) {
  const CalculatorNode* node = item.Node();
  bool was_idle;
  int tasks_to_add = 0;
  {
    absl::MutexLock lock(&mutex_);
    was_idle = IsIdle();
    queue_.push(item);
    ++num_tasks_to_add_;
    VLOG(4) << node->DebugName() << " was added to the scheduler queue.";

    // Now grab the tasks to execute while still holding the lock. This will
    // gather any waiting tasks, in addition to the one we just added.
    if (running_count_ > 0) {
      tasks_to_add = GetTasksToSubmitToExecutor();
    }
  }
  if (was_idle && idle_callback_) {
    // Became not idle.
    idle_callback_(false);
  }
  // Note: this should be done after calling idle_callback_(false) above.
  // This ensures that we never get an idle_callback_(true) that is not
  // preceded by the corresponding idle_callback_(false). See the comments on
  // SetIdleCallback for details.
  while (tasks_to_add > 0) {
    executor_->AddTask(this);
    --tasks_to_add;
  }
}

int SchedulerQueue::GetTasksToSubmitToExecutor() {
  int tasks_to_add = num_tasks_to_add_;
  num_tasks_to_add_ = 0;
  num_pending_tasks_ += tasks_to_add;
  return tasks_to_add;
}

void SchedulerQueue::SubmitWaitingTasksToExecutor() {
  // If a node is added to the scheduler queue while the queue is not running,
  // we do not immediately submit tasks to the executor. Here we check for any
  // such waiting tasks, and submit them.
  int tasks_to_add = 0;
  {
    absl::MutexLock lock(&mutex_);
    if (running_count_ > 0) {
      tasks_to_add = GetTasksToSubmitToExecutor();
    }
  }
  while (tasks_to_add > 0) {
    executor_->AddTask(this);
    --tasks_to_add;
  }
}

void SchedulerQueue::RunNextTask() {
  CalculatorNode* node;
  CalculatorContext* calculator_context;
  bool is_open_node;
  {
    absl::MutexLock lock(&mutex_);

    ABSL_CHECK(!queue_.empty())
        << "Called RunNextTask when the queue is empty. "
           "This should not happen.";

    node = queue_.top().Node();
    calculator_context = queue_.top().Context();
    is_open_node = queue_.top().IsOpenNode();
    queue_.pop();

    ABSL_CHECK(!node->Closed())
        << "Scheduled a node that was closed. This should not happen.";
  }

  // On iOS, calculators may rely on the existence of an autorelease pool
  // (either directly, or because system code they call does). We do not
  // want to rely on executors setting up an autorelease pool for us (e.g.
  // an executor creating standard pthread will not, by default), so we
  // do it here to ensure all executors are covered.
  AUTORELEASEPOOL {
    if (is_open_node) {
      ABSL_DCHECK(!calculator_context);
      OpenCalculatorNode(node);
    } else {
      RunCalculatorNode(node, calculator_context);
    }
  }

  bool is_idle;
  {
    absl::MutexLock lock(&mutex_);
    ABSL_DCHECK_GT(num_pending_tasks_, 0);
    --num_pending_tasks_;
    is_idle = IsIdle();
  }
  if (is_idle && idle_callback_) {
    // Became idle.
    idle_callback_(true);
  }
}

void SchedulerQueue::RunCalculatorNode(CalculatorNode* node,
                                       CalculatorContext* cc) {
  VLOG(3) << "Running " << node->DebugName();

  // If we are in the process of stopping the graph (due to tool::StatusStop()
  // from a non-source node or due to CalculatorGraph::CloseAllPacketSources),
  // we should not run any more sources.  Close the node if it is a source.
  if (shared_->stopping && node->IsSource()) {
    VLOG(4) << "Closing " << node->DebugName() << " due to StatusStop().";
    int64_t start_time = shared_->timer.StartNode();
    // It's OK to not reset/release the prepared CalculatorContext since a
    // source node always reuses the same CalculatorContext and Close() doesn't
    // access any inputs.
    // TODO: Should we pass tool::StatusStop() in this case?
    const absl::Status result =
        node->CloseNode(absl::OkStatus(), /*graph_run_ended=*/false);
    shared_->timer.EndNode(start_time);
    if (!result.ok()) {
      VLOG(3) << node->DebugName()
              << " had an error while closing due to StatusStop()!";
      shared_->error_callback(result);
    }
  } else {
    // Note that we don't need a lock because only one thread can execute this
    // due to the lock on running_nodes.
    int64_t start_time = shared_->timer.StartNode();
    const absl::Status result = node->ProcessNode(cc);
    shared_->timer.EndNode(start_time);

    if (!result.ok()) {
      if (result == tool::StatusStop()) {
        // Check if StatusStop was returned by a non-source node. This means
        // that all sources will be closed and no further sources should be
        // scheduled. The graph will be terminated as soon as its scheduler
        // queue becomes empty.
        ABSL_CHECK(!node->IsSource());  // ProcessNode takes care of
                                        // StatusStop() from sources.
        shared_->stopping = true;
      } else {
        // If we have an error in this calculator.
        VLOG(3) << node->DebugName() << " had an error!";
        shared_->error_callback(result);
      }
    }
  }

  VLOG(4) << "Done running " << node->DebugName();
  node->EndScheduling();
}

void SchedulerQueue::OpenCalculatorNode(CalculatorNode* node) {
  VLOG(3) << "Opening " << node->DebugName();
  int64_t start_time = shared_->timer.StartNode();
  const absl::Status result = node->OpenNode();
  shared_->timer.EndNode(start_time);
  if (!result.ok()) {
    VLOG(3) << node->DebugName() << " had an error!";
    shared_->error_callback(result);
    return;
  }
  node->NodeOpened();
}

void SchedulerQueue::CleanupAfterRun() {
  bool was_idle;
  {
    absl::MutexLock lock(&mutex_);
    was_idle = IsIdle();
    ABSL_CHECK_EQ(num_pending_tasks_, 0);
    ABSL_CHECK_EQ(num_tasks_to_add_, queue_.size());
    num_tasks_to_add_ = 0;
    while (!queue_.empty()) {
      queue_.pop();
    }
  }
  if (!was_idle && idle_callback_) {
    // Became idle.
    idle_callback_(true);
  }
}

}  // namespace internal
}  // namespace mediapipe
