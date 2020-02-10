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

#include "mediapipe/framework/scheduler.h"

#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/executor.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/tool/status_util.h"

namespace mediapipe {

namespace internal {

Scheduler::Scheduler(CalculatorGraph* graph)
    : graph_(graph), shared_(), default_queue_(&shared_) {
  shared_.error_callback =
      std::bind(&CalculatorGraph::RecordError, graph_, std::placeholders::_1);
  default_queue_.SetIdleCallback(std::bind(&Scheduler::QueueIdleStateChanged,
                                           this, std::placeholders::_1));
  scheduler_queues_.push_back(&default_queue_);
}

Scheduler::~Scheduler() {
  {
    absl::MutexLock lock(&state_mutex_);
    if (state_ == STATE_NOT_STARTED) {
      return;
    }
  }
  // If the application does not call WaitUntilDone, we should.
  // WaitUntilDone ensures that all queues are done and will no longer access
  // the scheduler.
  Cancel();
  WaitUntilDone().IgnoreError();
}

void Scheduler::Reset() {
  {
    absl::MutexLock lock(&state_mutex_);
    state_ = STATE_NOT_STARTED;
    graph_input_streams_closed_ = graph_->GraphInputStreamsClosed();
    throttled_graph_input_stream_count_ = 0;
    unthrottle_seq_num_ = 0;
    observed_output_signal_ = false;
  }
  for (auto queue : scheduler_queues_) {
    queue->Reset();
  }
  shared_.stopping = false;
  shared_.has_error = false;
}

void Scheduler::CloseAllSourceNodes() { shared_.stopping = true; }

void Scheduler::SetExecutor(Executor* executor) {
  CHECK_EQ(state_, STATE_NOT_STARTED)
      << "SetExecutor must not be called after the scheduler has started";
  default_queue_.SetExecutor(executor);
}

// TODO: Consider renaming this method CreateNonDefaultQueue.
::mediapipe::Status Scheduler::SetNonDefaultExecutor(const std::string& name,
                                                     Executor* executor) {
  RET_CHECK_EQ(state_, STATE_NOT_STARTED) << "SetNonDefaultExecutor must not "
                                             "be called after the scheduler "
                                             "has started";
  auto inserted = non_default_queues_.emplace(
      name, absl::make_unique<SchedulerQueue>(&shared_));
  RET_CHECK(inserted.second)
      << "SetNonDefaultExecutor must be called only once for the executor \""
      << name << "\"";

  SchedulerQueue* queue = inserted.first->second.get();
  queue->SetIdleCallback(std::bind(&Scheduler::QueueIdleStateChanged, this,
                                   std::placeholders::_1));
  queue->SetExecutor(executor);
  scheduler_queues_.push_back(queue);
  return ::mediapipe::OkStatus();
}

void Scheduler::SetQueuesRunning(bool running) {
  for (auto queue : scheduler_queues_) {
    queue->SetRunning(running);
  }
}

void Scheduler::SubmitWaitingTasksOnQueues() {
  for (auto queue : scheduler_queues_) {
    queue->SubmitWaitingTasksToExecutor();
  }
}

// Note: state_mutex_ is held when this function is entered or
// exited.
void Scheduler::HandleIdle() {
  if (handling_idle_) {
    // Someone is already inside this method.
    // Note: This can happen in the sections below where we unlock the mutex
    // and make more nodes runnable: the nodes can run and become idle again
    // while this method is in progress. In that case, the resulting calls to
    // HandleIdle are ignored, which is ok because the original method will
    // run the loop again.
    VLOG(2) << "HandleIdle: already in progress";
    return;
  }
  handling_idle_ = true;

  while (IsIdle() && (state_ == STATE_RUNNING || state_ == STATE_CANCELLING)) {
    // Remove active sources that are closed.
    CleanupActiveSources();

    // Quit if we have errors, or if there are no more packet sources.
    if (shared_.has_error ||
        (active_sources_.empty() && sources_queue_.empty() &&
         graph_input_streams_closed_)) {
      VLOG(2) << "HandleIdle: quitting";
      Quit();
      break;
    }

    // See if we can schedule the next layer of source nodes.
    if (active_sources_.empty() && !sources_queue_.empty()) {
      VLOG(2) << "HandleIdle: activating sources";
      // Note: TryToScheduleNextSourceLayer unlocks and locks state_mutex_
      // internally.
      bool did_activate = TryToScheduleNextSourceLayer();
      CHECK(did_activate || active_sources_.empty());
      continue;
    }

    // See if we can unthrottle some source nodes or graph input streams to
    // break deadlock. If we are still idle and there are active source nodes,
    // they must be throttled.
    if (!active_sources_.empty() || throttled_graph_input_stream_count_ > 0) {
      VLOG(2) << "HandleIdle: unthrottling";
      state_mutex_.Unlock();
      bool did_unthrottle = graph_->UnthrottleSources();
      state_mutex_.Lock();
      if (did_unthrottle) {
        continue;
      }
    }

    // Nothing left to do.
    break;
  }

  handling_idle_ = false;
}

// Note: state_mutex_ is held when this function is entered or exited.
// Once this function returns, the scheduler may be destructed as soon as
// state_mutex_ is unlocked.
void Scheduler::Quit() {
  // All calls to Calculator::Process() have returned (even if we had an
  // error).
  CHECK(state_ == STATE_RUNNING || state_ == STATE_CANCELLING);
  SetQueuesRunning(false);
  shared_.timer.EndRun();

  VLOG(2) << "Signaling scheduler termination";
  // Let other threads know that scheduler terminated.
  state_ = STATE_TERMINATED;
  state_cond_var_.SignalAll();
}

void Scheduler::Start() {
  VLOG(2) << "Starting scheduler";
  shared_.timer.StartRun();
  {
    absl::MutexLock lock(&state_mutex_);
    CHECK_EQ(state_, STATE_NOT_STARTED);
    state_ = STATE_RUNNING;
    SetQueuesRunning(true);

    // Get the ball rolling.
    HandleIdle();
  }
  SubmitWaitingTasksOnQueues();
}

void Scheduler::AddApplicationThreadTask(std::function<void()> task) {
  absl::MutexLock lock(&state_mutex_);
  app_thread_tasks_.push_back(std::move(task));
  if (app_thread_tasks_.size() == 1) {
    state_cond_var_.SignalAll();
  }
}

void Scheduler::ThrottledGraphInputStream() {
  absl::MutexLock lock(&state_mutex_);
  ++throttled_graph_input_stream_count_;
}

void Scheduler::UnthrottledGraphInputStream() {
  absl::MutexLock lock(&state_mutex_);
  --throttled_graph_input_stream_count_;
  ++unthrottle_seq_num_;
  state_cond_var_.SignalAll();
}

void Scheduler::WaitUntilGraphInputStreamUnthrottled(
    absl::Mutex* secondary_mutex) {
  // Since we want to support multiple concurrent calls to this method, we
  // cannot use a simple boolean flag like in WaitForObservedOutput: when one
  // invocation sees and erases the flag, it would make it invisible to the
  // others. Instead, we use a sequence number. Each call records the current
  // sequence number before unlocking. If an unthrottle event occurred after
  // that point, the sequence number will differ.
  int seq_num;
  {
    absl::MutexLock lock(&state_mutex_);
    seq_num = unthrottle_seq_num_;
  }
  secondary_mutex->Unlock();
  ApplicationThreadAwait(
      [this, seq_num]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mutex_) {
        return (unthrottle_seq_num_ != seq_num) || state_ == STATE_TERMINATED;
      });
  secondary_mutex->Lock();
}

void Scheduler::EmittedObservedOutput() {
  absl::MutexLock lock(&state_mutex_);
  observed_output_signal_ = true;
  if (waiting_for_observed_output_) {
    state_cond_var_.SignalAll();
  }
}

::mediapipe::Status Scheduler::WaitForObservedOutput() {
  bool observed = false;
  ApplicationThreadAwait(
      [this, &observed]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mutex_) {
        observed = observed_output_signal_;
        observed_output_signal_ = false;
        waiting_for_observed_output_ = !observed && state_ != STATE_TERMINATED;
        // Wait until the member waiting_for_observed_output_ becomes false.
        return !waiting_for_observed_output_;
      });
  return observed ? ::mediapipe::OkStatus()
                  : ::mediapipe::OutOfRangeError("Graph is done.");
}

// Idleness requires:
// 1. either the graph has no source nodes or all source nodes are closed, and
// 2. no packets are added to graph input streams.
// For simplicity, we only allow WaitUntilIdle() to be called on a graph with
// no source nodes. (This is enforced by CalculatorGraph::WaitUntilIdle().)
// The application must ensure no other threads are adding packets to graph
// input streams while a WaitUntilIdle() call is in progress.
::mediapipe::Status Scheduler::WaitUntilIdle() {
  RET_CHECK_NE(state_, STATE_NOT_STARTED);
  ApplicationThreadAwait(std::bind(&Scheduler::IsIdle, this));
  return ::mediapipe::OkStatus();
}

::mediapipe::Status Scheduler::WaitUntilDone() {
  RET_CHECK_NE(state_, STATE_NOT_STARTED);
  ApplicationThreadAwait([this]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mutex_) {
    return state_ == STATE_TERMINATED;
  });
  return ::mediapipe::OkStatus();
}

void Scheduler::ApplicationThreadAwait(
    const std::function<bool()>& stop_condition) {
  absl::MutexLock lock(&state_mutex_);
  while (!stop_condition()) {
    if (app_thread_tasks_.empty()) {
      state_cond_var_.Wait(&state_mutex_);
    } else {
      std::function<void()> task = std::move(app_thread_tasks_.front());
      app_thread_tasks_.pop_front();
      state_mutex_.Unlock();
      task();
      state_mutex_.Lock();
    }
  }
}

bool Scheduler::IsIdle() { return non_idle_queue_count_ == 0; }

void Scheduler::AddedPacketToGraphInputStream() {
  if (state_ == STATE_TERMINATED) {
    return;
  }
  absl::MutexLock lock(&state_mutex_);
  // It seems that the only thing it really needs to do is to check if more
  // unthrottling needs to be done.
  HandleIdle();
}

// Note: This may be called while we are already in STATE_TERMINATED.
void Scheduler::ClosedAllGraphInputStreams() {
  absl::MutexLock lock(&state_mutex_);
  graph_input_streams_closed_ = true;
  // This is called to check whether we should quit.
  HandleIdle();
}

// TODO: If the node isn't successfully scheduled, we must properly
// handle the pending calculator context. For example, the caller should dispose
// of the calculator context and put it into a pending calculator context
// container.
void Scheduler::ScheduleNodeIfNotThrottled(
    CalculatorNode* node, CalculatorContext* calculator_context) {
  DCHECK(node);
  DCHECK(calculator_context);
  if (!graph_->IsNodeThrottled(node->Id())) {
    node->GetSchedulerQueue()->AddNode(node, calculator_context);
  }
}

void Scheduler::ScheduleNodeForOpen(CalculatorNode* node) {
  DCHECK(node);
  VLOG(1) << "Scheduling OpenNode of calculator " << node->DebugName();
  node->GetSchedulerQueue()->AddNodeForOpen(node);
}

void Scheduler::ScheduleUnthrottledReadyNodes(
    const std::vector<CalculatorNode*>& nodes_to_schedule) {
  for (CalculatorNode* node : nodes_to_schedule) {
    // Source nodes always reuse the default calculator context because they
    // can't be executed in parallel.
    CHECK(node->IsSource());
    CalculatorContext* default_context = node->GetDefaultCalculatorContext();
    node->GetSchedulerQueue()->AddNode(node, default_context);
  }
}

void Scheduler::CleanupActiveSources() {
  // Remove sources from the back of the active sources vector if they have
  // been closed. We only remove from the back because it is cheap to remove
  // elements at the end of a std::vector.
  while (!active_sources_.empty()) {
    CalculatorNode* active_source = active_sources_.back();
    if (active_source->Closed()) {
      active_sources_.pop_back();
    } else {
      break;
    }
  }
}

bool Scheduler::TryToScheduleNextSourceLayer() {
  VLOG(3) << "TryToScheduleNextSourceLayer";

  CHECK(active_sources_.empty());
  CHECK(!sources_queue_.empty());

  if (!unopened_sources_.empty() &&
      (*unopened_sources_.begin())->source_layer() <
          sources_queue_.top().Node()->source_layer()) {
    // If no graph input streams are open, then there are no packet sources in
    // the graph. It's a deadlock.
    if (graph_input_streams_closed_) {
      graph_->RecordError(::mediapipe::UnknownError(
          "Detected a deadlock because source nodes cannot be activated when a "
          "source node at a lower layer is still not opened."));
    }
    return false;
  }

  // contexts[i] stores the CalculatorContext to be used with
  // active_sources_[i].
  std::vector<CalculatorContext*> contexts;
  bool activated_any = false;
  while (!sources_queue_.empty()) {
    CalculatorNode* node = sources_queue_.top().Node();
    // Only add sources with the same layer number.
    if (activated_any &&
        (node->source_layer() != active_sources_.back()->source_layer())) {
      break;
    }
    active_sources_.emplace_back(node);
    contexts.emplace_back(sources_queue_.top().Context());
    activated_any = true;
    sources_queue_.pop();
  }
  if (!activated_any) {
    return false;
  }

  state_mutex_.Unlock();
  // Add all the sources in a layer to the scheduler queue at once to
  // guarantee they are scheduled in a round-robin fashion. Pause the
  // scheduler queue until all the sources have been added.
  SetQueuesRunning(false);
  for (int i = 0; i < active_sources_.size(); ++i) {
    CalculatorNode* node = active_sources_[i];
    node->ActivateNode();
    ScheduleNodeIfNotThrottled(node, contexts[i]);
  }
  SetQueuesRunning(true);
  SubmitWaitingTasksOnQueues();
  state_mutex_.Lock();
  return true;
}

void Scheduler::AddUnopenedSourceNode(CalculatorNode* node) {
  CHECK_EQ(state_, STATE_NOT_STARTED) << "AddUnopenedSourceNode can only be "
                                         "called before starting the scheduler";
  unopened_sources_.insert(node);
}

void Scheduler::AddNodeToSourcesQueue(CalculatorNode* node) {
  // Source nodes always reuse the default calculator context because they
  // can't be executed in parallel.
  CalculatorContext* default_context = node->GetDefaultCalculatorContext();
  absl::MutexLock lock(&state_mutex_);
  sources_queue_.push(SchedulerQueue::Item(node, default_context));
  unopened_sources_.erase(node);
}

void Scheduler::AssignNodeToSchedulerQueue(CalculatorNode* node) {
  SchedulerQueue* queue;
  if (!node->Executor().empty()) {
    auto iter = non_default_queues_.find(node->Executor());
    CHECK(iter != non_default_queues_.end());
    queue = iter->second.get();
  } else {
    queue = &default_queue_;
  }
  node->SetSchedulerQueue(queue);
}

void Scheduler::QueueIdleStateChanged(bool idle) {
  absl::MutexLock lock(&state_mutex_);
  non_idle_queue_count_ += (idle ? -1 : 1);
  VLOG(2) << "active queues: " << non_idle_queue_count_;
  if (non_idle_queue_count_ == 0) {
    state_cond_var_.SignalAll();
    // Here we need to check if we should activate sources, unthrottle, or
    // quit.
    // Note: when non_idle_queue_count_ == 0, we know that we are the last
    // queue remaining active. However, the application thread may still end
    // up calling HandleIdle, e.g. via the Cancel method, and that call may
    // quit the graph. Therefore, we should not unlock the mutex between
    // decrementing non_idle_queue_count_ and calling HandleIdle.
    HandleIdle();
  }
}

void Scheduler::Pause() {
  absl::MutexLock lock(&state_mutex_);
  if (state_ != STATE_RUNNING) {
    return;
  }
  state_ = STATE_PAUSED;
  SetQueuesRunning(false);
}

void Scheduler::Resume() {
  {
    absl::MutexLock lock(&state_mutex_);
    if (state_ != STATE_PAUSED) {
      return;
    }
    state_ = STATE_RUNNING;
    SetQueuesRunning(true);
    // If HandleIdle was called while graph was paused, it did nothing. So call
    // it now.
    HandleIdle();
  }
  SubmitWaitingTasksOnQueues();
}

void Scheduler::Cancel() {
  {
    absl::MutexLock lock(&state_mutex_);
    if (state_ != STATE_RUNNING && state_ != STATE_PAUSED) {
      return;
    }
    graph_->RecordError(::mediapipe::CancelledError());
    if (state_ == STATE_PAUSED) {
      // Keep the scheduler queue running, since we need to exhaust it.
      SetQueuesRunning(true);
    }
    state_ = STATE_CANCELLING;
    // Because we have recorded an error, this will cause the graph to quit.
    HandleIdle();
  }
  SubmitWaitingTasksOnQueues();
}

bool Scheduler::IsPaused() {
  absl::MutexLock lock(&state_mutex_);
  return state_ == STATE_PAUSED;
}

bool Scheduler::IsTerminated() {
  absl::MutexLock lock(&state_mutex_);
  return state_ == STATE_TERMINATED;
}

void Scheduler::CleanupAfterRun() {
  {
    absl::MutexLock lock(&state_mutex_);
    while (!sources_queue_.empty()) {
      sources_queue_.pop();
    }
    CHECK(app_thread_tasks_.empty());
  }
  for (auto queue : scheduler_queues_) {
    queue->CleanupAfterRun();
  }
  unopened_sources_.clear();
  active_sources_.clear();
  shared_.has_error = false;
}

internal::SchedulerTimes Scheduler::GetSchedulerTimes() {
  CHECK_EQ(state_, STATE_TERMINATED);
  return shared_.timer.GetSchedulerTimes();
}

}  // namespace internal
}  // namespace mediapipe
