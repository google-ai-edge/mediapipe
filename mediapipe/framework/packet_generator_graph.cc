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

#include "mediapipe/framework/packet_generator_graph.h"

#include <deque>
#include <functional>
#include <memory>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/delegating_executor.h"
#include "mediapipe/framework/executor.h"
#include "mediapipe/framework/packet_generator.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/proto_ns.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/framework/thread_pool_executor.h"
#include "mediapipe/framework/tool/status_util.h"

namespace mediapipe {

namespace {

// Create the input side packet set for a generator (provided by
// index in the canonical config).  unrunnable is set to true if the
// generator cannot be run given the currently available side packets
// (and false otherwise).  If an error occurs then unrunnable and
// input_side_packet_set are undefined.
::mediapipe::Status CreateInputsForGenerator(
    const ValidatedGraphConfig& validated_graph, int generator_index,
    const std::map<std::string, Packet>& side_packets,
    PacketSet* input_side_packet_set, bool* unrunnable) {
  const NodeTypeInfo& node_type_info =
      validated_graph.GeneratorInfos()[generator_index];
  const auto& generator_name = validated_graph.Config()
                                   .packet_generator(generator_index)
                                   .packet_generator();
  // Fill the PacketSet (if possible).
  *unrunnable = false;
  std::vector<::mediapipe::Status> statuses;
  for (CollectionItemId id = node_type_info.InputSidePacketTypes().BeginId();
       id < node_type_info.InputSidePacketTypes().EndId(); ++id) {
    const std::string& name =
        node_type_info.InputSidePacketTypes().TagMap()->Names()[id.value()];

    std::map<std::string, Packet>::const_iterator it = side_packets.find(name);
    if (it == side_packets.end()) {
      *unrunnable = true;
      continue;
    }
    input_side_packet_set->Get(id) = it->second;
    ::mediapipe::Status status =
        node_type_info.InputSidePacketTypes().Get(id).Validate(
            input_side_packet_set->Get(id));
    if (!status.ok()) {
      statuses.push_back(tool::AddStatusPrefix(
          absl::StrCat("Input side packet \"", name,
                       "\" for PacketGenerator \"", generator_name,
                       "\" is not of the correct type: "),
          status));
    }
  }
  if (!statuses.empty()) {
    return tool::CombinedStatus(
        absl::StrCat(generator_name, " had invalid configuration."), statuses);
  }
  return ::mediapipe::OkStatus();
}

// Generate the packets from a PacketGenerator, place them in
// output_side_packet_set, and validate their types.
::mediapipe::Status Generate(const ValidatedGraphConfig& validated_graph,
                             int generator_index,
                             const PacketSet& input_side_packet_set,
                             PacketSet* output_side_packet_set) {
  const NodeTypeInfo& node_type_info =
      validated_graph.GeneratorInfos()[generator_index];
  const PacketGeneratorConfig& generator_config =
      validated_graph.Config().packet_generator(generator_index);
  const auto& generator_name = generator_config.packet_generator();

  ASSIGN_OR_RETURN(
      auto static_access,
      internal::StaticAccessToGeneratorRegistry::CreateByNameInNamespace(
          validated_graph.Package(), generator_name),
      _ << generator_name << " is not a valid PacketGenerator.");
  MP_RETURN_IF_ERROR(static_access->Generate(generator_config.options(),
                                             input_side_packet_set,
                                             output_side_packet_set))
          .SetPrepend()
      << generator_name << "::Generate() failed. ";

  MP_RETURN_IF_ERROR(ValidatePacketSet(node_type_info.OutputSidePacketTypes(),
                                       *output_side_packet_set))
          .SetPrepend()
      << generator_name
      << "::Generate() output packets were of incorrect type: ";
  return ::mediapipe::OkStatus();
}

// GeneratorScheduler schedules the packet generators in a validated graph for
// execution on an executor.
class GeneratorScheduler {
 public:
  // If "executor" is null, a DelegatingExecutor will be created internally.
  // "initial" must be set to true for the first pass and false for subsequent
  // passes. If "initial" is false, non_base_generators contains the non-base
  // PacketGenerators (those not run at initialize time due to missing
  // dependencies).
  GeneratorScheduler(const ValidatedGraphConfig* validated_graph,
                     ::mediapipe::Executor* executor,
                     const std::vector<int>& non_base_generators, bool initial);

  // Run a PacketGenerator on a given executor on the provided input
  // side packets.  After running the generator, schedule any generators
  // which became runnable.
  void GenerateAndScheduleNext(int generator_index,
                               std::map<std::string, Packet>* side_packets,
                               std::unique_ptr<PacketSet> input_side_packet_set)
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Iterate through all generators in the config, scheduling any that
  // are runnable (and haven't been scheduled yet).
  void ScheduleAllRunnableGenerators(
      std::map<std::string, Packet>* side_packets) ABSL_LOCKS_EXCLUDED(mutex_);

  // Waits until there are no pending tasks.
  void WaitUntilIdle() ABSL_LOCKS_EXCLUDED(mutex_);

  // Stores the indexes of the packet generators that were not scheduled (or
  // rather, not executed) in non_scheduled_generators. Returns the combined
  // error status if there were errors while running the packet generators.
  // NOTE: This method should only be called when there are no pending tasks.
  ::mediapipe::Status GetNonScheduledGenerators(
      std::vector<int>* non_scheduled_generators) const;

 private:
  // Called by delegating_executor_ to add a task.
  void AddApplicationThreadTask(std::function<void()> task);

  // Run all the application thread tasks (which are kept track of in
  // app_thread_tasks_).
  void RunApplicationThreadTasks() ABSL_LOCKS_EXCLUDED(app_thread_mutex_);

  const ValidatedGraphConfig* const validated_graph_;
  ::mediapipe::Executor* executor_;

  mutable absl::Mutex mutex_;
  // The number of pending tasks.
  int num_tasks_ ABSL_GUARDED_BY(mutex_) = 0;
  // This condition variable is signaled when num_tasks_ becomes 0.
  absl::CondVar idle_condvar_;
  // Accumulates the error statuses while running the packet generators.
  std::vector<::mediapipe::Status> statuses_ ABSL_GUARDED_BY(mutex_);
  // scheduled_generators_[i] is true if the packet generator with index i was
  // scheduled (or rather, executed).
  std::vector<bool> scheduled_generators_ ABSL_GUARDED_BY(mutex_);

  absl::Mutex app_thread_mutex_;
  // Tasks to be executed on the application thread.
  std::deque<std::function<void()>> app_thread_tasks_
      ABSL_GUARDED_BY(app_thread_mutex_);
  std::unique_ptr<internal::DelegatingExecutor> delegating_executor_;
};

GeneratorScheduler::GeneratorScheduler(
    const ValidatedGraphConfig* validated_graph,
    ::mediapipe::Executor* executor,
    const std::vector<int>& non_base_generators, bool initial)
    : validated_graph_(validated_graph),
      executor_(executor),
      scheduled_generators_(validated_graph_->Config().packet_generator_size(),
                            !initial) {
  if (!executor_) {
    // Run on the application thread.
    delegating_executor_ = absl::make_unique<internal::DelegatingExecutor>(
        std::bind(&GeneratorScheduler::AddApplicationThreadTask, this,
                  std::placeholders::_1));
    executor_ = delegating_executor_.get();
  }

  if (!initial) {
    // Only schedule the non-base generators.
    for (int generator_index : non_base_generators) {
      scheduled_generators_[generator_index] = false;
    }
  }
}

void GeneratorScheduler::GenerateAndScheduleNext(
    int generator_index, std::map<std::string, Packet>* side_packets,
    std::unique_ptr<PacketSet> input_side_packet_set) {
  {
    absl::MutexLock lock(&mutex_);
    if (!statuses_.empty()) {
      // Return early, don't run the generator if we already have errors.
      return;
    }
  }
  PacketSet output_side_packet_set(
      validated_graph_->GeneratorInfos()[generator_index]
          .OutputSidePacketTypes()
          .TagMap());
  VLOG(1) << "Running generator " << generator_index;
  ::mediapipe::Status status =
      Generate(*validated_graph_, generator_index, *input_side_packet_set,
               &output_side_packet_set);

  {
    absl::MutexLock lock(&mutex_);
    if (!status.ok()) {
      statuses_.push_back(std::move(status));
      return;
    }
    // Add packets to side_packets .
    for (CollectionItemId id = output_side_packet_set.BeginId();
         id < output_side_packet_set.EndId(); ++id) {
      const auto& name = output_side_packet_set.TagMap()->Names()[id.value()];
      auto item = side_packets->emplace(name, output_side_packet_set.Get(id));
      if (!item.second) {
        statuses_.push_back(::mediapipe::AlreadyExistsError(
            absl::StrCat("Side packet \"", name, "\" was defined twice.")));
      }
    }
    if (!statuses_.empty()) {
      return;
    }
  }

  // Check all generators and schedule any that have become runnable.
  // TODO Instead of checking all of them, only check ones
  // that have input side packets which we have just produced.
  ScheduleAllRunnableGenerators(side_packets);
}

void GeneratorScheduler::ScheduleAllRunnableGenerators(
    std::map<std::string, Packet>* side_packets) {
  absl::MutexLock lock(&mutex_);
  const auto& generators = validated_graph_->Config().packet_generator();

  for (int index = 0; index < generators.size(); ++index) {
    if (scheduled_generators_[index]) {
      continue;
    }
    bool is_unrunnable = false;
    // TODO Input side packet set should only be created once.
    auto input_side_packet_set =
        absl::make_unique<PacketSet>(validated_graph_->GeneratorInfos()[index]
                                         .InputSidePacketTypes()
                                         .TagMap());

    ::mediapipe::Status status =
        CreateInputsForGenerator(*validated_graph_, index, *side_packets,
                                 input_side_packet_set.get(), &is_unrunnable);
    if (!status.ok()) {
      statuses_.push_back(std::move(status));
      continue;
    }
    if (is_unrunnable) {
      continue;
    }
    // The Generator is runnable, schedule a callback to run it.
    scheduled_generators_[index] = true;
    VLOG(1) << "Scheduling generator " << index;
    // Get around the fact that we can't capture a unique_ptr (this
    // means a memory leak will result if the lambda is not run).
    PacketSet* input_side_packet_set_ptr = input_side_packet_set.release();
    ++num_tasks_;
    mutex_.Unlock();
    executor_->Schedule(
        [this, index, side_packets, input_side_packet_set_ptr]() {
          GenerateAndScheduleNext(
              index, side_packets,
              std::unique_ptr<PacketSet>(input_side_packet_set_ptr));
          {
            absl::MutexLock lock(&mutex_);
            --num_tasks_;
            if (num_tasks_ == 0) {
              idle_condvar_.Signal();
            }
          }
        });
    mutex_.Lock();
  }
}

void GeneratorScheduler::WaitUntilIdle() {
  if (executor_ == delegating_executor_.get()) {
    // Run the tasks on the application thread.
    RunApplicationThreadTasks();
  } else {
    absl::MutexLock lock(&mutex_);
    while (num_tasks_ != 0) {
      idle_condvar_.Wait(&mutex_);
    }
  }
}

::mediapipe::Status GeneratorScheduler::GetNonScheduledGenerators(
    std::vector<int>* non_scheduled_generators) const {
  non_scheduled_generators->clear();

  absl::MutexLock lock(&mutex_);
  if (!statuses_.empty()) {
    return tool::CombinedStatus("PacketGeneratorGraph failed.", statuses_);
  }
  for (int i = 0; i < scheduled_generators_.size(); ++i) {
    if (!scheduled_generators_[i]) {
      non_scheduled_generators->push_back(i);
    }
  }
  return ::mediapipe::OkStatus();
}

void GeneratorScheduler::AddApplicationThreadTask(std::function<void()> task) {
  absl::MutexLock lock(&app_thread_mutex_);
  app_thread_tasks_.push_back(std::move(task));
}

void GeneratorScheduler::RunApplicationThreadTasks() {
  while (true) {
    std::function<void()> task_callback;
    {
      // Get the next task.
      absl::MutexLock lock(&app_thread_mutex_);
      if (app_thread_tasks_.empty()) {
        break;
      }
      task_callback = std::move(app_thread_tasks_.front());
      app_thread_tasks_.pop_front();
    }
    // Run the next task.  Don't hold any lock, since this task could
    // schedule further tasks to be run on the application thread.
    task_callback();
  }
}

}  // namespace

PacketGeneratorGraph::~PacketGeneratorGraph() {}

::mediapipe::Status PacketGeneratorGraph::Initialize(
    const ValidatedGraphConfig* validated_graph,
    ::mediapipe::Executor* executor,
    const std::map<std::string, Packet>& input_side_packets) {
  validated_graph_ = validated_graph;
  executor_ = executor;
  base_packets_ = input_side_packets;
  MP_RETURN_IF_ERROR(
      validated_graph_->CanAcceptSidePackets(input_side_packets));
  return ExecuteGenerators(&base_packets_, &non_base_generators_,
                           /*initial=*/true);
}

::mediapipe::Status PacketGeneratorGraph::RunGraphSetup(
    const std::map<std::string, Packet>& input_side_packets,
    std::map<std::string, Packet>* output_side_packets) const {
  *output_side_packets = base_packets_;
  for (const std::pair<const std::string, Packet>& item : input_side_packets) {
    auto iter = output_side_packets->find(item.first);
    if (iter != output_side_packets->end()) {
      return ::mediapipe::AlreadyExistsError(
          absl::StrCat("Side packet \"", iter->first, "\" was defined twice."));
    }
    output_side_packets->insert(iter, item);
  }
  std::vector<int> non_scheduled_generators;

  MP_RETURN_IF_ERROR(
      validated_graph_->CanAcceptSidePackets(input_side_packets));
  // This type check on the required side packets is redundant with
  // error checking in ExecuteGenerators, but we do it now to fail early.
  MP_RETURN_IF_ERROR(
      validated_graph_->ValidateRequiredSidePackets(*output_side_packets));
  MP_RETURN_IF_ERROR(ExecuteGenerators(
      output_side_packets, &non_scheduled_generators, /*initial=*/false));
  RET_CHECK(non_scheduled_generators.empty())
      << "Some Generators were unrunnable (validation should have failed).\n"
         "Generator indexes: "
      << absl::StrJoin(non_scheduled_generators, ", ");
  return ::mediapipe::OkStatus();
}

::mediapipe::Status PacketGeneratorGraph::ExecuteGenerators(
    std::map<std::string, Packet>* output_side_packets,
    std::vector<int>* non_scheduled_generators, bool initial) const {
  VLOG(1) << "ExecuteGenerators initial == " << initial;

  // Iterate through the generators and produce as many output
  // side packets as we can. The generators that don't have all the
  // required input side packets are put into non_scheduled_generators.
  // The ValidatedGraphConfig object is expected to already have sorted
  // generators in topological order.
  GeneratorScheduler scheduler(validated_graph_, executor_,
                               non_base_generators_, initial);
  scheduler.ScheduleAllRunnableGenerators(output_side_packets);
  // Do not return early if scheduler encountered an error.  The lambdas
  // in the executor must run in order to free resources.

  scheduler.WaitUntilIdle();

  // It is safe to return now, since all the tasks have run.
  return scheduler.GetNonScheduledGenerators(non_scheduled_generators);
}

}  // namespace mediapipe
