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

#include "mediapipe/framework/calculator_graph.h"

#include <stdio.h>

#include <algorithm>
#include <map>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_base.h"
#include "mediapipe/framework/counter_factory.h"
#include "mediapipe/framework/delegating_executor.h"
#include "mediapipe/framework/input_stream_manager.h"
#include "mediapipe/framework/mediapipe_profiling.h"
#include "mediapipe/framework/packet_generator.h"
#include "mediapipe/framework/packet_generator.pb.h"
#include "mediapipe/framework/packet_set.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/core_proto_inc.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/source_location.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/framework/status_handler.h"
#include "mediapipe/framework/status_handler.pb.h"
#include "mediapipe/framework/thread_pool_executor.h"
#include "mediapipe/framework/thread_pool_executor.pb.h"
#include "mediapipe/framework/tool/fill_packet_set.h"
#include "mediapipe/framework/tool/status_util.h"
#include "mediapipe/framework/tool/tag_map.h"
#include "mediapipe/framework/tool/validate.h"
#include "mediapipe/framework/tool/validate_name.h"
#include "mediapipe/framework/validated_graph_config.h"
#include "mediapipe/gpu/graph_support.h"
#include "mediapipe/util/cpu_util.h"
#ifndef MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#endif  // !defined(MEDIAPIPE_DISABLE_GPU)

namespace mediapipe {

namespace {

// Forcefully terminates the framework when the number of errors exceeds this
// threshold.
constexpr int kMaxNumAccumulatedErrors = 1000;
constexpr char kApplicationThreadExecutorType[] = "ApplicationThreadExecutor";

}  // namespace

void CalculatorGraph::ScheduleAllOpenableNodes() {
  // This method can only be called before the scheduler_.Start() call and the
  // graph input streams' SetHeader() calls because it is safe to call
  // node->ReadyForOpen() only before any node or graph input stream has
  // propagated header packets or generated output side packets, either of
  // which may cause a downstream node to be scheduled for OpenNode().
  for (CalculatorNode& node : *nodes_) {
    if (node.ReadyForOpen()) {
      scheduler_.ScheduleNodeForOpen(&node);
    }
  }
}

void CalculatorGraph::GraphInputStream::SetHeader(const Packet& header) {
  shard_.SetHeader(header);
  manager_->PropagateHeader();
  manager_->LockIntroData();
}

void CalculatorGraph::GraphInputStream::PropagateUpdatesToMirrors() {
  // Since GraphInputStream doesn't allow SetOffset() and
  // SetNextTimestampBound(), the timestamp bound to propagate is only
  // determined by the timestamp of the output packets.
  CHECK(!shard_.IsEmpty()) << "Shard with name \"" << manager_->Name()
                           << "\" failed";
  manager_->PropagateUpdatesToMirrors(
      shard_.LastAddedPacketTimestamp().NextAllowedInStream(), &shard_);
}

void CalculatorGraph::GraphInputStream::Close() {
  if (!shard_.IsEmpty()) {
    manager_->PropagateUpdatesToMirrors(Timestamp::Done(), &shard_);
  }
  manager_->Close();
}

CalculatorGraph::CalculatorGraph()
    : profiler_(std::make_shared<ProfilingContext>()), scheduler_(this) {
  counter_factory_ = absl::make_unique<BasicCounterFactory>();
}

CalculatorGraph::CalculatorGraph(const CalculatorGraphConfig& config)
    : CalculatorGraph() {
  counter_factory_ = absl::make_unique<BasicCounterFactory>();
  MEDIAPIPE_CHECK_OK(Initialize(config));
}

// Defining the destructor here lets us use incomplete types in the header;
// they only need to be fully visible here, where their destructor is
// instantiated.
CalculatorGraph::~CalculatorGraph() {
  // Stop periodic profiler output to ublock Executor destructors.
  ::mediapipe::Status status = profiler()->Stop();
  if (!status.ok()) {
    LOG(ERROR) << "During graph destruction: " << status;
  }
}

::mediapipe::Status CalculatorGraph::InitializePacketGeneratorGraph(
    const std::map<std::string, Packet>& side_packets) {
  // Create and initialize the output side packets.
  if (!validated_graph_->OutputSidePacketInfos().empty()) {
    output_side_packets_ = absl::make_unique<OutputSidePacketImpl[]>(
        validated_graph_->OutputSidePacketInfos().size());
  }
  for (int index = 0; index < validated_graph_->OutputSidePacketInfos().size();
       ++index) {
    const EdgeInfo& edge_info =
        validated_graph_->OutputSidePacketInfos()[index];
    MP_RETURN_IF_ERROR(output_side_packets_[index].Initialize(
        edge_info.name, edge_info.packet_type));
  }

  // If use_application_thread_ is true, the default executor is a
  // DelegatingExecutor. This DelegatingExecutor is tightly coupled to
  // scheduler_ and therefore cannot be used by packet_generator_graph_.
  Executor* default_executor = nullptr;
  if (!use_application_thread_) {
    default_executor = executors_[""].get();
    CHECK(default_executor);
  }
  // If default_executor is nullptr, then packet_generator_graph_ will create
  // its own DelegatingExecutor to use the application thread.
  return packet_generator_graph_.Initialize(validated_graph_.get(),
                                            default_executor, side_packets);
}

::mediapipe::Status CalculatorGraph::InitializeStreams() {
  any_packet_type_.SetAny();

  // Create and initialize the input streams.
  input_stream_managers_ = absl::make_unique<InputStreamManager[]>(
      validated_graph_->InputStreamInfos().size());
  for (int index = 0; index < validated_graph_->InputStreamInfos().size();
       ++index) {
    const EdgeInfo& edge_info = validated_graph_->InputStreamInfos()[index];
    MP_RETURN_IF_ERROR(input_stream_managers_[index].Initialize(
        edge_info.name, edge_info.packet_type, edge_info.back_edge));
  }

  // Create and initialize the output streams.
  output_stream_managers_ = absl::make_unique<OutputStreamManager[]>(
      validated_graph_->OutputStreamInfos().size());
  for (int index = 0; index < validated_graph_->OutputStreamInfos().size();
       ++index) {
    const EdgeInfo& edge_info = validated_graph_->OutputStreamInfos()[index];
    MP_RETURN_IF_ERROR(output_stream_managers_[index].Initialize(
        edge_info.name, edge_info.packet_type));
  }

  // Initialize GraphInputStreams.
  int graph_input_stream_count = 0;
  ASSIGN_OR_RETURN(
      auto input_tag_map,
      tool::TagMap::Create(validated_graph_->Config().input_stream()));
  for (const auto& stream_name : input_tag_map->Names()) {
    RET_CHECK(!::mediapipe::ContainsKey(graph_input_streams_, stream_name))
            .SetNoLogging()
        << "CalculatorGraph Initialization failed, graph input stream \""
        << stream_name << "\" was specified twice.";
    int output_stream_index = validated_graph_->OutputStreamIndex(stream_name);
    RET_CHECK_LE(0, output_stream_index).SetNoLogging();
    const EdgeInfo& edge_info =
        validated_graph_->OutputStreamInfos()[output_stream_index];
    RET_CHECK(NodeTypeInfo::NodeType::GRAPH_INPUT_STREAM ==
              edge_info.parent_node.type)
        .SetNoLogging();

    graph_input_streams_[stream_name] = absl::make_unique<GraphInputStream>(
        &output_stream_managers_[output_stream_index]);

    // Assign a virtual node ID to each graph input stream so we can treat
    // these as regular nodes for throttling.
    graph_input_stream_node_ids_[stream_name] =
        validated_graph_->CalculatorInfos().size() + graph_input_stream_count;
    ++graph_input_stream_count;
  }

  // Set the default mode for graph input streams.
  {
    absl::MutexLock lock(&full_input_streams_mutex_);
    graph_input_stream_add_mode_ = GraphInputStreamAddMode::WAIT_TILL_NOT_FULL;
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status CalculatorGraph::InitializeCalculatorNodes() {
  // Check if the user has specified a maximum queue size for an input stream.
  max_queue_size_ = validated_graph_->Config().max_queue_size();
  max_queue_size_ = max_queue_size_ ? max_queue_size_ : 100;

  // Use a local variable to avoid needing to lock errors_.
  std::vector<::mediapipe::Status> errors;

  // Create and initialize all the nodes in the graph.
  nodes_ = absl::make_unique<absl::FixedArray<CalculatorNode>>(
      validated_graph_->CalculatorInfos().size());
  for (int node_id = 0; node_id < validated_graph_->CalculatorInfos().size();
       ++node_id) {
    // buffer_size_hint will be positive if one was specified in
    // the graph proto.
    int buffer_size_hint = 0;
    const ::mediapipe::Status result = (*nodes_)[node_id].Initialize(
        validated_graph_.get(), node_id, input_stream_managers_.get(),
        output_stream_managers_.get(), output_side_packets_.get(),
        &buffer_size_hint, profiler_);
    if (buffer_size_hint > 0) {
      max_queue_size_ = std::max(max_queue_size_, buffer_size_hint);
    }
    if (!result.ok()) {
      // Collect as many errors as we can before failing.
      errors.push_back(result);
    }
  }
  if (!errors.empty()) {
    return tool::CombinedStatus(
        "CalculatorGraph::InitializeCalculatorNodes failed: ", errors);
  }

  VLOG(2) << "Maximum input stream queue size based on graph config: "
          << max_queue_size_;
  return ::mediapipe::OkStatus();
}

::mediapipe::Status CalculatorGraph::InitializeProfiler() {
  profiler_->Initialize(*validated_graph_);
  return ::mediapipe::OkStatus();
}

::mediapipe::Status CalculatorGraph::InitializeExecutors() {
  // If the ExecutorConfig for the default executor leaves the executor type
  // unspecified, default_executor_options points to the
  // ThreadPoolExecutorOptions in that ExecutorConfig. Otherwise,
  // default_executor_options is null.
  const ThreadPoolExecutorOptions* default_executor_options = nullptr;
  bool use_application_thread = false;
  for (const ExecutorConfig& executor_config :
       validated_graph_->Config().executor()) {
    if (::mediapipe::ContainsKey(executors_, executor_config.name())) {
      if (!executor_config.type().empty()) {
        return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
               << "ExecutorConfig for \"" << executor_config.name()
               << "\" has a \"type\" field but is also provided to the graph "
                  "with a CalculatorGraph::SetExecutor() call.";
      }
      continue;
    }
    if (executor_config.name().empty()) {
      // Executor name "" refers to the default executor.
      if (executor_config.type().empty()) {
        // For the default executor, an unspecified type means letting the
        // framework choose an appropriate executor type.
        default_executor_options = &executor_config.options().GetExtension(
            ThreadPoolExecutorOptions::ext);
        continue;
      }
      if (executor_config.type() == kApplicationThreadExecutorType) {
        // For the default executor, the type "ApplicationThreadExecutor" means
        // running synchronously on the calling thread.
        use_application_thread = true;
        continue;
      }
    }
    if (executor_config.type().empty()) {
      return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "ExecutorConfig for \"" << executor_config.name()
             << "\" does not have a \"type\" field. The executor \""
             << executor_config.name()
             << "\" must be provided to the graph with a "
                "CalculatorGraph::SetExecutor() call.";
    }
    // clang-format off
    ASSIGN_OR_RETURN(Executor* executor,
                     ExecutorRegistry::CreateByNameInNamespace(
                         validated_graph_->Package(),
                         executor_config.type(), executor_config.options()));
    // clang-format on
    MEDIAPIPE_CHECK_OK(SetExecutorInternal(
        executor_config.name(), std::shared_ptr<Executor>(executor)));
  }

  if (!::mediapipe::ContainsKey(executors_, "")) {
    MP_RETURN_IF_ERROR(InitializeDefaultExecutor(default_executor_options,
                                                 use_application_thread));
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status CalculatorGraph::InitializeDefaultExecutor(
    const ThreadPoolExecutorOptions* default_executor_options,
    bool use_application_thread) {
  // If specified, run synchronously on the calling thread.
  if (use_application_thread) {
    use_application_thread_ = true;
    MEDIAPIPE_CHECK_OK(SetExecutorInternal(
        "", std::make_shared<internal::DelegatingExecutor>(
                std::bind(&internal::Scheduler::AddApplicationThreadTask,
                          &scheduler_, std::placeholders::_1))));
    return ::mediapipe::OkStatus();
  }

  // Check the number of threads specified in the proto.
  int num_threads = default_executor_options == nullptr
                        ? 0
                        : default_executor_options->num_threads();

  // If the default (0 or -1) was specified, pick a suitable number of threads
  // depending on the number of processors in this system and the number of
  // calculators and packet generators in the calculator graph.
  if (num_threads == 0 || num_threads == -1) {
    num_threads = std::min(
        mediapipe::NumCPUCores(),
        std::max({validated_graph_->Config().node().size(),
                  validated_graph_->Config().packet_generator().size(), 1}));
  }
  MP_RETURN_IF_ERROR(
      CreateDefaultThreadPool(default_executor_options, num_threads));
  return ::mediapipe::OkStatus();
}

::mediapipe::Status CalculatorGraph::Initialize(
    std::unique_ptr<ValidatedGraphConfig> validated_graph,
    const std::map<std::string, Packet>& side_packets) {
  RET_CHECK(!initialized_).SetNoLogging()
      << "CalculatorGraph can be initialized only once.";
  RET_CHECK(validated_graph->Initialized()).SetNoLogging()
      << "validated_graph is not initialized.";
  validated_graph_ = std::move(validated_graph);

  MP_RETURN_IF_ERROR(InitializeExecutors());
  MP_RETURN_IF_ERROR(InitializePacketGeneratorGraph(side_packets));
  MP_RETURN_IF_ERROR(InitializeStreams());
  MP_RETURN_IF_ERROR(InitializeCalculatorNodes());
#ifdef MEDIAPIPE_PROFILER_AVAILABLE
  MP_RETURN_IF_ERROR(InitializeProfiler());
#endif

  initialized_ = true;
  return ::mediapipe::OkStatus();
}

::mediapipe::Status CalculatorGraph::Initialize(
    const CalculatorGraphConfig& input_config) {
  return Initialize(input_config, {});
}

::mediapipe::Status CalculatorGraph::Initialize(
    const CalculatorGraphConfig& input_config,
    const std::map<std::string, Packet>& side_packets) {
  auto validated_graph = absl::make_unique<ValidatedGraphConfig>();
  MP_RETURN_IF_ERROR(validated_graph->Initialize(input_config));
  return Initialize(std::move(validated_graph), side_packets);
}

::mediapipe::Status CalculatorGraph::Initialize(
    const std::vector<CalculatorGraphConfig>& input_configs,
    const std::vector<CalculatorGraphTemplate>& input_templates,
    const std::map<std::string, Packet>& side_packets,
    const std::string& graph_type, const Subgraph::SubgraphOptions* options) {
  auto validated_graph = absl::make_unique<ValidatedGraphConfig>();
  MP_RETURN_IF_ERROR(validated_graph->Initialize(input_configs, input_templates,
                                                 graph_type, options));
  return Initialize(std::move(validated_graph), side_packets);
}

::mediapipe::Status CalculatorGraph::ObserveOutputStream(
    const std::string& stream_name,
    std::function<::mediapipe::Status(const Packet&)> packet_callback) {
  RET_CHECK(initialized_).SetNoLogging()
      << "CalculatorGraph is not initialized.";
  // TODO Allow output observers to be attached by graph level
  // tag/index.
  int output_stream_index = validated_graph_->OutputStreamIndex(stream_name);
  if (output_stream_index < 0) {
    return ::mediapipe::NotFoundErrorBuilder(MEDIAPIPE_LOC)
           << "Unable to attach observer to output stream \"" << stream_name
           << "\" because it doesn't exist.";
  }
  auto observer = absl::make_unique<internal::OutputStreamObserver>();
  MP_RETURN_IF_ERROR(observer->Initialize(
      stream_name, &any_packet_type_, std::move(packet_callback),
      &output_stream_managers_[output_stream_index]));
  graph_output_streams_.push_back(std::move(observer));
  return ::mediapipe::OkStatus();
}

::mediapipe::StatusOr<OutputStreamPoller>
CalculatorGraph::AddOutputStreamPoller(const std::string& stream_name) {
  RET_CHECK(initialized_).SetNoLogging()
      << "CalculatorGraph is not initialized.";
  int output_stream_index = validated_graph_->OutputStreamIndex(stream_name);
  if (output_stream_index < 0) {
    return ::mediapipe::NotFoundErrorBuilder(MEDIAPIPE_LOC)
           << "Unable to attach observer to output stream \"" << stream_name
           << "\" because it doesn't exist.";
  }
  auto internal_poller = std::make_shared<internal::OutputStreamPollerImpl>();
  MP_RETURN_IF_ERROR(internal_poller->Initialize(
      stream_name, &any_packet_type_,
      std::bind(&CalculatorGraph::UpdateThrottledNodes, this,
                std::placeholders::_1, std::placeholders::_2),
      &output_stream_managers_[output_stream_index]));
  OutputStreamPoller poller(internal_poller);
  graph_output_streams_.push_back(std::move(internal_poller));
  return std::move(poller);
}

::mediapipe::StatusOr<Packet> CalculatorGraph::GetOutputSidePacket(
    const std::string& packet_name) {
  int side_packet_index = validated_graph_->OutputSidePacketIndex(packet_name);
  if (side_packet_index < 0) {
    return ::mediapipe::NotFoundErrorBuilder(MEDIAPIPE_LOC)
           << "Unable to get the output side packet \"" << packet_name
           << "\" because it doesn't exist.";
  }
  Packet output_packet;
  if (scheduler_.IsTerminated()) {
    // Side-packets from calculators can be retrieved only after the graph is
    // done.
    output_packet = output_side_packets_[side_packet_index].GetPacket();
  }
  if (output_packet.IsEmpty()) {
    // See if it exists in the base packets that come from PacketGenerators.
    // TODO: Update/remove this after b/119671096 is resolved.
    auto base_packets = packet_generator_graph_.BasePackets();
    auto base_packet_iter = base_packets.find(packet_name);
    auto current_run_side_packet_iter =
        current_run_side_packets_.find(packet_name);
    if (base_packet_iter != base_packets.end() &&
        !base_packet_iter->second.IsEmpty()) {
      output_packet = base_packet_iter->second;
    } else if (current_run_side_packet_iter !=
                   current_run_side_packets_.end() &&
               !current_run_side_packet_iter->second.IsEmpty()) {
      output_packet = current_run_side_packet_iter->second;
    } else {
      return ::mediapipe::UnavailableErrorBuilder(MEDIAPIPE_LOC)
             << "The output side packet \"" << packet_name
             << "\" is unavailable.";
    }
  }
  return output_packet;
}

::mediapipe::Status CalculatorGraph::Run(
    const std::map<std::string, Packet>& extra_side_packets) {
  RET_CHECK(graph_input_streams_.empty()).SetNoLogging()
      << "When using graph input streams, call StartRun() instead of Run() so "
         "that AddPacketToInputStream() and CloseInputStream() can be called.";
  MP_RETURN_IF_ERROR(StartRun(extra_side_packets, {}));
  return WaitUntilDone();
}

::mediapipe::Status CalculatorGraph::StartRun(
    const std::map<std::string, Packet>& extra_side_packets,
    const std::map<std::string, Packet>& stream_headers) {
  RET_CHECK(initialized_).SetNoLogging()
      << "CalculatorGraph is not initialized.";
  MP_RETURN_IF_ERROR(PrepareForRun(extra_side_packets, stream_headers));
  MP_RETURN_IF_ERROR(profiler_->Start(executors_[""].get()));
  scheduler_.Start();
  return ::mediapipe::OkStatus();
}

#ifndef MEDIAPIPE_DISABLE_GPU
::mediapipe::Status CalculatorGraph::SetGpuResources(
    std::shared_ptr<::mediapipe::GpuResources> resources) {
  RET_CHECK(!ContainsKey(service_packets_, kGpuService.key))
      << "The GPU resources have already been configured.";
  service_packets_[kGpuService.key] =
      MakePacket<std::shared_ptr<::mediapipe::GpuResources>>(
          std::move(resources));
  return ::mediapipe::OkStatus();
}

std::shared_ptr<::mediapipe::GpuResources> CalculatorGraph::GetGpuResources()
    const {
  auto service_iter = service_packets_.find(kGpuService.key);
  if (service_iter == service_packets_.end()) return nullptr;
  return service_iter->second.Get<std::shared_ptr<::mediapipe::GpuResources>>();
}

::mediapipe::StatusOr<std::map<std::string, Packet>>
CalculatorGraph::PrepareGpu(const std::map<std::string, Packet>& side_packets) {
  std::map<std::string, Packet> additional_side_packets;
  bool update_sp = false;
  bool uses_gpu = false;
  for (const auto& node : *nodes_) {
    if (node.UsesGpu()) {
      uses_gpu = true;
      break;
    }
  }
  if (uses_gpu) {
    auto service_iter = service_packets_.find(kGpuService.key);
    bool has_service = service_iter != service_packets_.end();

    auto legacy_sp_iter = side_packets.find(kGpuSharedSidePacketName);
    // Workaround for b/116875321: CalculatorRunner provides an empty packet,
    // instead of just leaving it unset.
    bool has_legacy_sp = legacy_sp_iter != side_packets.end() &&
                         !legacy_sp_iter->second.IsEmpty();

    std::shared_ptr<::mediapipe::GpuResources> gpu_resources;
    if (has_service) {
      if (has_legacy_sp) {
        LOG(WARNING)
            << "::mediapipe::GpuSharedData provided as a side packet while the "
            << "graph already had one; ignoring side packet";
      }
      gpu_resources = service_iter->second
                          .Get<std::shared_ptr<::mediapipe::GpuResources>>();
      update_sp = true;
    } else {
      if (has_legacy_sp) {
        gpu_resources =
            legacy_sp_iter->second.Get<::mediapipe::GpuSharedData*>()
                ->gpu_resources;
      } else {
        ASSIGN_OR_RETURN(gpu_resources, ::mediapipe::GpuResources::Create());
        update_sp = true;
      }
      service_packets_[kGpuService.key] =
          MakePacket<std::shared_ptr<::mediapipe::GpuResources>>(gpu_resources);
    }

    // Create or replace the legacy side packet if needed.
    if (update_sp) {
      legacy_gpu_shared_.reset(new ::mediapipe::GpuSharedData(gpu_resources));
      additional_side_packets[kGpuSharedSidePacketName] =
          MakePacket<::mediapipe::GpuSharedData*>(legacy_gpu_shared_.get());
    }

    // Set up executors.
    for (auto& node : *nodes_) {
      if (node.UsesGpu()) {
        MP_RETURN_IF_ERROR(gpu_resources->PrepareGpuNode(&node));
      }
    }
    for (const auto& name_executor : gpu_resources->GetGpuExecutors()) {
      MP_RETURN_IF_ERROR(
          SetExecutorInternal(name_executor.first, name_executor.second));
    }
  }
  return additional_side_packets;
}
#endif  // !defined(MEDIAPIPE_DISABLE_GPU)

::mediapipe::Status CalculatorGraph::PrepareForRun(
    const std::map<std::string, Packet>& extra_side_packets,
    const std::map<std::string, Packet>& stream_headers) {
  if (VLOG_IS_ON(1)) {
    for (const auto& item : extra_side_packets) {
      VLOG(1) << "Adding extra_side_packet with name: " << item.first;
    }
  }

  {
    absl::MutexLock lock(&error_mutex_);
    errors_.clear();
    has_error_ = false;
  }
  num_closed_graph_input_streams_ = 0;

  std::map<std::string, Packet> additional_side_packets;
#ifndef MEDIAPIPE_DISABLE_GPU
  ASSIGN_OR_RETURN(additional_side_packets, PrepareGpu(extra_side_packets));
#endif  // !defined(MEDIAPIPE_DISABLE_GPU)

  const std::map<std::string, Packet>* input_side_packets;
  if (!additional_side_packets.empty()) {
    additional_side_packets.insert(extra_side_packets.begin(),
                                   extra_side_packets.end());
    input_side_packets = &additional_side_packets;
  } else {
    input_side_packets = &extra_side_packets;
  }

  current_run_side_packets_.clear();
  ::mediapipe::Status generator_status = packet_generator_graph_.RunGraphSetup(
      *input_side_packets, &current_run_side_packets_);

  CallStatusHandlers(GraphRunState::PRE_RUN, generator_status);

  if (!generator_status.ok()) {
    return generator_status;
  }

  // If there was an error on the CallStatusHandlers (PRE_RUN), it was stored
  // in the error list. We return immediately notifying this to the caller.
  ::mediapipe::Status error_status;
  if (has_error_) {
    GetCombinedErrors(&error_status);
    LOG(ERROR) << error_status;
    return error_status;
  }

  if (VLOG_IS_ON(1)) {
    std::vector<std::string> input_side_packet_names;
    for (const auto& item : current_run_side_packets_) {
      input_side_packet_names.push_back(item.first);
    }
    VLOG(1) << "Final input side packet names are: "
            << absl::StrJoin(input_side_packet_names, ",");
  }

  Executor* default_executor = nullptr;
  if (!use_application_thread_) {
    default_executor = executors_[""].get();
    RET_CHECK(default_executor);
  }
  scheduler_.Reset();

  {
    absl::MutexLock lock(&full_input_streams_mutex_);
    // Initialize a count per source node to store the number of input streams
    // that are full and are affected by the source node. A node is considered
    // to be throttled if the count corresponding to this node is non-zero.
    // i.e. there is at least one affected stream which is full. We treat the
    // graph input streams as nodes because they might need to be throttled.
    full_input_streams_.clear();
    full_input_streams_.resize(validated_graph_->CalculatorInfos().size() +
                               graph_input_streams_.size());
  }

  for (auto& item : graph_input_streams_) {
    item.second->PrepareForRun(
        std::bind(&CalculatorGraph::RecordError, this, std::placeholders::_1));
  }
  for (int index = 0; index < validated_graph_->OutputSidePacketInfos().size();
       ++index) {
    output_side_packets_[index].PrepareForRun(
        std::bind(&CalculatorGraph::RecordError, this, std::placeholders::_1));
  }
  for (CalculatorNode& node : *nodes_) {
    InputStreamManager::QueueSizeCallback queue_size_callback =
        std::bind(&CalculatorGraph::UpdateThrottledNodes, this,
                  std::placeholders::_1, std::placeholders::_2);
    node.SetQueueSizeCallbacks(queue_size_callback, queue_size_callback);
    scheduler_.AssignNodeToSchedulerQueue(&node);
    const ::mediapipe::Status result = node.PrepareForRun(
        current_run_side_packets_, service_packets_,
        std::bind(&internal::Scheduler::ScheduleNodeForOpen, &scheduler_,
                  &node),
        std::bind(&internal::Scheduler::AddNodeToSourcesQueue, &scheduler_,
                  &node),
        std::bind(&internal::Scheduler::ScheduleNodeIfNotThrottled, &scheduler_,
                  &node, std::placeholders::_1),
        std::bind(&CalculatorGraph::RecordError, this, std::placeholders::_1),
        counter_factory_.get());
    if (!result.ok()) {
      // Collect as many errors as we can before failing.
      RecordError(result);
    }
  }
  for (auto& graph_output_stream : graph_output_streams_) {
    graph_output_stream->PrepareForRun(
        [&graph_output_stream, this] {
          ::mediapipe::Status status = graph_output_stream->Notify();
          if (!status.ok()) {
            RecordError(status);
          }
          scheduler_.EmittedObservedOutput();
        },
        [this](::mediapipe::Status status) { RecordError(status); });
  }

  if (GetCombinedErrors(&error_status)) {
    LOG(ERROR) << error_status;
    CleanupAfterRun(&error_status);
    return error_status;
  }

  // Ensure that the latest value of max queue size is passed to all input
  // streams.
  for (auto& node : *nodes_) {
    node.SetMaxInputStreamQueueSize(max_queue_size_);
  }

  // Allow graph input streams to override the global max queue size.
  for (const auto& name_max : graph_input_stream_max_queue_size_) {
    std::unique_ptr<GraphInputStream>* stream =
        ::mediapipe::FindOrNull(graph_input_streams_, name_max.first);
    RET_CHECK(stream).SetNoLogging() << absl::Substitute(
        "SetInputStreamMaxQueueSize called on \"$0\" which is not a "
        "graph input stream.",
        name_max.first);
    (*stream)->SetMaxQueueSize(name_max.second);
  }

  for (CalculatorNode& node : *nodes_) {
    if (node.IsSource()) {
      scheduler_.AddUnopenedSourceNode(&node);
      has_sources_ = true;
    }
  }

  VLOG(2) << "Opening calculators.";
  // Open the calculators.
  ScheduleAllOpenableNodes();

  // Header has to be set after the above preparation, since the header is
  // propagated to the connected streams. In addition, setting the header
  // packet may make a node ready for OpenNode(), and we should not schedule
  // OpenNode() before the ScheduleAllOpenableNodes() call.
  for (auto& item : graph_input_streams_) {
    auto header = stream_headers.find(item.first);
    if (header != stream_headers.end()) {
      item.second->SetHeader(header->second);
    } else {
      // SetHeader() not only sets the header but also propagates it to the
      // mirrors. Propagate the header to mirrors even if the header is empty
      // to inform mirrors that they can proceed.
      item.second->SetHeader(Packet());
    }
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status CalculatorGraph::WaitUntilIdle() {
  MP_RETURN_IF_ERROR(scheduler_.WaitUntilIdle());
  VLOG(2) << "Scheduler idle.";
  ::mediapipe::Status status = ::mediapipe::OkStatus();
  if (GetCombinedErrors(&status)) {
    LOG(ERROR) << status;
  }
  return status;
}

::mediapipe::Status CalculatorGraph::WaitUntilDone() {
  VLOG(2) << "Waiting for scheduler to terminate...";
  MP_RETURN_IF_ERROR(scheduler_.WaitUntilDone());
  VLOG(2) << "Scheduler terminated.";

  return FinishRun();
}

::mediapipe::Status CalculatorGraph::WaitForObservedOutput() {
  return scheduler_.WaitForObservedOutput();
}

::mediapipe::Status CalculatorGraph::AddPacketToInputStream(
    const std::string& stream_name, const Packet& packet) {
  return AddPacketToInputStreamInternal(stream_name, packet);
}

::mediapipe::Status CalculatorGraph::AddPacketToInputStream(
    const std::string& stream_name, Packet&& packet) {
  return AddPacketToInputStreamInternal(stream_name, std::move(packet));
}

// We avoid having two copies of this code for AddPacketToInputStream(
// const Packet&) and AddPacketToInputStream(Packet &&) by having this
// internal-only templated version.  T&& is a forwarding reference here, so
// std::forward will deduce the correct type as we pass along packet.
template <typename T>
::mediapipe::Status CalculatorGraph::AddPacketToInputStreamInternal(
    const std::string& stream_name, T&& packet) {
  std::unique_ptr<GraphInputStream>* stream =
      ::mediapipe::FindOrNull(graph_input_streams_, stream_name);
  RET_CHECK(stream).SetNoLogging() << absl::Substitute(
      "AddPacketToInputStream called on input stream \"$0\" which is not a "
      "graph input stream.",
      stream_name);
  int node_id =
      ::mediapipe::FindOrDie(graph_input_stream_node_ids_, stream_name);
  CHECK_GE(node_id, validated_graph_->CalculatorInfos().size());
  {
    absl::MutexLock lock(&full_input_streams_mutex_);
    if (graph_input_stream_add_mode_ ==
        GraphInputStreamAddMode::ADD_IF_NOT_FULL) {
      if (has_error_) {
        ::mediapipe::Status error_status;
        GetCombinedErrors("Graph has errors: ", &error_status);
        return error_status;
      }
      // Return with StatusUnavailable if this stream is being throttled.
      if (!full_input_streams_[node_id].empty()) {
        return ::mediapipe::UnavailableErrorBuilder(MEDIAPIPE_LOC)
               << "Graph is throttled.";
      }
    } else if (graph_input_stream_add_mode_ ==
               GraphInputStreamAddMode::WAIT_TILL_NOT_FULL) {
      // Wait until this stream is not being throttled.
      // TODO: instead of checking has_error_, we could just check
      // if the graph is done. That could also be indicated by returning an
      // error from WaitUntilGraphInputStreamUnthrottled.
      while (!has_error_ && !full_input_streams_[node_id].empty()) {
        // TODO: allow waiting for a specific stream?
        scheduler_.WaitUntilGraphInputStreamUnthrottled(
            &full_input_streams_mutex_);
      }
      if (has_error_) {
        ::mediapipe::Status error_status;
        GetCombinedErrors("Graph has errors: ", &error_status);
        return error_status;
      }
    }
  }

  // Adding profiling info for a new packet entering the graph.
  const std::string* stream_id = &(*stream)->GetManager()->Name();
  profiler_->LogEvent(TraceEvent(TraceEvent::PROCESS)
                          .set_is_finish(true)
                          .set_input_ts(packet.Timestamp())
                          .set_stream_id(stream_id)
                          .set_packet_ts(packet.Timestamp())
                          .set_packet_data_id(&packet));

  // InputStreamManager is thread safe. GraphInputStream is not, so this method
  // should not be called by multiple threads concurrently. Note that this could
  // potentially lead to the max queue size being exceeded by one packet at most
  // because we don't have the lock over the input stream.
  (*stream)->AddPacket(std::forward<T>(packet));
  if (has_error_) {
    ::mediapipe::Status error_status;
    GetCombinedErrors("Graph has errors: ", &error_status);
    return error_status;
  }
  (*stream)->PropagateUpdatesToMirrors();

  VLOG(2) << "Packet added directly to: " << stream_name;
  // Note: one reason why we need to call the scheduler here is that we have
  // re-throttled the graph input streams, and we may need to unthrottle them
  // again if the graph is still idle. Unthrottling basically only lets in one
  // packet at a time. TODO: add test.
  scheduler_.AddedPacketToGraphInputStream();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status CalculatorGraph::SetInputStreamMaxQueueSize(
    const std::string& stream_name, int max_queue_size) {
  // graph_input_streams_ has not been filled in yet, so we'll check this when
  // it is applied when the graph is started.
  graph_input_stream_max_queue_size_[stream_name] = max_queue_size;
  return ::mediapipe::OkStatus();
}

bool CalculatorGraph::HasInputStream(const std::string& stream_name) {
  return ::mediapipe::FindOrNull(graph_input_streams_, stream_name) != nullptr;
}

::mediapipe::Status CalculatorGraph::CloseInputStream(
    const std::string& stream_name) {
  std::unique_ptr<GraphInputStream>* stream =
      ::mediapipe::FindOrNull(graph_input_streams_, stream_name);
  RET_CHECK(stream).SetNoLogging() << absl::Substitute(
      "CloseInputStream called on input stream \"$0\" which is not a graph "
      "input stream.",
      stream_name);
  // The following IsClosed() and Close() sequence is not atomic. Multiple
  // threads cannot call CloseInputStream() on the same stream_name at the same
  // time.
  if ((*stream)->IsClosed()) {
    return ::mediapipe::OkStatus();
  }

  (*stream)->Close();

  if (++num_closed_graph_input_streams_ == graph_input_streams_.size()) {
    scheduler_.ClosedAllGraphInputStreams();
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status CalculatorGraph::CloseAllInputStreams() {
  for (auto& item : graph_input_streams_) {
    item.second->Close();
  }

  num_closed_graph_input_streams_ = graph_input_streams_.size();
  scheduler_.ClosedAllGraphInputStreams();

  return ::mediapipe::OkStatus();
}

::mediapipe::Status CalculatorGraph::CloseAllPacketSources() {
  for (auto& item : graph_input_streams_) {
    item.second->Close();
  }

  num_closed_graph_input_streams_ = graph_input_streams_.size();
  scheduler_.ClosedAllGraphInputStreams();
  scheduler_.CloseAllSourceNodes();

  return ::mediapipe::OkStatus();
}

void CalculatorGraph::RecordError(const ::mediapipe::Status& error) {
  VLOG(2) << "RecordError called with " << error;
  {
    absl::MutexLock lock(&error_mutex_);
    errors_.push_back(error);
    has_error_ = true;
    scheduler_.SetHasError(true);
    for (const auto& stream : graph_output_streams_) {
      stream->NotifyError();
    }
    if (errors_.size() > kMaxNumAccumulatedErrors) {
      for (const ::mediapipe::Status& error : errors_) {
        LOG(ERROR) << error;
      }
      LOG(FATAL) << "Forcefully aborting to prevent the framework running out "
                    "of memory.";
    }
  }
}

bool CalculatorGraph::GetCombinedErrors(::mediapipe::Status* error_status) {
  return GetCombinedErrors("CalculatorGraph::Run() failed in Run: ",
                           error_status);
}

bool CalculatorGraph::GetCombinedErrors(const std::string& error_prefix,
                                        ::mediapipe::Status* error_status) {
  absl::MutexLock lock(&error_mutex_);
  if (!errors_.empty()) {
    *error_status = tool::CombinedStatus(error_prefix, errors_);
    return true;
  }
  return false;
}

void CalculatorGraph::CallStatusHandlers(GraphRunState graph_run_state,
                                         const ::mediapipe::Status& status) {
  for (int status_handler_index = 0;
       status_handler_index < validated_graph_->Config().status_handler_size();
       ++status_handler_index) {
    const auto& handler_config =
        validated_graph_->Config().status_handler(status_handler_index);
    const auto& handler_type = handler_config.status_handler();

    const auto& status_handler_info =
        validated_graph_->StatusHandlerInfos()[status_handler_index];
    const PacketTypeSet& packet_type_set =
        status_handler_info.InputSidePacketTypes();
    ::mediapipe::StatusOr<std::unique_ptr<PacketSet>> packet_set_statusor =
        tool::FillPacketSet(packet_type_set, current_run_side_packets_,
                            nullptr);
    if (!packet_set_statusor.ok()) {
      RecordError(::mediapipe::StatusBuilder(
                      std::move(packet_set_statusor).status(), MEDIAPIPE_LOC)
                      .SetPrepend()
                  << "Skipping run of " << handler_type << ": ");
      continue;
    }
    ::mediapipe::StatusOr<
        std::unique_ptr<internal::StaticAccessToStatusHandler>>
        static_access_statusor = internal::StaticAccessToStatusHandlerRegistry::
            CreateByNameInNamespace(validated_graph_->Package(), handler_type);
    CHECK(static_access_statusor.ok()) << handler_type << " is not registered.";
    auto static_access = std::move(static_access_statusor).ValueOrDie();
    ::mediapipe::Status handler_result;
    if (graph_run_state == GraphRunState::PRE_RUN) {
      handler_result = static_access->HandlePreRunStatus(
          handler_config.options(), *packet_set_statusor.ValueOrDie(), status);
    } else {  // POST_RUN
      handler_result = static_access->HandleStatus(
          handler_config.options(), *packet_set_statusor.ValueOrDie(), status);
    }
    if (!handler_result.ok()) {
      ::mediapipe::StatusBuilder builder(std::move(handler_result),
                                         MEDIAPIPE_LOC);
      builder.SetPrepend() << handler_type;
      if (graph_run_state == GraphRunState::PRE_RUN) {
        builder << "::HandlePreRunStatus failed: ";
      } else {  // POST_RUN
        builder << "::HandleStatus failed: ";
      }
      RecordError(builder);
    }
  }
}

int CalculatorGraph::GetMaxInputStreamQueueSize() { return max_queue_size_; }

void CalculatorGraph::UpdateThrottledNodes(InputStreamManager* stream,
                                           bool* stream_was_full) {
  // TODO Change the throttling code to use the index directly
  // rather than looking up a stream name.
  int node_index = validated_graph_->OutputStreamToNode(stream->Name());
  absl::flat_hash_set<int> owned_set;
  const absl::flat_hash_set<int>* upstream_nodes;
  if (node_index >= validated_graph_->CalculatorInfos().size()) {
    // TODO just create a NodeTypeInfo object for each virtual node.
    owned_set.insert(node_index);
    upstream_nodes = &owned_set;
  } else {
    upstream_nodes =
        &validated_graph_->CalculatorInfos()[node_index].AncestorSources();
  }
  CHECK(upstream_nodes);
  std::vector<CalculatorNode*> nodes_to_schedule;

  {
    absl::MutexLock lock(&full_input_streams_mutex_);
    // Note that the change in stream status is recomputed here within the
    // MutexLock in order to avoid interference between callbacks arriving
    // out of order.
    // Note that |stream_was_full| is maintained by the node throttling logic
    // in this function and is guarded by full_input_streams_mutex_.
    bool stream_is_full = stream->IsFull();
    if (*stream_was_full != stream_is_full) {
      for (int node_id : *upstream_nodes) {
        VLOG(2) << "Stream \"" << stream->Name() << "\" is "
                << (stream_is_full ? "throttling" : "no longer throttling")
                << " node with node ID " << node_id;
        ::mediapipe::LogEvent(
            profiler_.get(),
            TraceEvent(stream_is_full ? TraceEvent::THROTTLED
                                      : TraceEvent::UNTHROTTLED)
                .set_stream_id(&stream->Name()));
        bool was_throttled = !full_input_streams_[node_id].empty();
        if (stream_is_full) {
          DCHECK_EQ(full_input_streams_[node_id].count(stream), 0);
          full_input_streams_[node_id].insert(stream);
        } else {
          DCHECK_EQ(full_input_streams_[node_id].count(stream), 1);
          full_input_streams_[node_id].erase(stream);
        }

        bool is_throttled = !full_input_streams_[node_id].empty();
        bool is_graph_input_stream =
            node_id >= validated_graph_->CalculatorInfos().size();
        if (is_graph_input_stream) {
          // Making these calls while holding full_input_streams_mutex_
          // ensures they are correctly serialized.
          // Note: !is_throttled implies was_throttled, but not vice versa.
          if (!is_throttled) {
            scheduler_.UnthrottledGraphInputStream();
          } else if (!was_throttled && is_throttled) {
            scheduler_.ThrottledGraphInputStream();
          }
        } else {
          if (!is_throttled) {
            CalculatorNode& node = (*nodes_)[node_id];
            // Add this node to the scheduler queue if possible.
            if (node.Active() && !node.Closed()) {
              nodes_to_schedule.emplace_back(&node);
            }
          }
        }
      }
    }
    *stream_was_full = stream_is_full;
  }

  if (!nodes_to_schedule.empty()) {
    scheduler_.ScheduleUnthrottledReadyNodes(nodes_to_schedule);
  }
}

bool CalculatorGraph::IsNodeThrottled(int node_id) {
  absl::MutexLock lock(&full_input_streams_mutex_);
  return max_queue_size_ != -1 && !full_input_streams_[node_id].empty();
}

// Returns true if an input stream serves as a graph-output-stream.
bool IsGraphOutputStream(
    InputStreamManager* stream,
    const std::vector<std::shared_ptr<internal::GraphOutputStream>>&
        graph_output_streams) {
  for (auto& graph_output_stream : graph_output_streams) {
    if (stream == graph_output_stream->input_stream()) {
      return true;
    }
  }
  return false;
}

bool CalculatorGraph::UnthrottleSources() {
  // NOTE: We can be sure that this function will grow input streams enough
  // to unthrottle at least one source node.  The current stream queue sizes
  // will remain unchanged until at least one source node becomes unthrottled.
  // This is a sufficient because succesfully growing at least one full input
  // stream during each call to UnthrottleSources will eventually resolve
  // each deadlock.
  absl::flat_hash_set<InputStreamManager*> full_streams;
  {
    absl::MutexLock lock(&full_input_streams_mutex_);
    for (absl::flat_hash_set<InputStreamManager*>& s : full_input_streams_) {
      for (auto& stream : s) {
        // The queue size of a graph output stream shouldn't change. Throttling
        // should continue until the caller of the graph output stream consumes
        // enough packets.
        if (!IsGraphOutputStream(stream, graph_output_streams_)) {
          full_streams.insert(stream);
        }
      }
    }
  }
  for (InputStreamManager* stream : full_streams) {
    if (Config().report_deadlock()) {
      RecordError(::mediapipe::UnavailableError(absl::StrCat(
          "Detected a deadlock due to input throttling for: \"", stream->Name(),
          "\". All calculators are idle while packet sources remain active "
          "and throttled.  Consider adjusting \"max_queue_size\" or "
          "\"resolve_deadlock\".")));
      continue;
    }
    int new_size = stream->QueueSize() + 1;
    stream->SetMaxQueueSize(new_size);
    LOG_EVERY_N(WARNING, 100)
        << "Resolved a deadlock by increasing max_queue_size of input stream: "
        << stream->Name() << " to: " << new_size
        << ". Consider increasing max_queue_size for better performance.";
  }
  return !full_streams.empty();
}

CalculatorGraph::GraphInputStreamAddMode
CalculatorGraph::GetGraphInputStreamAddMode() const {
  absl::MutexLock lock(&full_input_streams_mutex_);
  return graph_input_stream_add_mode_;
}

void CalculatorGraph::SetGraphInputStreamAddMode(GraphInputStreamAddMode mode) {
  absl::MutexLock lock(&full_input_streams_mutex_);
  graph_input_stream_add_mode_ = mode;
}

void CalculatorGraph::Cancel() {
  // TODO This function should return ::mediapipe::Status.
  scheduler_.Cancel();
}

void CalculatorGraph::Pause() { scheduler_.Pause(); }

void CalculatorGraph::Resume() { scheduler_.Resume(); }

::mediapipe::Status CalculatorGraph::SetServicePacket(
    const GraphServiceBase& service, Packet p) {
  // TODO: check that the graph has not been started!
  service_packets_[service.key] = std::move(p);
  return ::mediapipe::OkStatus();
}

Packet CalculatorGraph::GetServicePacket(const GraphServiceBase& service) {
  auto it = service_packets_.find(service.key);
  if (it == service_packets_.end()) {
    return {};
  }
  return it->second;
}

::mediapipe::Status CalculatorGraph::SetExecutorInternal(
    const std::string& name, std::shared_ptr<Executor> executor) {
  if (!executors_.emplace(name, executor).second) {
    return ::mediapipe::AlreadyExistsErrorBuilder(MEDIAPIPE_LOC)
           << "SetExecutor must be called only once for the executor \"" << name
           << "\"";
  }
  if (name.empty()) {
    scheduler_.SetExecutor(executor.get());
  } else {
    MP_RETURN_IF_ERROR(scheduler_.SetNonDefaultExecutor(name, executor.get()));
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status CalculatorGraph::SetExecutor(
    const std::string& name, std::shared_ptr<Executor> executor) {
  RET_CHECK(!initialized_)
      << "SetExecutor can only be called before Initialize()";
  if (IsReservedExecutorName(name)) {
    return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "\"" << name << "\" is a reserved executor name.";
  }
  return SetExecutorInternal(name, std::move(executor));
}

::mediapipe::Status CalculatorGraph::CreateDefaultThreadPool(
    const ThreadPoolExecutorOptions* default_executor_options,
    int num_threads) {
  MediaPipeOptions extendable_options;
  ThreadPoolExecutorOptions* options =
      extendable_options.MutableExtension(ThreadPoolExecutorOptions::ext);
  if (default_executor_options != nullptr) {
    options->CopyFrom(*default_executor_options);
  }
  options->set_num_threads(num_threads);
  // clang-format off
  ASSIGN_OR_RETURN(Executor* executor,
                   ThreadPoolExecutor::Create(extendable_options));
  // clang-format on
  return SetExecutorInternal("", std::shared_ptr<Executor>(executor));
}

// static
bool CalculatorGraph::IsReservedExecutorName(const std::string& name) {
  return ValidatedGraphConfig::IsReservedExecutorName(name);
}

::mediapipe::Status CalculatorGraph::FinishRun() {
  // Check for any errors that may have occurred.
  ::mediapipe::Status status = ::mediapipe::OkStatus();
  MP_RETURN_IF_ERROR(profiler_->Stop());
  GetCombinedErrors(&status);
  CleanupAfterRun(&status);
  return status;
}

void CalculatorGraph::CleanupAfterRun(::mediapipe::Status* status) {
  for (auto& item : graph_input_streams_) {
    item.second->Close();
  }

  CallStatusHandlers(GraphRunState::POST_RUN, *status);
  if (has_error_) {
    // Obtain the combined status again, so that it includes the new errors
    // added by CallStatusHandlers.
    GetCombinedErrors(status);
    CHECK(!status->ok());
  } else {
    MEDIAPIPE_CHECK_OK(*status);
  }

  for (CalculatorNode& node : *nodes_) {
    node.CleanupAfterRun(*status);
  }

  for (auto& graph_output_stream : graph_output_streams_) {
    graph_output_stream->input_stream()->Close();
  }

  scheduler_.CleanupAfterRun();

  {
    absl::MutexLock lock(&error_mutex_);
    errors_.clear();
    has_error_ = false;
  }

  {
    absl::MutexLock lock(&full_input_streams_mutex_);
    full_input_streams_.clear();
  }
  // Note: output_side_packets_ and current_run_side_packets_ are not cleared
  // in order to enable GetOutputSidePacket after WaitUntilDone.
}

const OutputStreamManager* CalculatorGraph::FindOutputStreamManager(
    const std::string& name) {
  return &output_stream_managers_
              .get()[validated_graph_->OutputStreamIndex(name)];
}

namespace {
void PrintTimingToInfo(const std::string& label, int64 timer_value) {
  const int64 total_seconds = timer_value / 1000000ll;
  const int64 days = total_seconds / (3600ll * 24ll);
  const int64 hours = (total_seconds / 3600ll) % 24ll;
  const int64 minutes = (total_seconds / 60ll) % 60ll;
  const int64 seconds = total_seconds % 60ll;
  const int64 milliseconds = (timer_value / 1000ll) % 1000ll;
  LOG(INFO) << label << " took "
            << absl::StrFormat(
                   "%02lld days, %02lld:%02lld:%02lld.%03lld (total seconds: "
                   "%lld.%06lld)",
                   days, hours, minutes, seconds, milliseconds, total_seconds,
                   timer_value % int64{1000000});
}

bool MetricElementComparator(const std::pair<std::string, int64>& e1,
                             const std::pair<std::string, int64>& e2) {
  return e1.second > e2.second;
}
}  // namespace

::mediapipe::Status CalculatorGraph::GetCalculatorProfiles(
    std::vector<CalculatorProfile>* profiles) const {
  return profiler_->GetCalculatorProfiles(profiles);
}

}  // namespace mediapipe
