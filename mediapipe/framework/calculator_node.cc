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

#include "mediapipe/framework/calculator_node.h"

#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_base.h"
#include "mediapipe/framework/counter_factory.h"
#include "mediapipe/framework/input_stream_manager.h"
#include "mediapipe/framework/mediapipe_profiling.h"
#include "mediapipe/framework/output_stream_manager.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/packet_set.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/proto_ns.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/source_location.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/tool/name_util.h"
#include "mediapipe/framework/tool/status_util.h"
#include "mediapipe/framework/tool/tag_map.h"
#include "mediapipe/framework/tool/validate_name.h"

namespace mediapipe {

namespace {

const PacketType* GetPacketType(const PacketTypeSet& packet_type_set,
                                const std::string& tag, const int index) {
  CollectionItemId id;
  if (tag.empty()) {
    id = packet_type_set.GetId("", index);
  } else {
    id = packet_type_set.GetId(tag, 0);
  }
  ABSL_CHECK(id.IsValid()) << "Internal mediapipe error.";
  return &packet_type_set.Get(id);
}

// Copies a TagMap omitting entries with certain names.
std::shared_ptr<tool::TagMap> RemoveNames(const tool::TagMap& tag_map,
                                          std::set<std::string> names) {
  auto tag_index_names = tag_map.CanonicalEntries();
  for (auto id = tag_map.EndId() - 1; id >= tag_map.BeginId(); --id) {
    std::string name = tag_map.Names()[id.value()];
    if (names.count(name) > 0) {
      tag_index_names.erase(tag_index_names.begin() + id.value());
    }
  }
  return tool::TagMap::Create(tag_index_names).value();
}

// Copies matching entries from another Collection.
template <class CollectionType>
void CopyCollection(const CollectionType& other, CollectionType* result) {
  auto tag_map = result->TagMap();
  for (auto id = tag_map->BeginId(); id != tag_map->EndId(); ++id) {
    auto tag_index = tag_map->TagAndIndexFromId(id);
    auto other_id = other.GetId(tag_index.first, tag_index.second);
    if (other_id.IsValid()) {
      result->Get(id) = other.Get(other_id);
    }
  }
}

// Copies packet types omitting entries that are optional and not provided.
std::unique_ptr<PacketTypeSet> RemoveOmittedPacketTypes(
    const PacketTypeSet& packet_types,
    const std::map<std::string, Packet>& all_side_packets,
    const ValidatedGraphConfig* validated_graph) {
  std::set<std::string> omitted_names;
  for (auto id = packet_types.BeginId(); id != packet_types.EndId(); ++id) {
    std::string name = packet_types.TagMap()->Names()[id.value()];
    if (packet_types.Get(id).IsOptional() &&
        validated_graph->IsExternalSidePacket(name) &&
        all_side_packets.count(name) == 0) {
      omitted_names.insert(name);
    }
  }
  auto tag_map = RemoveNames(*packet_types.TagMap(), omitted_names);
  auto result = std::make_unique<PacketTypeSet>(tag_map);
  CopyCollection(packet_types, result.get());
  return result;
}

}  // namespace

CalculatorNode::CalculatorNode() {}

Timestamp CalculatorNode::SourceProcessOrder(
    const CalculatorContext* cc) const {
  return calculator_->SourceProcessOrder(cc);
}

absl::Status CalculatorNode::Initialize(
    const ValidatedGraphConfig* validated_graph, NodeTypeInfo::NodeRef node_ref,
    InputStreamManager* input_stream_managers,
    OutputStreamManager* output_stream_managers,
    OutputSidePacketImpl* output_side_packets, int* buffer_size_hint,
    std::shared_ptr<ProfilingContext> profiling_context) {
  RET_CHECK(buffer_size_hint) << "buffer_size_hint is NULL";
  validated_graph_ = validated_graph;
  profiling_context_ = profiling_context;

  const CalculatorGraphConfig::Node* node_config;
  if (node_ref.type == NodeTypeInfo::NodeType::CALCULATOR) {
    node_config = &validated_graph_->Config().node(node_ref.index);
    name_ = tool::CanonicalNodeName(validated_graph_->Config(), node_ref.index);
    node_type_info_ = &validated_graph_->CalculatorInfos()[node_ref.index];
  } else if (node_ref.type == NodeTypeInfo::NodeType::PACKET_GENERATOR) {
    const PacketGeneratorConfig& pg_config =
        validated_graph_->Config().packet_generator(node_ref.index);
    name_ = absl::StrCat("__pg_", node_ref.index, "_",
                         pg_config.packet_generator());
    node_type_info_ = &validated_graph_->GeneratorInfos()[node_ref.index];
    node_config = &node_type_info_->Contract().GetWrapperConfig();
  } else {
    return absl::InvalidArgumentError(
        "node_ref is not a calculator or packet generator");
  }

  max_in_flight_ = node_config->max_in_flight();
  max_in_flight_ = max_in_flight_ ? max_in_flight_ : 1;
  if (!node_config->executor().empty()) {
    executor_ = node_config->executor();
  }
  source_layer_ = node_config->source_layer();

  const CalculatorContract& contract = node_type_info_->Contract();

  // TODO Propagate types between calculators when SetAny is used.

  MP_RETURN_IF_ERROR(InitializeOutputSidePackets(
      node_type_info_->OutputSidePacketTypes(), output_side_packets));

  MP_RETURN_IF_ERROR(InitializeInputSidePackets(output_side_packets));

  MP_RETURN_IF_ERROR(
      InitializeOutputStreamHandler(node_config->output_stream_handler(),
                                    node_type_info_->OutputStreamTypes()));
  MP_RETURN_IF_ERROR(InitializeOutputStreams(output_stream_managers));

  calculator_state_ = absl::make_unique<CalculatorState>(
      name_, node_ref.index, node_config->calculator(), *node_config,
      profiling_context_);

  // Inform the scheduler that this node has buffering behavior and that the
  // maximum input queue size should be adjusted accordingly.
  *buffer_size_hint = node_config->buffer_size_hint();

  calculator_context_manager_.Initialize(
      calculator_state_.get(), node_type_info_->InputStreamTypes().TagMap(),
      node_type_info_->OutputStreamTypes().TagMap(),
      /*calculator_run_in_parallel=*/max_in_flight_ > 1);

  // The graph specified InputStreamHandler takes priority.
  const bool graph_specified =
      node_config->input_stream_handler().has_input_stream_handler();
  const bool calc_specified =
      !(node_type_info_->GetInputStreamHandler().empty());

  // Only use calculator ISH if available, and if the graph ISH is not set.
  InputStreamHandlerConfig handler_config;
  const bool use_calc_specified = calc_specified && !graph_specified;
  if (use_calc_specified) {
    *(handler_config.mutable_input_stream_handler()) =
        node_type_info_->GetInputStreamHandler();
    *(handler_config.mutable_options()) =
        node_type_info_->GetInputStreamHandlerOptions();
  }

  // Use calculator or graph specified InputStreamHandler, or the default ISH
  // already set from graph.
  MP_RETURN_IF_ERROR(InitializeInputStreamHandler(
      use_calc_specified ? handler_config : node_config->input_stream_handler(),
      node_type_info_->InputStreamTypes()));

  for (auto& stream : output_stream_handler_->OutputStreams()) {
    stream->Spec()->offset_enabled =
        (contract.GetTimestampOffset() != TimestampDiff::Unset());
    stream->Spec()->offset = contract.GetTimestampOffset();
  }
  input_stream_handler_->SetProcessTimestampBounds(
      contract.GetProcessTimestampBounds());

  return InitializeInputStreams(input_stream_managers, output_stream_managers);
}

absl::Status CalculatorNode::InitializeOutputSidePackets(
    const PacketTypeSet& output_side_packet_types,
    OutputSidePacketImpl* output_side_packets) {
  output_side_packets_ =
      absl::make_unique<OutputSidePacketSet>(output_side_packet_types.TagMap());
  int base_index = node_type_info_->OutputSidePacketBaseIndex();
  RET_CHECK_LE(0, base_index);
  for (CollectionItemId id = output_side_packets_->BeginId();
       id < output_side_packets_->EndId(); ++id) {
    output_side_packets_->GetPtr(id) =
        &output_side_packets[base_index + id.value()];
  }
  return absl::OkStatus();
}

absl::Status CalculatorNode::InitializeInputSidePackets(
    OutputSidePacketImpl* output_side_packets) {
  int base_index = node_type_info_->InputSidePacketBaseIndex();
  RET_CHECK_LE(0, base_index);
  // Set all the mirrors.
  for (CollectionItemId id = node_type_info_->InputSidePacketTypes().BeginId();
       id < node_type_info_->InputSidePacketTypes().EndId(); ++id) {
    int output_side_packet_index =
        validated_graph_->InputSidePacketInfos()[base_index + id.value()]
            .upstream;
    if (output_side_packet_index < 0) {
      // Not generated by a graph node. Comes from an extra side packet
      // provided to the graph.
      continue;
    }
    OutputSidePacketImpl* origin_output_side_packet =
        &output_side_packets[output_side_packet_index];
    VLOG(2) << "Adding mirror for input side packet with id " << id.value()
            << " and flat index " << base_index + id.value()
            << " which will be connected to output side packet with flat index "
            << output_side_packet_index;
    origin_output_side_packet->AddMirror(&input_side_packet_handler_, id);
  }
  return absl::OkStatus();
}

absl::Status CalculatorNode::InitializeOutputStreams(
    OutputStreamManager* output_stream_managers) {
  RET_CHECK(output_stream_managers) << "output_stream_managers is NULL";
  RET_CHECK_LE(0, node_type_info_->OutputStreamBaseIndex());
  OutputStreamManager* current_output_stream_managers =
      &output_stream_managers[node_type_info_->OutputStreamBaseIndex()];
  return output_stream_handler_->InitializeOutputStreamManagers(
      current_output_stream_managers);
}

absl::Status CalculatorNode::InitializeInputStreams(
    InputStreamManager* input_stream_managers,
    OutputStreamManager* output_stream_managers) {
  RET_CHECK(input_stream_managers) << "input_stream_managers is NULL";
  RET_CHECK(output_stream_managers) << "output_stream_managers is NULL";
  RET_CHECK_LE(0, node_type_info_->InputStreamBaseIndex());
  InputStreamManager* current_input_stream_managers =
      &input_stream_managers[node_type_info_->InputStreamBaseIndex()];
  MP_RETURN_IF_ERROR(input_stream_handler_->InitializeInputStreamManagers(
      current_input_stream_managers));

  // Set all the mirrors.
  for (CollectionItemId id = node_type_info_->InputStreamTypes().BeginId();
       id < node_type_info_->InputStreamTypes().EndId(); ++id) {
    int output_stream_index =
        validated_graph_
            ->InputStreamInfos()[node_type_info_->InputStreamBaseIndex() +
                                 id.value()]
            .upstream;
    RET_CHECK_LE(0, output_stream_index);
    OutputStreamManager* origin_output_stream_manager =
        &output_stream_managers[output_stream_index];
    VLOG(2) << "Adding mirror for input stream with id " << id.value()
            << " and flat index "
            << node_type_info_->InputStreamBaseIndex() + id.value()
            << " which will be connected to output stream with flat index "
            << output_stream_index;
    origin_output_stream_manager->AddMirror(input_stream_handler_.get(), id);
  }
  return absl::OkStatus();
}

absl::Status CalculatorNode::InitializeInputStreamHandler(
    const InputStreamHandlerConfig& handler_config,
    const PacketTypeSet& input_stream_types) {
  const ProtoString& input_stream_handler_name =
      handler_config.input_stream_handler();
  RET_CHECK(!input_stream_handler_name.empty());
  MP_ASSIGN_OR_RETURN(
      input_stream_handler_,
      InputStreamHandlerRegistry::CreateByNameInNamespace(
          validated_graph_->Package(), input_stream_handler_name,
          input_stream_types.TagMap(), &calculator_context_manager_,
          handler_config.options(),
          /*calculator_run_in_parallel=*/max_in_flight_ > 1),
      _ << "\"" << input_stream_handler_name
        << "\" is not a registered input stream handler.");

  return absl::OkStatus();
}

absl::Status CalculatorNode::InitializeOutputStreamHandler(
    const OutputStreamHandlerConfig& handler_config,
    const PacketTypeSet& output_stream_types) {
  const ProtoString& output_stream_handler_name =
      handler_config.output_stream_handler();
  RET_CHECK(!output_stream_handler_name.empty());
  MP_ASSIGN_OR_RETURN(
      output_stream_handler_,
      OutputStreamHandlerRegistry::CreateByNameInNamespace(
          validated_graph_->Package(), output_stream_handler_name,
          output_stream_types.TagMap(), &calculator_context_manager_,
          handler_config.options(),
          /*calculator_run_in_parallel=*/max_in_flight_ > 1),
      _ << "\"" << output_stream_handler_name
        << "\" is not a registered output stream handler.");
  return absl::OkStatus();
}

absl::Status CalculatorNode::ConnectShardsToStreams(
    CalculatorContext* calculator_context) {
  RET_CHECK(calculator_context);
  MP_RETURN_IF_ERROR(
      input_stream_handler_->SetupInputShards(&calculator_context->Inputs()));
  return output_stream_handler_->SetupOutputShards(
      &calculator_context->Outputs());
}

void CalculatorNode::SetExecutor(const std::string& executor) {
  absl::MutexLock status_lock(&status_mutex_);
  ABSL_CHECK_LT(status_, kStateOpened);
  executor_ = executor;
}

bool CalculatorNode::Prepared() const {
  absl::MutexLock status_lock(&status_mutex_);
  return status_ >= kStatePrepared;
}

bool CalculatorNode::Opened() const {
  absl::MutexLock status_lock(&status_mutex_);
  return status_ >= kStateOpened;
}

bool CalculatorNode::Active() const {
  absl::MutexLock status_lock(&status_mutex_);
  return status_ >= kStateActive;
}

bool CalculatorNode::Closed() const {
  absl::MutexLock status_lock(&status_mutex_);
  return status_ >= kStateClosed;
}

void CalculatorNode::SetMaxInputStreamQueueSize(int max_queue_size) {
  ABSL_CHECK(input_stream_handler_);
  input_stream_handler_->SetMaxQueueSize(max_queue_size);
}

absl::Status CalculatorNode::PrepareForRun(
    const std::map<std::string, Packet>& all_side_packets,
    const std::map<std::string, Packet>& service_packets,
    std::function<void()> ready_for_open_callback,
    std::function<void()> source_node_opened_callback,
    std::function<void(CalculatorContext*)> schedule_callback,
    std::function<void(absl::Status)> error_callback,
    CounterFactory* counter_factory) {
  RET_CHECK(ready_for_open_callback) << "ready_for_open_callback is NULL";
  RET_CHECK(schedule_callback) << "schedule_callback is NULL";
  RET_CHECK(error_callback) << "error_callback is NULL";
  calculator_state_->ResetBetweenRuns();

  ready_for_open_callback_ = std::move(ready_for_open_callback);
  source_node_opened_callback_ = std::move(source_node_opened_callback);
  input_stream_handler_->PrepareForRun(
      [this]() { CalculatorNode::InputStreamHeadersReady(); },
      [this]() { CalculatorNode::CheckIfBecameReady(); },
      std::move(schedule_callback), error_callback);
  output_stream_handler_->PrepareForRun(error_callback);

  const auto& contract = Contract();
  input_side_packet_types_ = RemoveOmittedPacketTypes(
      contract.InputSidePackets(), all_side_packets, validated_graph_);
  MP_RETURN_IF_ERROR(input_side_packet_handler_.PrepareForRun(
      input_side_packet_types_.get(), all_side_packets,
      [this]() { CalculatorNode::InputSidePacketsReady(); },
      std::move(error_callback)));
  calculator_state_->SetInputSidePackets(
      &input_side_packet_handler_.InputSidePackets());
  calculator_state_->SetOutputSidePackets(output_side_packets_.get());
  calculator_state_->SetCounterFactory(counter_factory);

  for (const auto& svc_req : contract.ServiceRequests()) {
    const auto& req = svc_req.second;
    auto it = service_packets.find(req.Service().key);
    if (it == service_packets.end()) {
      RET_CHECK(req.IsOptional())
          << "required service '" << req.Service().key << "' was not provided";
    } else {
      MP_RETURN_IF_ERROR(
          calculator_state_->SetServicePacket(req.Service(), it->second));
    }
  }

  MP_RETURN_IF_ERROR(calculator_context_manager_.PrepareForRun(std::bind(
      &CalculatorNode::ConnectShardsToStreams, this, std::placeholders::_1)));

  MP_ASSIGN_OR_RETURN(
      auto calculator_factory,
      CalculatorBaseRegistry::CreateByNameInNamespace(
          validated_graph_->Package(), calculator_state_->CalculatorType()));
  calculator_ = calculator_factory->CreateCalculator(
      calculator_context_manager_.GetDefaultCalculatorContext());

  needs_to_close_ = false;

  {
    absl::MutexLock status_lock(&status_mutex_);
    status_ = kStatePrepared;
    scheduling_state_ = kIdle;
    current_in_flight_ = 0;
    input_stream_headers_ready_called_ = false;
    input_side_packets_ready_called_ = false;
    input_stream_headers_ready_ =
        (input_stream_handler_->UnsetHeaderCount() == 0);
    input_side_packets_ready_ =
        (input_side_packet_handler_.MissingInputSidePacketCount() == 0);
  }
  return absl::OkStatus();
}

namespace {
// Returns the Packet sent to an OutputSidePacket, or an empty packet
// if none available.
const Packet GetPacket(const OutputSidePacket& out) {
  auto impl = static_cast<const OutputSidePacketImpl*>(&out);
  return (impl == nullptr) ? Packet() : impl->GetPacket();
}

// Resends the output-side-packets from the previous graph run.
absl::Status ResendSidePackets(CalculatorContext* cc) {
  auto& outs = cc->OutputSidePackets();
  for (CollectionItemId id = outs.BeginId(); id < outs.EndId(); ++id) {
    Packet packet = GetPacket(outs.Get(id));
    if (!packet.IsEmpty()) {
      // OutputSidePacket::Set re-announces the side-packet to its mirrors.
      outs.Get(id).Set(packet);
    }
  }
  return absl::OkStatus();
}
}  // namespace

bool CalculatorNode::OutputsAreConstant(CalculatorContext* cc) {
  if (cc->Inputs().NumEntries() > 0 || cc->Outputs().NumEntries() > 0) {
    return false;
  }
  if (input_side_packet_handler_.InputSidePacketsChanged()) {
    return false;
  }
  return true;
}

absl::Status CalculatorNode::OpenNode() {
  VLOG(2) << "CalculatorNode::OpenNode() for " << DebugName();

  CalculatorContext* default_context =
      calculator_context_manager_.GetDefaultCalculatorContext();
  InputStreamShardSet* inputs = &default_context->Inputs();
  // The upstream calculators may set the headers in the output streams during
  // Calculator::Open(), needs to update the header packets in input stream
  // shards.
  input_stream_handler_->UpdateInputShardHeaders(inputs);
  OutputStreamShardSet* outputs = &default_context->Outputs();
  output_stream_handler_->PrepareOutputs(Timestamp::Unstarted(), outputs);
  calculator_context_manager_.PushInputTimestampToContext(
      default_context, Timestamp::Unstarted());

  absl::Status result;
  if (OutputsAreConstant(default_context)) {
    result = ResendSidePackets(default_context);
  } else {
    MEDIAPIPE_PROFILING(OPEN, default_context);
    LegacyCalculatorSupport::Scoped<CalculatorContext> s(default_context);
    result = calculator_->Open(default_context);
  }

  calculator_context_manager_.PopInputTimestampFromContext(default_context);
  if (IsSource()) {
    // A source node has a dummy input timestamp of 0 for Process(). This input
    // timestamp is not popped until Close() is called.
    calculator_context_manager_.PushInputTimestampToContext(default_context,
                                                            Timestamp(0));
  }

  ABSL_LOG_IF(FATAL, result == tool::StatusStop()) << absl::Substitute(
      "Open() on node \"$0\" returned tool::StatusStop() which should only be "
      "used to signal that a source node is done producing data.",
      DebugName());
  MP_RETURN_IF_ERROR(result).SetPrepend() << absl::Substitute(
      "Calculator::Open() for node \"$0\" failed: ", DebugName());
  needs_to_close_ = true;

  bool offset_enabled = false;
  for (auto& stream : output_stream_handler_->OutputStreams()) {
    offset_enabled = offset_enabled || stream->Spec()->offset_enabled;
  }
  if (offset_enabled && input_stream_handler_->SyncSetCount() > 1) {
    ABSL_LOG(WARNING) << absl::Substitute(
        "Calculator node \"$0\" is configured with multiple input sync-sets "
        "and an output timestamp-offset, which will often conflict due to "
        "the order of packet arrival.  With multiple input sync-sets, use "
        "SetProcessTimestampBounds in place of SetTimestampOffset.",
        DebugName());
  }

  output_stream_handler_->Open(outputs);

  {
    absl::MutexLock status_lock(&status_mutex_);
    status_ = kStateOpened;
  }

  return absl::OkStatus();
}

void CalculatorNode::ActivateNode() {
  absl::MutexLock status_lock(&status_mutex_);
  ABSL_CHECK_EQ(status_, kStateOpened) << DebugName();
  status_ = kStateActive;
}

void CalculatorNode::CloseInputStreams() {
  {
    absl::MutexLock status_lock(&status_mutex_);
    if (status_ == kStateClosed) {
      return;
    }
  }
  VLOG(2) << "Closing node " << DebugName() << " input streams.";

  // Clear the input queues and prevent the upstream nodes from filling them
  // back in.  We may still get ProcessNode called on us after this.
  input_stream_handler_->Close();
}

void CalculatorNode::CloseOutputStreams(OutputStreamShardSet* outputs) {
  {
    absl::MutexLock status_lock(&status_mutex_);
    if (status_ == kStateClosed) {
      return;
    }
  }
  VLOG(2) << "Closing node " << DebugName() << " output streams.";
  output_stream_handler_->Close(outputs);
}

absl::Status CalculatorNode::CloseNode(const absl::Status& graph_status,
                                       bool graph_run_ended) {
  {
    absl::MutexLock status_lock(&status_mutex_);
    RET_CHECK_NE(status_, kStateClosed)
        << "CloseNode() must only be called once.";
  }

  CloseInputStreams();
  CalculatorContext* default_context =
      calculator_context_manager_.GetDefaultCalculatorContext();
  OutputStreamShardSet* outputs = &default_context->Outputs();
  output_stream_handler_->PrepareOutputs(Timestamp::Done(), outputs);
  if (IsSource()) {
    calculator_context_manager_.PopInputTimestampFromContext(default_context);
    calculator_context_manager_.PushInputTimestampToContext(default_context,
                                                            Timestamp::Done());
  }
  calculator_context_manager_.SetGraphStatusInContext(default_context,
                                                      graph_status);

  absl::Status result;

  if (OutputsAreConstant(default_context)) {
    // Do nothing.
    result = absl::OkStatus();
  } else {
    MEDIAPIPE_PROFILING(CLOSE, default_context);
    LegacyCalculatorSupport::Scoped<CalculatorContext> s(default_context);
    result = calculator_->Close(default_context);
  }
  needs_to_close_ = false;

  ABSL_LOG_IF(FATAL, result == tool::StatusStop()) << absl::Substitute(
      "Close() on node \"$0\" returned tool::StatusStop() which should only be "
      "used to signal that a source node is done producing data.",
      DebugName());

  // If the graph run has ended, we are cleaning up after the run and don't
  // need to propagate updates to mirrors, so we can skip this
  // CloseOutputStreams() call. CleanupAfterRun() will close the output
  // streams.
  if (!graph_run_ended) {
    CloseOutputStreams(outputs);
  }

  {
    absl::MutexLock status_lock(&status_mutex_);
    status_ = kStateClosed;
  }

  MP_RETURN_IF_ERROR(result).SetPrepend() << absl::Substitute(
      "Calculator::Close() for node \"$0\" failed: ", DebugName());

  VLOG(2) << "Closed node " << DebugName();
  return absl::OkStatus();
}

void CalculatorNode::CleanupAfterRun(const absl::Status& graph_status) {
  if (needs_to_close_) {
    calculator_context_manager_.PushInputTimestampToContext(
        calculator_context_manager_.GetDefaultCalculatorContext(),
        Timestamp::Done());
    CloseNode(graph_status, /*graph_run_ended=*/true).IgnoreError();
  }
  calculator_ = nullptr;
  // All pending output packets are automatically dropped when calculator
  // context manager destroys all calculator context objects.
  calculator_context_manager_.CleanupAfterRun();

  CloseInputStreams();
  // All output stream shards have been destroyed by calculator context manager.
  CloseOutputStreams(/*outputs=*/nullptr);

  {
    absl::MutexLock lock(&status_mutex_);
    status_ = kStateUninitialized;
    scheduling_state_ = kIdle;
    current_in_flight_ = 0;
  }
}

void CalculatorNode::SchedulingLoop() {
  int max_allowance = 0;
  {
    absl::MutexLock lock(&status_mutex_);
    if (status_ == kStateClosed) {
      scheduling_state_ = kIdle;
      return;
    }
    max_allowance = max_in_flight_ - current_in_flight_;
  }
  while (true) {
    Timestamp input_bound;
    // input_bound is set to a meaningful value iff the latest readiness of the
    // node is kNotReady when ScheduleInvocations() returns.
    input_stream_handler_->ScheduleInvocations(max_allowance, &input_bound);
    if (input_bound != Timestamp::Unset()) {
      // Updates the minimum timestamp for which a new packet could possibly
      // arrive.
      output_stream_handler_->UpdateTaskTimestampBound(input_bound);
    }

    {
      absl::MutexLock lock(&status_mutex_);
      if (scheduling_state_ == kSchedulingPending &&
          current_in_flight_ < max_in_flight_) {
        max_allowance = max_in_flight_ - current_in_flight_;
        scheduling_state_ = kScheduling;
      } else {
        scheduling_state_ = kIdle;
        break;
      }
    }
  }
}

bool CalculatorNode::ReadyForOpen() const {
  absl::MutexLock lock(&status_mutex_);
  return input_stream_headers_ready_ && input_side_packets_ready_;
}

void CalculatorNode::InputStreamHeadersReady() {
  bool ready_for_open = false;
  {
    absl::MutexLock lock(&status_mutex_);
    ABSL_CHECK_EQ(status_, kStatePrepared) << DebugName();
    ABSL_CHECK(!input_stream_headers_ready_called_);
    input_stream_headers_ready_called_ = true;
    input_stream_headers_ready_ = true;
    ready_for_open = input_side_packets_ready_;
  }
  if (ready_for_open) {
    ready_for_open_callback_();
  }
}

void CalculatorNode::InputSidePacketsReady() {
  bool ready_for_open = false;
  {
    absl::MutexLock lock(&status_mutex_);
    ABSL_CHECK_EQ(status_, kStatePrepared) << DebugName();
    ABSL_CHECK(!input_side_packets_ready_called_);
    input_side_packets_ready_called_ = true;
    input_side_packets_ready_ = true;
    ready_for_open = input_stream_headers_ready_;
  }
  if (ready_for_open) {
    ready_for_open_callback_();
  }
}

void CalculatorNode::CheckIfBecameReady() {
  {
    absl::MutexLock lock(&status_mutex_);
    // Doesn't check if status_ is kStateActive since the function can only be
    // invoked by non-source nodes.
    if (status_ != kStateOpened) {
      return;
    }
    if (scheduling_state_ == kIdle && current_in_flight_ < max_in_flight_) {
      scheduling_state_ = kScheduling;
    } else {
      if (scheduling_state_ == kScheduling) {
        // Changes the state to scheduling pending if another thread is doing
        // the scheduling.
        scheduling_state_ = kSchedulingPending;
      }
      return;
    }
  }
  SchedulingLoop();
}

void CalculatorNode::NodeOpened() {
  if (IsSource()) {
    source_node_opened_callback_();
  } else if (input_stream_handler_->NumInputStreams() != 0) {
    // A node with input streams may have received input packets generated by
    // the upstreams nodes' Open() or Process() methods. Check if the node is
    // ready to run.
    CheckIfBecameReady();
  }
}

void CalculatorNode::EndScheduling() {
  {
    absl::MutexLock lock(&status_mutex_);
    if (status_ != kStateOpened && status_ != kStateActive) {
      return;
    }
    --current_in_flight_;
    ABSL_CHECK_GE(current_in_flight_, 0);

    if (scheduling_state_ == kScheduling) {
      // Changes the state to scheduling pending if another thread is doing the
      // scheduling.
      scheduling_state_ = kSchedulingPending;
      return;
    } else if (scheduling_state_ == kSchedulingPending) {
      // Quits when another thread is already doing the scheduling.
      return;
    }
    scheduling_state_ = kScheduling;
  }
  SchedulingLoop();
}

bool CalculatorNode::TryToBeginScheduling() {
  absl::MutexLock lock(&status_mutex_);
  if (current_in_flight_ < max_in_flight_) {
    ++current_in_flight_;
    return true;
  }
  return false;
}

std::string CalculatorNode::DebugInputStreamNames() const {
  return input_stream_handler_->DebugStreamNames();
}

std::string CalculatorNode::DebugName() const {
  ABSL_DCHECK(calculator_state_);
  return calculator_state_->NodeName();
}

// TODO: Split this function.
absl::Status CalculatorNode::ProcessNode(
    CalculatorContext* calculator_context) {
  if (IsSource()) {
    // This is a source Calculator.
    if (Closed()) {
      return absl::OkStatus();
    }

    const Timestamp input_timestamp = calculator_context->InputTimestamp();

    OutputStreamShardSet* outputs = &calculator_context->Outputs();
    output_stream_handler_->PrepareOutputs(input_timestamp, outputs);

    VLOG(2) << "Calling Calculator::Process() for node: " << DebugName();
    absl::Status result;

    {
      MEDIAPIPE_PROFILING(PROCESS, calculator_context);
      LegacyCalculatorSupport::Scoped<CalculatorContext> s(calculator_context);
      result = calculator_->Process(calculator_context);
    }

    bool node_stopped = false;
    if (!result.ok()) {
      if (result == tool::StatusStop()) {
        // Needs to call CloseNode().
        node_stopped = true;
      } else {
        return mediapipe::StatusBuilder(result, MEDIAPIPE_LOC).SetPrepend()
               << absl::Substitute(
                      "Calculator::Process() for node \"$0\" failed: ",
                      DebugName());
      }
    }
    output_stream_handler_->PostProcess(input_timestamp);
    if (node_stopped) {
      MP_RETURN_IF_ERROR(
          CloseNode(absl::OkStatus(), /*graph_run_ended=*/false));
    }
    return absl::OkStatus();
  } else {
    // This is not a source Calculator.
    InputStreamShardSet* const inputs = &calculator_context->Inputs();
    OutputStreamShardSet* const outputs = &calculator_context->Outputs();
    absl::Status result =
        absl::InternalError("Calculator context has no input packets.");

    int num_invocations = calculator_context_manager_.NumberOfContextTimestamps(
        *calculator_context);
    RET_CHECK(num_invocations <= 1 || max_in_flight_ <= 1)
        << "num_invocations:" << num_invocations
        << ", max_in_flight_:" << max_in_flight_;
    for (int i = 0; i < num_invocations; ++i) {
      const Timestamp input_timestamp = calculator_context->InputTimestamp();
      // The node is ready for Process().
      if (input_timestamp.IsAllowedInStream()) {
        input_stream_handler_->FinalizeInputSet(input_timestamp, inputs);
        output_stream_handler_->PrepareOutputs(input_timestamp, outputs);

        VLOG(2) << "Calling Calculator::Process() for node: " << DebugName()
                << " timestamp: " << input_timestamp;

        if (OutputsAreConstant(calculator_context)) {
          // Do nothing.
          result = absl::OkStatus();
        } else {
          MEDIAPIPE_PROFILING(PROCESS, calculator_context);
          LegacyCalculatorSupport::Scoped<CalculatorContext> s(
              calculator_context);
          result = calculator_->Process(calculator_context);
        }

        VLOG(2) << "Called Calculator::Process() for node: " << DebugName()
                << " timestamp: " << input_timestamp;

        // Removes one packet from each shard and progresses to the next input
        // timestamp.
        input_stream_handler_->ClearCurrentInputs(calculator_context);

        // Nodes are allowed to return StatusStop() to cause the termination
        // of the graph. This is different from an error in that it will
        // ensure that all sources will be closed and that packets in input
        // streams will be processed before the graph is terminated.
        if (!result.ok() && result != tool::StatusStop()) {
          return mediapipe::StatusBuilder(result, MEDIAPIPE_LOC).SetPrepend()
                 << absl::Substitute(
                        "Calculator::Process() for node \"$0\" failed: ",
                        DebugName());
        }
        output_stream_handler_->PostProcess(input_timestamp);
        if (result == tool::StatusStop()) {
          return result;
        }
      } else if (input_timestamp == Timestamp::Done()) {
        // Some or all the input streams are closed and there are not enough
        // open input streams for Process(). So this node needs to be closed
        // too.
        // If the streams are closed, there shouldn't be more input.
        ABSL_CHECK_EQ(calculator_context_manager_.NumberOfContextTimestamps(
                          *calculator_context),
                      1);
        return CloseNode(absl::OkStatus(), /*graph_run_ended=*/false);
      } else {
        RET_CHECK_FAIL()
            << "Invalid input timestamp in ProcessNode(). timestamp: "
            << input_timestamp;
      }
    }
    return result;
  }
}

void CalculatorNode::SetQueueSizeCallbacks(
    InputStreamManager::QueueSizeCallback becomes_full_callback,
    InputStreamManager::QueueSizeCallback becomes_not_full_callback) {
  ABSL_CHECK(input_stream_handler_);
  input_stream_handler_->SetQueueSizeCallbacks(
      std::move(becomes_full_callback), std::move(becomes_not_full_callback));
}

}  // namespace mediapipe
