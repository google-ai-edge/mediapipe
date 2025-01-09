#include "mediapipe/framework/tool/switch/graph_processor.h"

#include "absl/synchronization/mutex.h"

namespace mediapipe {

// TODO: add support for input and output side packets.
absl::Status GraphProcessor::Initialize(CalculatorGraphConfig graph_config) {
  graph_config_ = graph_config;

  MP_ASSIGN_OR_RETURN(graph_input_map_,
                      tool::TagMap::Create(graph_config_.input_stream()));
  MP_ASSIGN_OR_RETURN(graph_output_map_,
                      tool::TagMap::Create(graph_config_.output_stream()));
  return absl::OkStatus();
}

absl::Status GraphProcessor::AddPacket(CollectionItemId id, Packet packet) {
  absl::MutexLock lock(&graph_mutex_);
  const std::string& stream_name = graph_input_map_->Names().at(id.value());
  return graph_->AddPacketToInputStream(stream_name, packet);
}

std::shared_ptr<tool::TagMap> GraphProcessor::InputTags() {
  return graph_input_map_;
}

absl::Status GraphProcessor::SendPacket(CollectionItemId id, Packet packet) {
  MP_RETURN_IF_ERROR(WaitUntilInitialized());
  auto it = consumer_ids_.find(id);
  if (it == consumer_ids_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Consumer stream not found: ", id.value()));
  }
  return consumer_->AddPacket(it->second, packet);
}

void GraphProcessor::SetConsumer(PacketConsumer* consumer) {
  absl::MutexLock lock(&graph_mutex_);
  consumer_ = consumer;
  auto input_map = consumer_->InputTags();
  for (auto id = input_map->BeginId(); id != input_map->EndId(); ++id) {
    auto tag_index = input_map->TagAndIndexFromId(id);
    auto stream_id = graph_input_map_->GetId(tag_index.first, tag_index.second);
    consumer_ids_[stream_id] = id;
  }
}

absl::Status GraphProcessor::ObserveGraph() {
  for (auto id = graph_output_map_->BeginId(); id != graph_output_map_->EndId();
       ++id) {
    std::string stream_name = graph_output_map_->Names().at(id.value());
    MP_RETURN_IF_ERROR(graph_->ObserveOutputStream(
        stream_name,
        [this, id](const Packet& packet) { return SendPacket(id, packet); },
        true));
  }
  return absl::OkStatus();
}

absl::Status GraphProcessor::WaitUntilInitialized() {
  absl::MutexLock lock(&graph_mutex_);
  auto is_initialized = [this]() ABSL_SHARED_LOCKS_REQUIRED(graph_mutex_) {
    return graph_ != nullptr && consumer_ != nullptr;
  };
  graph_mutex_.AwaitWithTimeout(absl::Condition(&is_initialized),
                                absl::Seconds(4));
  RET_CHECK(is_initialized()) << "GraphProcessor initialization timed out.";
  return absl::OkStatus();
}

absl::Status GraphProcessor::Start() {
  absl::MutexLock lock(&graph_mutex_);
  graph_ = std::make_unique<CalculatorGraph>();

  // The graph is validated here with its specified inputs and output.
  MP_RETURN_IF_ERROR(graph_->Initialize(graph_config_, side_packets_));
  MP_RETURN_IF_ERROR(ObserveGraph());
  MP_RETURN_IF_ERROR(graph_->StartRun({}));
  return absl::OkStatus();
}

absl::Status GraphProcessor::Shutdown() {
  absl::MutexLock lock(&graph_mutex_);
  if (!graph_) {
    return absl::OkStatus();
  }
  MP_RETURN_IF_ERROR(graph_->CloseAllPacketSources());
  MP_RETURN_IF_ERROR(graph_->WaitUntilDone());
  graph_ = nullptr;
  return absl::OkStatus();
}

absl::Status GraphProcessor::WaitUntilIdle() {
  absl::MutexLock lock(&graph_mutex_);
  return graph_->WaitUntilIdle();
}

// TODO
absl::Status GraphProcessor::SetSidePacket(CollectionItemId id, Packet packet) {
  return absl::OkStatus();
}
// TODO
std::shared_ptr<tool::TagMap> GraphProcessor::SideInputTags() {
  return nullptr;
}
// TODO
void GraphProcessor::SetSideConsumer(SidePacketConsumer* consumer) {}

}  // namespace mediapipe
