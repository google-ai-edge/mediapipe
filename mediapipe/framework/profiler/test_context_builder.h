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

#ifndef MEDIAPIPE_FRAMEWORK_PROFILER_TEST_CONTEXT_BUILDER_H_
#define MEDIAPIPE_FRAMEWORK_PROFILER_TEST_CONTEXT_BUILDER_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/mediapipe_options.pb.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/framework/tool/tag_map.h"
#include "mediapipe/framework/tool/tag_map_helper.h"

namespace mediapipe {

using tool::TagMap;

// A builder for the CalculatorContext for testing a calculator node.
class TestContextBuilder {
  // An InputStreamHandler to initialize and fill input streams.
  class InputStreamWriter : public InputStreamHandler {
   public:
    using InputStreamHandler::InputStreamHandler;
    void set_packets(const std::vector<Packet>& packets) { packets_ = packets; }
    NodeReadiness GetNodeReadiness(Timestamp* min_stream_timestamp) {
      return NodeReadiness::kReadyForProcess;
    }
    void FillInputSet(Timestamp input_timestamp,
                      InputStreamShardSet* input_set) override {
      for (auto id = input_set->BeginId(); id < input_set->EndId(); ++id) {
        Packet packet = packets_[id.value()];
        AddPacketToShard(&input_set->Get(id), std::move(packet), false);
      }
    }
    std::vector<Packet> packets_;
  };

 public:
  TestContextBuilder() = default;
  TestContextBuilder(const std::string& node_name, int node_id,
                     const std::vector<std::string>& inputs,
                     const std::vector<std::string>& outputs) {
    Init(node_name, node_id, inputs, outputs);
  }

  // Initializes the input and output specs of the calculator node.
  // Also, creates the default calculator context for the calculator node.
  void Init(const std::string& node_name, int node_id,
            const std::vector<std::string>& inputs,
            const std::vector<std::string>& outputs) {
    static auto packet_type = new PacketType;
    packet_type->Set<std::string>();

    state_ = absl::make_unique<CalculatorState>(
        node_name, node_id, "PCalculator", CalculatorGraphConfig::Node(),
        nullptr);
    input_map_ = tool::CreateTagMap(inputs).value();
    output_map_ = tool::CreateTagMap(outputs).value();
    input_handler_ = absl::make_unique<InputStreamWriter>(
        input_map_, nullptr, MediaPipeOptions(), false);
    input_managers_.reset(new InputStreamManager[input_map_->NumEntries()]);
    for (auto id = input_map_->BeginId(); id < input_map_->EndId(); ++id) {
      MEDIAPIPE_CHECK_OK(input_managers_[id.value()].Initialize(
          input_map_->Names()[id.value()], packet_type, false));
    }
    MEDIAPIPE_CHECK_OK(
        input_handler_->InitializeInputStreamManagers(input_managers_.get()));
    for (auto id = output_map_->BeginId(); id < output_map_->EndId(); ++id) {
      static auto packet_type_ = new PacketType;
      packet_type_->Set<std::string>();
      OutputStreamSpec spec;
      spec.name = output_map_->Names()[id.value()];
      spec.packet_type = packet_type;
      spec.error_callback = [](const absl::Status& status) {
        ABSL_LOG(ERROR) << status;
      };
      output_specs_[spec.name] = spec;
    }
    context_ = CreateCalculatorContext();
  }

  // Initializes the input and output streams of a calculator context.
  std::unique_ptr<CalculatorContext> CreateCalculatorContext() {
    auto result = absl::make_unique<CalculatorContext>(state_.get(), input_map_,
                                                       output_map_);
    MEDIAPIPE_CHECK_OK(input_handler_->SetupInputShards(&result->Inputs()));
    for (auto id = output_map_->BeginId(); id < output_map_->EndId(); ++id) {
      auto& out_stream = result->Outputs().Get(id);
      const std::string& stream_name = output_map_->Names()[id.value()];
      out_stream.SetSpec(&output_specs_[stream_name]);
    }
    return result;
  }

  // Returns the calculator context.
  CalculatorContext* get() { return context_.get(); }

  // Resets the calculator context.
  void Clear() { context_ = CreateCalculatorContext(); }

  // Writes packets to the input streams of a calculator context.
  void AddInputs(const std::vector<Packet>& packets) {
    Timestamp input_timestamp = GetTimestamp(packets);
    input_handler_->set_packets(packets);
    input_handler_->FillInputSet(input_timestamp, &context_->Inputs());
    CalculatorContextManager().PushInputTimestampToContext(context_.get(),
                                                           input_timestamp);
  }

  // Writes packets to the output streams of a calculator context.
  void AddOutputs(const std::vector<std::vector<Packet>>& packets) {
    auto& out_map = context_->Outputs().TagMap();
    for (auto id = out_map->BeginId(); id < out_map->EndId(); ++id) {
      auto& out_stream = context_->Outputs().Get(id);
      for (const Packet& packet : packets[id.value()]) {
        out_stream.AddPacket(packet);
      }
    }
  }

  // Returns the Timestamp of the first non-empty packet.
  static Timestamp GetTimestamp(const std::vector<Packet>& packets) {
    for (const Packet& packet : packets) {
      if (!packet.IsEmpty()) {
        return packet.Timestamp();
      }
    }
    return Timestamp();
  }

  std::unique_ptr<CalculatorState> state_;
  std::unique_ptr<InputStreamWriter> input_handler_;
  std::unique_ptr<InputStreamManager[]> input_managers_;
  std::shared_ptr<TagMap> input_map_;
  std::shared_ptr<TagMap> output_map_;
  std::map<std::string, OutputStreamSpec> output_specs_;
  std::unique_ptr<CalculatorContext> context_;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_PROFILER_TEST_CONTEXT_BUILDER_H_
