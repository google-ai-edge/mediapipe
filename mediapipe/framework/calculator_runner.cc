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
//
// Definitions for CalculatorRunner.

#include "mediapipe/framework/calculator_runner.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

const char CalculatorRunner::kSourcePrefix[] = "source_for_";
const char CalculatorRunner::kSinkPrefix[] = "sink_for_";

namespace {

// Calculator generating a stream with the given contents.
// Inputs: none
// Outputs: 1, with the contents provided via the input side packet.
// Input side packets: 1, pointing to CalculatorRunner::StreamContents.
class CalculatorRunnerSourceCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->InputSidePackets()
        .Index(0)
        .Set<const CalculatorRunner::StreamContents*>();
    cc->Outputs().Index(0).SetAny();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    const auto* contents = cc->InputSidePackets()
                               .Index(0)
                               .Get<const CalculatorRunner::StreamContents*>();
    // Set the header and packets of the output stream.
    cc->Outputs().Index(0).SetHeader(contents->header);
    for (const Packet& packet : contents->packets) {
      cc->Outputs().Index(0).AddPacket(packet);
    }
    return absl::OkStatus();
  }
  absl::Status Process(CalculatorContext* cc) override {
    return tool::StatusStop();
  }
};
REGISTER_CALCULATOR(CalculatorRunnerSourceCalculator);

// Calculator recording the contents of a stream.
// Inputs: 1, with the contents written to the input side packet.
// Outputs: none
// Input side packets: 1, pointing to CalculatorRunner::StreamContents.
class CalculatorRunnerSinkCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->InputSidePackets().Index(0).Set<CalculatorRunner::StreamContents*>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    contents_ = cc->InputSidePackets()
                    .Index(0)
                    .Get<CalculatorRunner::StreamContents*>();
    contents_->header = cc->Inputs().Index(0).Header();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    contents_->packets.push_back(cc->Inputs().Index(0).Value());
    return absl::OkStatus();
  }

 private:
  CalculatorRunner::StreamContents* contents_ = nullptr;
};
REGISTER_CALCULATOR(CalculatorRunnerSinkCalculator);

}  // namespace

CalculatorRunner::CalculatorRunner(
    const CalculatorGraphConfig::Node& node_config) {
  MEDIAPIPE_CHECK_OK(InitializeFromNodeConfig(node_config));
}

absl::Status CalculatorRunner::InitializeFromNodeConfig(
    const CalculatorGraphConfig::Node& node_config) {
  node_config_ = node_config;

  if (node_config_.external_input_size() > 0) {
    RET_CHECK_EQ(0, node_config_.input_side_packet_size())
        << "Only one of input_side_packet or (deprecated) external_input can "
           "be set.";
    node_config_.mutable_external_input()->Swap(
        node_config_.mutable_input_side_packet());
  }

  ASSIGN_OR_RETURN(auto input_map,
                   tool::TagMap::Create(node_config_.input_stream()));
  inputs_ = absl::make_unique<StreamContentsSet>(input_map);

  ASSIGN_OR_RETURN(auto output_map,
                   tool::TagMap::Create(node_config_.output_stream()));
  outputs_ = absl::make_unique<StreamContentsSet>(output_map);

  ASSIGN_OR_RETURN(auto input_side_map,
                   tool::TagMap::Create(node_config_.input_side_packet()));
  input_side_packets_ = absl::make_unique<PacketSet>(input_side_map);

  ASSIGN_OR_RETURN(auto output_side_map,
                   tool::TagMap::Create(node_config_.output_side_packet()));
  output_side_packets_ = absl::make_unique<PacketSet>(output_side_map);

  return absl::OkStatus();
}

CalculatorRunner::CalculatorRunner(const std::string& calculator_type,
                                   const CalculatorOptions& options) {
  node_config_.set_calculator(calculator_type);
  *node_config_.mutable_options() = options;
  log_calculator_proto_ = true;
}

#if !defined(MEDIAPIPE_PROTO_LITE)
CalculatorRunner::CalculatorRunner(const std::string& node_config_string) {
  CalculatorGraphConfig::Node node_config;
  CHECK(
      proto_ns::TextFormat::ParseFromString(node_config_string, &node_config));
  MEDIAPIPE_CHECK_OK(InitializeFromNodeConfig(node_config));
}

CalculatorRunner::CalculatorRunner(const std::string& calculator_type,
                                   const std::string& options_string,
                                   int num_inputs, int num_outputs,
                                   int num_side_packets) {
  node_config_.set_calculator(calculator_type);
  CHECK(proto_ns::TextFormat::ParseFromString(options_string,
                                              node_config_.mutable_options()));
  SetNumInputs(num_inputs);
  SetNumOutputs(num_outputs);
  SetNumInputSidePackets(num_side_packets);
  // Reset log_calculator_proto to false, since it was set to true by
  // SetNum*() calls above.  This constructor is not deprecated but is
  // currently implemented in terms of deprecated functions.
  log_calculator_proto_ = false;
}
#endif

CalculatorRunner::~CalculatorRunner() {}

void CalculatorRunner::SetNumInputs(int n) {
  tool::TagAndNameInfo info;
  for (int i = 0; i < n; ++i) {
    info.names.push_back(absl::StrCat("input_", i));
  }
  InitializeInputs(info);
}

void CalculatorRunner::SetNumOutputs(int n) {
  tool::TagAndNameInfo info;
  for (int i = 0; i < n; ++i) {
    info.names.push_back(absl::StrCat("output_", i));
  }
  InitializeOutputs(info);
}

void CalculatorRunner::SetNumInputSidePackets(int n) {
  tool::TagAndNameInfo info;
  for (int i = 0; i < n; ++i) {
    info.names.push_back(absl::StrCat("side_packet_", i));
  }
  InitializeInputSidePackets(info);
}

void CalculatorRunner::InitializeInputs(const tool::TagAndNameInfo& info) {
  CHECK(graph_ == nullptr);
  MEDIAPIPE_CHECK_OK(
      tool::SetFromTagAndNameInfo(info, node_config_.mutable_input_stream()));
  inputs_.reset(new StreamContentsSet(info));
  log_calculator_proto_ = true;
}

void CalculatorRunner::InitializeOutputs(const tool::TagAndNameInfo& info) {
  CHECK(graph_ == nullptr);
  MEDIAPIPE_CHECK_OK(
      tool::SetFromTagAndNameInfo(info, node_config_.mutable_output_stream()));
  outputs_.reset(new StreamContentsSet(info));
  log_calculator_proto_ = true;
}

void CalculatorRunner::InitializeInputSidePackets(
    const tool::TagAndNameInfo& info) {
  CHECK(graph_ == nullptr);
  MEDIAPIPE_CHECK_OK(tool::SetFromTagAndNameInfo(
      info, node_config_.mutable_input_side_packet()));
  input_side_packets_.reset(new PacketSet(info));
  log_calculator_proto_ = true;
}

mediapipe::Counter* CalculatorRunner::GetCounter(const std::string& name) {
  return graph_->GetCounterFactory()->GetCounter(name);
}

std::map<std::string, int64> CalculatorRunner::GetCountersValues() {
  return graph_->GetCounterFactory()->GetCounterSet()->GetCountersValues();
}

absl::Status CalculatorRunner::BuildGraph() {
  if (graph_ != nullptr) {
    // The graph was already built.
    return absl::OkStatus();
  }
  RET_CHECK(inputs_) << "The inputs were not initialized.";
  RET_CHECK(outputs_) << "The outputs were not initialized.";
  RET_CHECK(input_side_packets_)
      << "The input side packets were not initialized.";

  CalculatorGraphConfig config;
  // Add the calculator node.
  *(config.add_node()) = node_config_;

  for (int i = 0; i < node_config_.input_stream_size(); ++i) {
    std::string name;
    std::string tag;
    int index;
    MP_RETURN_IF_ERROR(tool::ParseTagIndexName(node_config_.input_stream(i),
                                               &tag, &index, &name));
    // Add a source for each input stream.
    auto* node = config.add_node();
    node->set_calculator("CalculatorRunnerSourceCalculator");
    node->add_output_stream(name);
    node->add_input_side_packet(absl::StrCat(kSourcePrefix, name));
  }
  for (int i = 0; i < node_config_.output_stream_size(); ++i) {
    std::string name;
    std::string tag;
    int index;
    MP_RETURN_IF_ERROR(tool::ParseTagIndexName(node_config_.output_stream(i),
                                               &tag, &index, &name));
    // Add a sink for each output stream.
    auto* node = config.add_node();
    node->set_calculator("CalculatorRunnerSinkCalculator");
    node->add_input_stream(name);
    node->add_input_side_packet(absl::StrCat(kSinkPrefix, name));
  }
  config.set_num_threads(1);

  if (log_calculator_proto_) {
#if defined(MEDIAPIPE_PROTO_LITE)
    LOG(INFO) << "Please initialize CalculatorRunner using the recommended "
                 "constructor:\n    CalculatorRunner runner(node_config);";
#else
    std::string config_string;
    proto_ns::TextFormat::Printer printer;
    printer.SetInitialIndentLevel(4);
    printer.PrintToString(node_config_, &config_string);
    LOG(INFO) << "Please initialize CalculatorRunner using the recommended "
                 "constructor:\n    CalculatorRunner runner(R\"(\n"
              << config_string << "\n    )\");";
#endif
  }

  graph_ = absl::make_unique<CalculatorGraph>();
  MP_RETURN_IF_ERROR(graph_->Initialize(config));
  return absl::OkStatus();
}

absl::Status CalculatorRunner::Run() {
  MP_RETURN_IF_ERROR(BuildGraph());
  // Set the input side packets for the sources.
  std::map<std::string, Packet> input_side_packets;
  int positional_index = -1;
  for (int i = 0; i < node_config_.input_stream_size(); ++i) {
    std::string name;
    std::string tag;
    int index;
    MP_RETURN_IF_ERROR(tool::ParseTagIndexName(node_config_.input_stream(i),
                                               &tag, &index, &name));
    const CalculatorRunner::StreamContents* contents;
    if (index == -1) {
      // positional_index considers the case when the tag is empty, which is
      // always the case when index == -1. If we ever support indices for
      // non-empty tags ("ABC:input1" and "ABC:input2" with automatic indices),
      // this should be changed to use a map insted.
      contents = &inputs_->Get(tag, ++positional_index);
    } else {
      contents = &inputs_->Get(tag, index);
    }
    input_side_packets.emplace(absl::StrCat(kSourcePrefix, name),
                               Adopt(new auto(contents)));
  }
  // Set the input side packets for the calculator.
  positional_index = -1;
  for (int i = 0; i < node_config_.input_side_packet_size(); ++i) {
    std::string name;
    std::string tag;
    int index;
    MP_RETURN_IF_ERROR(tool::ParseTagIndexName(
        node_config_.input_side_packet(i), &tag, &index, &name));
    const Packet* packet;
    if (index == -1) {
      packet = &input_side_packets_->Get(tag, ++positional_index);
    } else {
      packet = &input_side_packets_->Get(tag, index);
    }
    input_side_packets.emplace(name, *packet);
  }
  // Set the input side packets for the sinks.
  positional_index = -1;
  for (int i = 0; i < node_config_.output_stream_size(); ++i) {
    std::string name;
    std::string tag;
    int index;
    MP_RETURN_IF_ERROR(tool::ParseTagIndexName(node_config_.output_stream(i),
                                               &tag, &index, &name));
    CalculatorRunner::StreamContents* contents;
    if (index == -1) {
      contents = &outputs_->Get(tag, ++positional_index);
    } else {
      contents = &outputs_->Get(tag, index);
    }
    // Clear |contents| because Run() may be called multiple times.
    *contents = CalculatorRunner::StreamContents();
    input_side_packets.emplace(absl::StrCat(kSinkPrefix, name),
                               Adopt(new auto(contents)));
  }
  MP_RETURN_IF_ERROR(graph_->Run(input_side_packets));

  positional_index = -1;
  for (int i = 0; i < node_config_.output_side_packet_size(); ++i) {
    std::string name;
    std::string tag;
    int index;
    MP_RETURN_IF_ERROR(tool::ParseTagIndexName(
        node_config_.output_side_packet(i), &tag, &index, &name));
    Packet& contents = output_side_packets_->Get(
        tag, (index == -1) ? ++positional_index : index);
    ASSIGN_OR_RETURN(contents, graph_->GetOutputSidePacket(name));
  }
  return absl::OkStatus();
}

}  // namespace mediapipe
