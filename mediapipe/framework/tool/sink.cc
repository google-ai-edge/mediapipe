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
// Forked from mediapipe/framework/tool/source.proto.
// The forked proto must remain identical to the original proto and should be
// ONLY used by mediapipe open source project.

#include "mediapipe/framework/tool/sink.h"

#include <memory>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "mediapipe/calculators/internal/callback_packet_calculator.pb.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_base.h"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/calculator_registry.h"
#include "mediapipe/framework/input_stream.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/source_location.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/framework/tool/name_util.h"

namespace mediapipe {

namespace tool {
namespace {
// Produces an output packet with the PostStream timestamp containing the
// input side packet.
class MediaPipeInternalSidePacketToPacketStreamCalculator
    : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->InputSidePackets().Index(0).SetAny();
    cc->Outputs().Index(0).SetSameAs(&cc->InputSidePackets().Index(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    cc->Outputs().Index(0).AddPacket(
        cc->InputSidePackets().Index(0).At(Timestamp::PostStream()));
    cc->Outputs().Index(0).Close();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    // The framework treats this calculator as a source calculator.
    return ::mediapipe::tool::StatusStop();
  }
};
REGISTER_CALCULATOR(MediaPipeInternalSidePacketToPacketStreamCalculator);
}  // namespace

void AddVectorSink(const std::string& stream_name,  //
                   CalculatorGraphConfig* config,   //
                   std::vector<Packet>* dumped_data) {
  CHECK(config);
  CHECK(dumped_data);

  std::string input_side_packet_name;
  tool::AddCallbackCalculator(stream_name, config, &input_side_packet_name,
                              /*use_std_function=*/true);

  auto* node = config->add_node();
  node->set_name(GetUnusedNodeName(
      *config, absl::StrCat("callback_packet_calculator_that_generators_",
                            input_side_packet_name)));
  node->set_calculator("CallbackPacketCalculator");
  node->add_output_side_packet(input_side_packet_name);
  CallbackPacketCalculatorOptions* options =
      node->mutable_options()->MutableExtension(
          CallbackPacketCalculatorOptions::ext);
  options->set_type(CallbackPacketCalculatorOptions::VECTOR_PACKET);
  char address[17];
  int written = snprintf(address, sizeof(address), "%p", dumped_data);
  CHECK(written > 0 && written < sizeof(address));
  options->set_pointer(address);
}

void AddPostStreamPacketSink(const std::string& stream_name,
                             CalculatorGraphConfig* config,
                             Packet* post_stream_packet) {
  CHECK(config);
  CHECK(post_stream_packet);

  std::string input_side_packet_name;
  tool::AddCallbackCalculator(stream_name, config, &input_side_packet_name,
                              /*use_std_function=*/true);
  auto* node = config->add_node();
  node->set_name(GetUnusedNodeName(
      *config, absl::StrCat("callback_packet_calculator_that_generators_",
                            input_side_packet_name)));
  node->set_calculator("CallbackPacketCalculator");
  node->add_output_side_packet(input_side_packet_name);
  CallbackPacketCalculatorOptions* options =
      node->mutable_options()->MutableExtension(
          CallbackPacketCalculatorOptions::ext);
  options->set_type(CallbackPacketCalculatorOptions::POST_STREAM_PACKET);
  char address[17];
  int written = snprintf(address, sizeof(address), "%p", post_stream_packet);
  CHECK(written > 0 && written < sizeof(address));
  options->set_pointer(address);
}

void AddSidePacketSink(const std::string& side_packet_name,
                       CalculatorGraphConfig* config, Packet* dumped_packet) {
  CHECK(config);
  CHECK(dumped_packet);

  CalculatorGraphConfig::Node* conversion_node = config->add_node();
  const std::string node_name = GetUnusedNodeName(
      *config,
      absl::StrCat("calculator_converts_side_packet_", side_packet_name));
  conversion_node->set_name(node_name);
  conversion_node->set_calculator(
      "MediaPipeInternalSidePacketToPacketStreamCalculator");
  conversion_node->add_input_side_packet(
      GetUnusedSidePacketName(*config, side_packet_name));

  const std::string output_stream_name =
      absl::StrCat(node_name, "_output_stream");
  conversion_node->add_output_stream(output_stream_name);
  AddPostStreamPacketSink(output_stream_name, config, dumped_packet);
}

void AddCallbackCalculator(const std::string& stream_name,
                           CalculatorGraphConfig* config,
                           std::string* callback_side_packet_name,
                           bool use_std_function) {
  CHECK(config);
  CHECK(callback_side_packet_name);
  CalculatorGraphConfig::Node* sink_node = config->add_node();
  sink_node->set_name(GetUnusedNodeName(
      *config,
      absl::StrCat("callback_calculator_that_collects_stream_", stream_name)));
  sink_node->set_calculator("CallbackCalculator");
  sink_node->add_input_stream(stream_name);

  const std::string input_side_packet_name =
      GetUnusedSidePacketName(*config, absl::StrCat(stream_name, "_callback"));
  *callback_side_packet_name = input_side_packet_name;
  if (use_std_function) {
    // Uses tag "CALLBACK" if the input side packet contains a std::function.
    sink_node->add_input_side_packet(
        absl::StrCat("CALLBACK:", input_side_packet_name));
  } else {
    LOG(FATAL) << "AddCallbackCalculator must use std::function";
  }
}

void AddMultiStreamCallback(
    const std::vector<std::string>& streams,
    std::function<void(const std::vector<Packet>&)> callback,
    CalculatorGraphConfig* config,
    std::pair<std::string, Packet>* side_packet) {
  CHECK(config);
  CHECK(side_packet);
  CalculatorGraphConfig::Node* sink_node = config->add_node();
  const std::string name = GetUnusedNodeName(
      *config, absl::StrCat("multi_callback_", absl::StrJoin(streams, "_")));
  sink_node->set_name(name);
  sink_node->set_calculator("CallbackCalculator");
  for (const auto& stream_name : streams) {
    sink_node->add_input_stream(stream_name);
  }

  const std::string input_side_packet_name =
      GetUnusedSidePacketName(*config, absl::StrCat(name, "_callback"));
  side_packet->first = input_side_packet_name;
  sink_node->add_input_side_packet(
      absl::StrCat("VECTOR_CALLBACK:", input_side_packet_name));

  side_packet->second =
      MakePacket<std::function<void(const std::vector<Packet>&)>>(
          std::move(callback));
}

void AddCallbackWithHeaderCalculator(const std::string& stream_name,
                                     const std::string& stream_header,
                                     CalculatorGraphConfig* config,
                                     std::string* callback_side_packet_name,
                                     bool use_std_function) {
  CHECK(config);
  CHECK(callback_side_packet_name);
  CalculatorGraphConfig::Node* sink_node = config->add_node();
  sink_node->set_name(GetUnusedNodeName(
      *config,
      absl::StrCat("callback_calculator_that_collects_stream_and_header_",
                   stream_name, "_", stream_header)));
  sink_node->set_calculator("CallbackWithHeaderCalculator");
  sink_node->add_input_stream(absl::StrCat("INPUT:", stream_name));
  sink_node->add_input_stream(absl::StrCat("HEADER:", stream_header));

  const std::string input_side_packet_name = GetUnusedSidePacketName(
      *config, absl::StrCat(stream_name, "_", stream_header, "_callback"));
  *callback_side_packet_name = input_side_packet_name;

  if (use_std_function) {
    // Uses tag "CALLBACK" if the input side packet contains a std::function.
    sink_node->add_input_side_packet(
        absl::StrCat("CALLBACK:", input_side_packet_name));
  } else {
    LOG(FATAL) << "AddCallbackWithHeaderCalculator must use std::function";
  }
}

// CallbackCalculator

// static
::mediapipe::Status CallbackCalculator::GetContract(CalculatorContract* cc) {
  bool allow_multiple_streams = false;
  // If the input side packet is specified using tag "CALLBACK" it must contain
  // a std::function, which may be generated by CallbackPacketCalculator.

  if (cc->InputSidePackets().HasTag("CALLBACK")) {
    cc->InputSidePackets()
        .Tag("CALLBACK")
        .Set<std::function<void(const Packet&)>>();
  } else if (cc->InputSidePackets().HasTag("VECTOR_CALLBACK")) {
    cc->InputSidePackets()
        .Tag("VECTOR_CALLBACK")
        .Set<std::function<void(const std::vector<Packet>&)>>();
    allow_multiple_streams = true;
  } else {
    return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "InputSidePackets must use tags.";
  }

  int count = allow_multiple_streams ? cc->Inputs().NumEntries("") : 1;
  for (int i = 0; i < count; ++i) {
    cc->Inputs().Index(i).SetAny();
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status CallbackCalculator::Open(CalculatorContext* cc) {
  if (cc->InputSidePackets().HasTag("CALLBACK")) {
    callback_ = cc->InputSidePackets()
                    .Tag("CALLBACK")
                    .Get<std::function<void(const Packet&)>>();
  } else if (cc->InputSidePackets().HasTag("VECTOR_CALLBACK")) {
    vector_callback_ =
        cc->InputSidePackets()
            .Tag("VECTOR_CALLBACK")
            .Get<std::function<void(const std::vector<Packet>&)>>();
  } else {
    LOG(FATAL) << "InputSidePackets must use tags.";
  }
  if (callback_ == nullptr && vector_callback_ == nullptr) {
    return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "missing callback.";
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status CallbackCalculator::Process(CalculatorContext* cc) {
  if (callback_) {
    callback_(cc->Inputs().Index(0).Value());
  } else if (vector_callback_) {
    int count = cc->Inputs().NumEntries("");
    std::vector<Packet> packets;
    packets.reserve(count);
    for (int i = 0; i < count; ++i) {
      packets.push_back(cc->Inputs().Index(i).Value());
    }
    vector_callback_(packets);
  }
  return ::mediapipe::OkStatus();
}

REGISTER_CALCULATOR(CallbackCalculator);

// CallbackWithHeaderCalculator

// static
::mediapipe::Status CallbackWithHeaderCalculator::GetContract(
    CalculatorContract* cc) {
  cc->Inputs().Tag("INPUT").SetAny();
  cc->Inputs().Tag("HEADER").SetAny();

  if (cc->InputSidePackets().UsesTags()) {
    CHECK(cc->InputSidePackets().HasTag("CALLBACK"));
    cc->InputSidePackets()
        .Tag("CALLBACK")
        .Set<std::function<void(const Packet&, const Packet&)>>();
  } else {
    return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "InputSidePackets must use tags.";
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status CallbackWithHeaderCalculator::Open(CalculatorContext* cc) {
  if (cc->InputSidePackets().UsesTags()) {
    callback_ = cc->InputSidePackets()
                    .Tag("CALLBACK")
                    .Get<std::function<void(const Packet&, const Packet&)>>();
  } else {
    LOG(FATAL) << "InputSidePackets must use tags.";
  }
  if (callback_ == nullptr) {
    return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "callback is nullptr.";
  }
  if (!cc->Inputs().HasTag("INPUT")) {
    return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "No input stream connected.";
  }
  if (!cc->Inputs().HasTag("HEADER")) {
    // Note: for the current MediaPipe header implementation, we just need to
    // connect the output stream to both of the two inputs: INPUT and HEADER.
    return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "No header stream connected.";
  }
  // If the input stream has the header, just use it as the header. Otherwise,
  // assume the header is coming from HEADER stream.
  if (!cc->Inputs().Tag("INPUT").Header().IsEmpty()) {
    header_packet_ = cc->Inputs().Tag("INPUT").Header();
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status CallbackWithHeaderCalculator::Process(
    CalculatorContext* cc) {
  if (!cc->Inputs().Tag("INPUT").Value().IsEmpty() &&
      header_packet_.IsEmpty()) {
    // Header packet should be available before we receive any normal input
    // stream packet.
    return ::mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
           << "Header not available!";
  }
  if (header_packet_.IsEmpty() &&
      !cc->Inputs().Tag("HEADER").Value().IsEmpty()) {
    header_packet_ = cc->Inputs().Tag("HEADER").Value();
  }
  if (!cc->Inputs().Tag("INPUT").Value().IsEmpty()) {
    callback_(cc->Inputs().Tag("INPUT").Value(), header_packet_);
  }
  return ::mediapipe::OkStatus();
}

REGISTER_CALCULATOR(CallbackWithHeaderCalculator);

}  // namespace tool
}  // namespace mediapipe
