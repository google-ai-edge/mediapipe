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

#include "mediapipe/framework/calculator_contract.h"

#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/framework/tool/tag_map.h"

namespace mediapipe {

::mediapipe::Status CalculatorContract::Initialize(
    const CalculatorGraphConfig::Node& node) {
  std::vector<::mediapipe::Status> statuses;

  auto input_stream_statusor = tool::TagMap::Create(node.input_stream());
  if (!input_stream_statusor.ok()) {
    statuses.push_back(std::move(input_stream_statusor).status());
  }
  auto output_stream_statusor = tool::TagMap::Create(node.output_stream());
  if (!output_stream_statusor.ok()) {
    statuses.push_back(std::move(output_stream_statusor).status());
  }
  auto input_side_packet_statusor =
      tool::TagMap::Create(node.input_side_packet());
  if (!input_side_packet_statusor.ok()) {
    statuses.push_back(std::move(input_side_packet_statusor).status());
  }
  auto output_side_packet_statusor =
      tool::TagMap::Create(node.output_side_packet());
  if (!output_side_packet_statusor.ok()) {
    statuses.push_back(std::move(output_side_packet_statusor).status());
  }

  if (!statuses.empty()) {
    auto builder = ::mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
                   << "Unable to initialize TagMaps for node.";
    for (const auto& status : statuses) {
      builder << "\n" << status.message();
    }
#if !(defined(MEDIAPIPE_LITE) || defined(MEDIAPIPE_MOBILE))
    builder << "\nFor calculator:\n";
    builder << node.DebugString();
#endif  // !(MEDIAPIPE_LITE || MEDIAPIPE_MOBILE)
    return std::move(builder);
  }

  node_config_ = &node;
  options_.Initialize(*node_config_);
  // Create the PacketTypeSets.
  inputs_ = absl::make_unique<PacketTypeSet>(
      std::move(input_stream_statusor).ValueOrDie());
  outputs_ = absl::make_unique<PacketTypeSet>(
      std::move(output_stream_statusor).ValueOrDie());
  input_side_packets_ = absl::make_unique<PacketTypeSet>(
      std::move(input_side_packet_statusor).ValueOrDie());
  output_side_packets_ = absl::make_unique<PacketTypeSet>(
      std::move(output_side_packet_statusor).ValueOrDie());
  return ::mediapipe::OkStatus();
}

::mediapipe::Status CalculatorContract::Initialize(
    const PacketGeneratorConfig& node) {
  std::vector<::mediapipe::Status> statuses;

  auto input_side_packet_statusor =
      tool::TagMap::Create(node.input_side_packet());
  if (!input_side_packet_statusor.ok()) {
    statuses.push_back(std::move(input_side_packet_statusor).status());
  }
  auto output_side_packet_statusor =
      tool::TagMap::Create(node.output_side_packet());
  if (!output_side_packet_statusor.ok()) {
    statuses.push_back(std::move(output_side_packet_statusor).status());
  }

  if (!statuses.empty()) {
    auto builder = UnknownErrorBuilder(MEDIAPIPE_LOC)
                   << "NodeTypeInfo Initialization failed.";
    for (const auto& status : statuses) {
      builder << "\n" << status.message();
    }
#if !(defined(MEDIAPIPE_LITE) || defined(MEDIAPIPE_MOBILE))
    builder << "\nFor packet_generator:\n";
    builder << node.DebugString();
#endif  // !(MEDIAPIPE_LITE || MEDIAPIPE_MOBILE)
    return std::move(builder);
  }

  input_side_packets_ = absl::make_unique<PacketTypeSet>(
      std::move(input_side_packet_statusor).ValueOrDie());
  output_side_packets_ = absl::make_unique<PacketTypeSet>(
      std::move(output_side_packet_statusor).ValueOrDie());
  return ::mediapipe::OkStatus();
}

::mediapipe::Status CalculatorContract::Initialize(
    const StatusHandlerConfig& node) {
  std::vector<::mediapipe::Status> statuses;

  auto input_side_packet_statusor =
      tool::TagMap::Create(node.input_side_packet());
  if (!input_side_packet_statusor.ok()) {
    statuses.push_back(std::move(input_side_packet_statusor).status());
  }

  if (!statuses.empty()) {
    auto builder = ::mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
                   << "NodeTypeInfo Initialization failed.";
    for (const auto& status : statuses) {
      builder << "\n" << status.message();
    }
#if !(defined(MEDIAPIPE_LITE) || defined(MEDIAPIPE_MOBILE))
    builder << "\nFor status_handler:\n";
    builder << node.DebugString();
#endif  // !(MEDIAPIPE_LITE || MEDIAPIPE_MOBILE)
    return std::move(builder);
  }

  input_side_packets_ = absl::make_unique<PacketTypeSet>(
      std::move(input_side_packet_statusor).ValueOrDie());
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
