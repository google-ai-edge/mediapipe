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

// Defines Helper functions.
#include "mediapipe/framework/tool/validate_type.h"

#include <memory>
#include <vector>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_contract.h"
#include "mediapipe/framework/legacy_calculator_support.h"
#include "mediapipe/framework/packet_generator.h"
#include "mediapipe/framework/packet_generator.pb.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/port/map_util.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/status_util.h"

#if !defined(MEDIAPIPE_MOBILE) && !defined(MEDIAPIPE_LITE)
#include "absl/synchronization/blocking_counter.h"
#include "mediapipe/framework/port/threadpool.h"
#include "mediapipe/util/cpu_util.h"
#endif  // !MEDIAPIPE_MOBILE && !MEDIAPIPE_LITE

namespace mediapipe {

namespace tool {
::mediapipe::Status RunGeneratorFillExpectations(
    const PacketGeneratorConfig& input_config, const std::string& package) {
  // TODO Remove conversion after everyone uses input/output
  // side packet.
  PacketGeneratorConfig config = input_config;

  ASSIGN_OR_RETURN(
      auto static_access,
      internal::StaticAccessToGeneratorRegistry::CreateByNameInNamespace(
          package, config.packet_generator()),
      _ << config.packet_generator()
        << " is not a registered packet generator.");

  CalculatorContract contract;
  MP_RETURN_IF_ERROR(contract.Initialize(config));

  {
    LegacyCalculatorSupport::Scoped<CalculatorContract> s(&contract);
    MP_RETURN_IF_ERROR(static_access->FillExpectations(
                           config.options(), &contract.InputSidePackets(),
                           &contract.OutputSidePackets()))
            .SetPrepend()
        << config.packet_generator() << "::FillExpectations failed: ";
  }

  // Check that everything got initialized.
  std::vector<::mediapipe::Status> statuses;
  statuses.push_back(ValidatePacketTypeSet(contract.InputSidePackets()));
  statuses.push_back(ValidatePacketTypeSet(contract.OutputSidePackets()));
  return tool::CombinedStatus(
      absl::StrCat(config.packet_generator(), "::FillExpectations failed: "),
      statuses);
}

::mediapipe::Status RunGenerateAndValidateTypes(
    const std::string& packet_generator_name,
    const PacketGeneratorOptions& extendable_options,
    const PacketSet& input_side_packets, PacketSet* output_side_packets,
    const std::string& package) {
  CHECK(output_side_packets);
  // Get static access to functions.
  ASSIGN_OR_RETURN(
      auto static_access,
      internal::StaticAccessToGeneratorRegistry::CreateByNameInNamespace(
          package, packet_generator_name),
      _ << packet_generator_name << " is not a registered packet generator.");
  // Create PacketTypeSets.
  PacketTypeSet input_side_packet_types(input_side_packets.TagMap());
  PacketTypeSet output_side_packet_types(output_side_packets->TagMap());

  // Fill the PacketTypeSets with type information.
  MP_RETURN_IF_ERROR(static_access->FillExpectations(extendable_options,
                                                     &input_side_packet_types,
                                                     &output_side_packet_types))
          .SetPrepend()
      << packet_generator_name << "::FillExpectations failed: ";
  // Check that the types were filled well.
  std::vector<::mediapipe::Status> statuses;
  statuses.push_back(ValidatePacketTypeSet(input_side_packet_types));
  statuses.push_back(ValidatePacketTypeSet(output_side_packet_types));
  MP_RETURN_IF_ERROR(tool::CombinedStatus(
      absl::StrCat(packet_generator_name, "::FillExpectations failed: "),
      statuses));

  MP_RETURN_IF_ERROR(
      ValidatePacketSet(input_side_packet_types, input_side_packets))
          .SetPrepend()
      << packet_generator_name
      << "::FillExpectations expected different input type than those given: ";
  MP_RETURN_IF_ERROR(static_access->Generate(extendable_options,
                                             input_side_packets,
                                             output_side_packets))
          .SetPrepend()
      << packet_generator_name << "::Generate failed: ";
  MP_RETURN_IF_ERROR(
      ValidatePacketSet(output_side_packet_types, *output_side_packets))
          .SetPrepend()
      << packet_generator_name
      << "::FillExpectations expected different "
         "output type than those produced: ";
  return ::mediapipe::OkStatus();
}

}  // namespace tool
}  // namespace mediapipe
