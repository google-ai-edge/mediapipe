// Copyright 2024 The MediaPipe Authors.
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

#include "mediapipe/calculators/util/resource_provider_calculator.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/util/resource_provider_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/resources.h"

namespace mediapipe::api2 {

absl::Status ResourceProviderCalculator::UpdateContract(CalculatorContext* cc) {
  RET_CHECK_GT(kResources(cc).Count(), 0)
      << "At least one output resource must be specified.";
  const bool uses_side_packets = kIds(cc).Count() > 0;
  const auto& opts = cc->Options<ResourceProviderCalculatorOptions>();
  const bool uses_options = opts.resource_id_size() > 0;
  RET_CHECK(uses_side_packets ^ uses_options)
      << "Either side packets or options must be used, not both.";

  if (uses_side_packets) {
    RET_CHECK_EQ(kIds(cc).Count(), kResources(cc).Count());
  } else if (uses_options) {
    RET_CHECK_EQ(opts.resource_id_size(), kResources(cc).Count());
  }
  return absl::OkStatus();
}

absl::Status ResourceProviderCalculator::Open(CalculatorContext* cc) {
  const bool uses_side_packets = kIds(cc).Count() > 0;
  const auto& opts = cc->Options<ResourceProviderCalculatorOptions>();
  Resources::Options res_opts = {};
  res_opts.read_as_binary =
      opts.read_mode() != ResourceProviderCalculatorOptions::READ_MODE_TEXT;

  auto get_resource_id_fn = [&](int i) -> absl::StatusOr<absl::string_view> {
    if (uses_side_packets) {
      return kIds(cc)[i].Get();
    }
    return opts.resource_id(i);
  };

  for (int i = 0; i < kResources(cc).Count(); ++i) {
    MP_ASSIGN_OR_RETURN(absl::string_view res_id, get_resource_id_fn(i));
    MP_ASSIGN_OR_RETURN(std::unique_ptr<Resource> res,
                        cc->GetResources().Get(res_id, res_opts));
    Packet<Resource> res_packet = api2::PacketAdopting(std::move(res));
    kResources(cc)[i].Set(std::move(res_packet));
  }
  return absl::OkStatus();
}

MEDIAPIPE_REGISTER_NODE(ResourceProviderCalculator)

}  // namespace mediapipe::api2
