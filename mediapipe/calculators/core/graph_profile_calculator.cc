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

#include <memory>

#include "mediapipe/calculators/core/graph_profile_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_profile.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace api2 {

// This calculator periodically copies the GraphProfile from
// mediapipe::GraphProfiler::CaptureProfile to the "PROFILE" output stream.
//
// Similarly to the log files saved by GraphProfiler::WriteProfile when trace
// logging is enabled, the first captured profile contains the full
// canonicalized graph config and, if tracing is enabled, calculator names in
// graph traces. Subsequent profiles omit this information.
//
// Example config:
// node {
//   calculator: "GraphProfileCalculator"
//   output_stream: "FRAME:any_frame"
//   output_stream: "PROFILE:graph_profile"
// }
//
class GraphProfileCalculator : public Node {
 public:
  static constexpr Input<AnyType>::Multiple kFrameIn{"FRAME"};
  static constexpr Output<GraphProfile> kProfileOut{"PROFILE"};

  MEDIAPIPE_NODE_CONTRACT(kFrameIn, kProfileOut);

  static absl::Status UpdateContract(CalculatorContract* cc) {
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    auto options = cc->Options<::mediapipe::GraphProfileCalculatorOptions>();

    bool first_profile = prev_profile_ts_ == Timestamp::Unset();
    if (first_profile ||
        cc->InputTimestamp() - prev_profile_ts_ >= options.profile_interval()) {
      prev_profile_ts_ = cc->InputTimestamp();
      GraphProfile result;
      MP_RETURN_IF_ERROR(cc->GetProfilingContext()->CaptureProfile(
          &result, first_profile ? PopulateGraphConfig::kFull
                                 : PopulateGraphConfig::kNo));
      kProfileOut(cc).Send(result);
    }
    return absl::OkStatus();
  }

 private:
  Timestamp prev_profile_ts_;
};

MEDIAPIPE_REGISTER_NODE(GraphProfileCalculator);

}  // namespace api2
}  // namespace mediapipe
