// Copyright 2022 The MediaPipe Authors.
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

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "mediapipe/calculators/core/bypass_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace api2 {

using mediapipe::BypassCalculatorOptions;

// Defines a "bypass" channel to use in place of a disabled feature subgraph.
// By default, all inputs are discarded and all outputs are ignored.
// Certain input streams can be passed to corresponding output streams
// by specifying them in "pass_input_stream" and "pass_output_stream" options.
// All output streams are updated with timestamp bounds indicating completed
// output.
//
// Note that this calculator is designed for use as a contained_node in a
// SwitchContainer.  For this reason, any input and output tags are accepted,
// and stream semantics are specified through BypassCalculatorOptions.
//
// Example config:
//     node {
//       calculator: "BypassCalculator"
//       input_stream: "APPEARANCES:appearances_post_facenet"
//       input_stream: "VIDEO:video_frame"
//       input_stream: "FEATURE_CONFIG:feature_config"
//       input_stream: "ENABLE:gaze_enabled"
//       output_stream: "APPEARANCES:analyzed_appearances"
//       output_stream: "FEDERATED_GAZE_OUTPUT:federated_gaze_output"
//       node_options: {
//         [type.googleapis.com/mediapipe.BypassCalculatorOptions] {
//           pass_input_stream: "APPEARANCES"
//           pass_output_stream: "APPEARANCES"
//         }
//       }
//     }
//
class BypassCalculator : public Node {
 public:
  static constexpr mediapipe::api2::Input<int>::Optional kNotNeeded{"N_N_"};
  MEDIAPIPE_NODE_CONTRACT(kNotNeeded);
  using IdMap = std::map<CollectionItemId, CollectionItemId>;

  // Returns the map of passthrough input and output stream ids.
  static absl::StatusOr<IdMap> GetPassMap(
      const BypassCalculatorOptions& options, const tool::TagMap& input_map,
      const tool::TagMap& output_map) {
    IdMap result;
    auto& input_streams = options.pass_input_stream();
    auto& output_streams = options.pass_output_stream();
    int size = std::min(input_streams.size(), output_streams.size());
    for (int i = 0; i < size; ++i) {
      std::pair<std::string, int> in_tag, out_tag;
      MP_RETURN_IF_ERROR(tool::ParseTagIndex(options.pass_input_stream(i),
                                             &in_tag.first, &in_tag.second));
      MP_RETURN_IF_ERROR(tool::ParseTagIndex(options.pass_output_stream(i),
                                             &out_tag.first, &out_tag.second));
      auto input_id = input_map.GetId(in_tag.first, in_tag.second);
      auto output_id = output_map.GetId(out_tag.first, out_tag.second);
      result[input_id] = output_id;
    }
    return result;
  }

  // Identifies all specified streams as "Any" packet type.
  // Identifies passthrough streams as "Same" packet type.
  static absl::Status UpdateContract(CalculatorContract* cc) {
    auto options = cc->Options<BypassCalculatorOptions>();
    RET_CHECK_EQ(options.pass_input_stream().size(),
                 options.pass_output_stream().size());
    MP_ASSIGN_OR_RETURN(
        auto pass_streams,
        GetPassMap(options, *cc->Inputs().TagMap(), *cc->Outputs().TagMap()));
    std::set<CollectionItemId> pass_out;
    for (auto entry : pass_streams) {
      pass_out.insert(entry.second);
      cc->Inputs().Get(entry.first).SetAny();
      cc->Outputs().Get(entry.second).SetSameAs(&cc->Inputs().Get(entry.first));
    }
    for (auto id = cc->Inputs().BeginId(); id != cc->Inputs().EndId(); ++id) {
      if (pass_streams.count(id) == 0) {
        cc->Inputs().Get(id).SetAny();
      }
    }
    for (auto id = cc->Outputs().BeginId(); id != cc->Outputs().EndId(); ++id) {
      if (pass_out.count(id) == 0) {
        cc->Outputs().Get(id).SetAny();
      }
    }
    for (auto id = cc->InputSidePackets().BeginId();
         id != cc->InputSidePackets().EndId(); ++id) {
      cc->InputSidePackets().Get(id).SetAny();
    }
    return absl::OkStatus();
  }

  // Saves the map of passthrough input and output stream ids.
  absl::Status Open(CalculatorContext* cc) override {
    auto options = cc->Options<BypassCalculatorOptions>();
    MP_ASSIGN_OR_RETURN(
        pass_streams_,
        GetPassMap(options, *cc->Inputs().TagMap(), *cc->Outputs().TagMap()));
    return absl::OkStatus();
  }

  // Copies packets between passthrough input and output streams.
  // Updates timestamp bounds on all output streams.
  absl::Status Process(CalculatorContext* cc) override {
    std::set<CollectionItemId> pass_out;
    for (auto entry : pass_streams_) {
      pass_out.insert(entry.second);
      auto& packet = cc->Inputs().Get(entry.first).Value();
      if (packet.Timestamp() == cc->InputTimestamp()) {
        cc->Outputs().Get(entry.second).AddPacket(packet);
      }
    }
    Timestamp bound = cc->InputTimestamp().NextAllowedInStream();
    for (auto id = cc->Outputs().BeginId(); id != cc->Outputs().EndId(); ++id) {
      if (pass_out.count(id) == 0) {
        cc->Outputs().Get(id).SetNextTimestampBound(
            std::max(cc->Outputs().Get(id).NextTimestampBound(), bound));
      }
    }
    return absl::OkStatus();
  }

  // Close all output streams.
  absl::Status Close(CalculatorContext* cc) override {
    for (auto id = cc->Outputs().BeginId(); id != cc->Outputs().EndId(); ++id) {
      cc->Outputs().Get(id).Close();
    }
    return absl::OkStatus();
  }

 private:
  IdMap pass_streams_;
};

MEDIAPIPE_REGISTER_NODE(BypassCalculator);

}  // namespace api2
}  // namespace mediapipe
