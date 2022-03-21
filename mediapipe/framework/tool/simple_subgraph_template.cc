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
// This template is used by the mediapipe_simple_subgraph macro in
// //mediapipe/framework/tool/mediapipe_graph.bzl

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/subgraph.h"

namespace mediapipe {

// clang-format off
static const char binary_graph[] =
#include "{{SUBGRAPH_INC_FILE_PATH}}"
    ;  // NOLINT(whitespace/semicolon)

class {{SUBGRAPH_CLASS_NAME}} : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
        const SubgraphOptions& /*options*/) {
    CalculatorGraphConfig config;
    // Note: this is a binary protobuf serialization, and may include NUL
    // bytes. The trailing NUL added to the string literal should be excluded.
    if (config.ParseFromArray(binary_graph, sizeof(binary_graph) - 1)) {
      return config;
    } else {
      return absl::InternalError("Could not parse subgraph.");
    }
  }
};
REGISTER_MEDIAPIPE_GRAPH({{SUBGRAPH_CLASS_NAME}});
// clang-format on

}  // namespace mediapipe
