// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/util/graph_builder_utils.h"

#include <string>
#include <utility>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"

namespace mediapipe {
namespace {

bool StartsWithTag(absl::string_view name, absl::string_view tag) {
  constexpr absl::string_view kDelimiter(":");
  return absl::StartsWith(name, absl::StrCat(tag, kDelimiter));
}

}  // namespace

bool HasInput(const CalculatorGraphConfig::Node& node, absl::string_view tag) {
  for (int i = 0; i < node.input_stream_size(); ++i) {
    if (StartsWithTag(node.input_stream(i), tag)) {
      return true;
    }
  }
  return false;
}

bool HasSideInput(const CalculatorGraphConfig::Node& node,
                  absl::string_view tag) {
  for (int i = 0; i < node.input_side_packet_size(); ++i) {
    if (StartsWithTag(node.input_side_packet(i), tag)) {
      return true;
    }
  }
  return false;
}

bool HasOutput(const CalculatorGraphConfig::Node& node, absl::string_view tag) {
  for (int i = 0; i < node.output_stream_size(); ++i) {
    if (StartsWithTag(node.output_stream(i), tag)) {
      return true;
    }
  }
  return false;
}

}  // namespace mediapipe
