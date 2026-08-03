// Copyright 2026 The MediaPipe Authors.
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
// libFuzzer harness for the CalculatorGraphConfig protobuf parser surface.
//
// First-cut harness: raw byte input parsed via CalculatorGraphConfig::ParseFromArray
// then handed to CalculatorGraph::Initialize. Memory-safety violations in the
// proto parsing or graph initialization paths surface as ASAN/UBSAN crashes.

#include <cstddef>
#include <cstdint>

#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_graph.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  mediapipe::CalculatorGraphConfig config;
  if (!config.ParseFromArray(data, static_cast<int>(size))) {
    // Wire-format rejected by protobuf — not an interesting input.
    return 0;
  }
  mediapipe::CalculatorGraph graph;
  // absl::Status from Initialize is intentionally discarded; we only crash
  // on memory-safety violations, not on semantic errors.
  (void)graph.Initialize(config);
  return 0;
}
