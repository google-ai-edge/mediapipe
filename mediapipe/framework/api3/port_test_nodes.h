// Copyright 2025 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_FRAMEWORK_API3_PORT_TEST_NODES_H_
#define MEDIAPIPE_FRAMEWORK_API3_PORT_TEST_NODES_H_

#include <string>

#include "absl/strings/string_view.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"

namespace mediapipe::api3 {

inline constexpr absl::string_view kFooName = "Foo";
struct FooNode : Node<kFooName> {
  template <typename S>
  struct Contract {
    Input<S, int> input{"INPUT"};
    SideInput<S, std::string> side_input{"SIDE_INPUT"};
    Output<S, int> output{"OUTPUT"};
    SideOutput<S, std::string> side_output{"SIDE_OUTPUT"};
  };
};

inline constexpr absl::string_view kRepeatedFooName = "RepeatedFoo";
struct RepeatedFooNode : Node<kRepeatedFooName> {
  template <typename S>
  struct Contract {
    Repeated<Input<S, int>> input{"INPUT"};
    Repeated<SideInput<S, std::string>> side_input{"SIDE_INPUT"};
    Repeated<Output<S, int>> output{"OUTPUT"};
    Repeated<SideOutput<S, std::string>> side_output{"SIDE_OUTPUT"};
  };
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_PORT_TEST_NODES_H_
