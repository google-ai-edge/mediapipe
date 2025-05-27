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

#ifndef MEDIAPIPE_FRAMEWORK_API3_NODE_TEST_H_
#define MEDIAPIPE_FRAMEWORK_API3_NODE_TEST_H_

#include <string>

#include "absl/strings/string_view.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/api3/testing/bar.pb.h"
#include "mediapipe/framework/api3/testing/foo.pb.h"

namespace mediapipe::api3 {

inline constexpr absl::string_view kFooName = "Foo";
struct FooNode : Node<kFooName> {
  template <typename S>
  struct Contract {
    Input<S, int> in{"IN"};
    Optional<Input<S, int>> optional_in{"OPTIONAL_IN"};
    Repeated<Input<S, int>> repeated_in{"REPEATED_IN"};

    SideInput<S, std::string> side_in{"SIDE_IN"};
    Optional<SideInput<S, std::string>> optional_side_in{"OPTIONAL_SIDE_IN"};
    Repeated<SideInput<S, std::string>> repeated_side_in{"REPEATED_SIDE_IN"};

    Output<S, int> out{"OUT"};
    Optional<Output<S, int>> optional_out{"OPTIONAL_OUT"};
    Repeated<Output<S, int>> repeated_out{"REPEATED_OUT"};

    SideOutput<S, std::string> side_out{"SIDE_OUT"};
    Optional<SideOutput<S, std::string>> optional_side_out{"OPTIONAL_SIDE_OUT"};
    Repeated<SideOutput<S, std::string>> repeated_side_out{"REPEATED_SIDE_OUT"};

    Options<S, FooOptions> options;
  };
};

inline constexpr absl::string_view kBarName = "Bar";
struct BarNode : Node<kBarName> {
  template <typename S>
  struct Contract {
    Input<S, int> in{"IN"};
    Optional<Input<S, int>> optional_in{"OPTIONAL_IN"};
    Repeated<Input<S, int>> repeated_in{"REPEATED_IN"};

    SideInput<S, std::string> side_in{"SIDE_IN"};
    Optional<SideInput<S, std::string>> optional_side_in{"OPTIONAL_SIDE_IN"};
    Repeated<SideInput<S, std::string>> repeated_side_in{"REPEATED_SIDE_IN"};

    Output<S, int> out{"OUT"};
    Optional<Output<S, int>> optional_out{"OPTIONAL_OUT"};
    Repeated<Output<S, int>> repeated_out{"REPEATED_OUT"};

    SideOutput<S, std::string> side_out{"SIDE_OUT"};
    Optional<SideOutput<S, std::string>> optional_side_out{"OPTIONAL_SIDE_OUT"};
    Repeated<SideOutput<S, std::string>> repeated_side_out{"REPEATED_SIDE_OUT"};

    Options<S, FooOptions> options;
  };
};

// Split/external contract.

template <typename S>
struct Bar {
  Input<S, int> in{"IN"};
  Optional<Input<S, int>> optional_in{"OPTIONAL_IN"};
  Repeated<Input<S, int>> repeated_in{"REPEATED_IN"};

  SideInput<S, std::string> side_in{"SIDE_IN"};
  Optional<SideInput<S, std::string>> optional_side_in{"OPTIONAL_SIDE_IN"};
  Repeated<SideInput<S, std::string>> repeated_side_in{"REPEATED_SIDE_IN"};

  Output<S, int> out{"OUT"};
  Optional<Output<S, int>> optional_out{"OPTIONAL_OUT"};
  Repeated<Output<S, int>> repeated_out{"REPEATED_OUT"};

  SideOutput<S, std::string> side_out{"SIDE_OUT"};
  Optional<SideOutput<S, std::string>> optional_side_out{"OPTIONAL_SIDE_OUT"};
  Repeated<SideOutput<S, std::string>> repeated_side_out{"REPEATED_SIDE_OUT"};

  Options<S, BarOptions> options;
};

// Nodes sharing the same split/external contract.
inline constexpr absl::string_view kBarAName = "BarA";
struct BarANode : Node<kBarAName> {
  template <typename S>
  using Contract = Bar<S>;
};

inline constexpr absl::string_view kBarBName = "BarB";
struct BarBNode : Node<kBarBName> {
  template <typename S>
  using Contract = Bar<S>;
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_NODE_TEST_H_
