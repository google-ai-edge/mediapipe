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

#ifndef MEDIAPIPE_FRAMEWORK_API3_CALCULATOR_TEST_H_
#define MEDIAPIPE_FRAMEWORK_API3_CALCULATOR_TEST_H_

#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/api3/calculator_contract.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/api3/testing/bar.pb.h"
#include "mediapipe/framework/api3/testing/foo.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/graph_service.h"

namespace mediapipe::api3 {

// This will be set on the contract UpdateContract of the implementation.
inline constexpr GraphService<std::string> kTestStringService(
    "kTestService", GraphServiceBase::kDisallowDefaultInitialization);

inline constexpr absl::string_view kPassThroughName = "PassThrough";
struct PassThroughNode : Node<kPassThroughName> {
  template <typename S>
  struct Contract {
    Input<S, int> in{"IN"};
    SideInput<S, std::string> side_in{"SIDE_IN"};
    Output<S, int> out{"OUT"};
    SideOutput<S, std::string> side_out{"SIDE_OUT"};

    Options<S, FooOptions> foo_options;
    Options<S, BarOptions> bar_options;

    static absl::Status UpdateContract(
        CalculatorContract<PassThroughNode>& cc) {
      cc.SetInputStreamHandler("DefaultInputStreamHandler");
      cc.SetTimestampOffset(TimestampDiff::Unset());
      cc.UseService(kTestStringService);
      return absl::OkStatus();
    }
  };
};

class PassThroughNodeImpl
    : public Calculator<PassThroughNode, PassThroughNodeImpl> {
 public:
  static absl::Status UpdateContract(CalculatorContract<PassThroughNode>& cc);
  absl::Status Open(CalculatorContext<PassThroughNode>& cc) final;
  absl::Status Process(CalculatorContext<PassThroughNode>& cc) final;
  absl::Status Close(CalculatorContext<PassThroughNode>& cc) final;
};

// Example of other implementations of the same contract: "pure" interfaces,
// with implementations hidden in .cc file.

template <typename S>
struct PassThrough {
  Input<S, int> in{"IN"};
  SideInput<S, std::string> side_in{"SIDE_IN"};
  Output<S, int> out{"OUT"};
  SideOutput<S, std::string> side_out{"SIDE_OUT"};

  template <typename N>
  static absl::Status UpdateContract(CalculatorContract<N>& cc) {
    cc.SetInputStreamHandler("DefaultInputStreamHandler");
    cc.SetTimestampOffset(TimestampDiff::Unset());
    cc.UseService(kTestStringService);
    return absl::OkStatus();
  }
};

inline constexpr absl::string_view kSharedPassThroughA = "SharedPassThroughA";
struct SharedPassThroughANode : Node<kSharedPassThroughA> {
  template <typename S>
  using Contract = PassThrough<S>;
};

inline constexpr absl::string_view kSharedPassThroughB = "SharedPassThroughB";
struct SharedPassThroughBNode : Node<kSharedPassThroughB> {
  template <typename S>
  using Contract = PassThrough<S>;
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_CALCULATOR_TEST_H_
