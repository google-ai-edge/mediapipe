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

#ifndef MEDIAPIPE_FRAMEWORK_API3_CALCULATOR_H_
#define MEDIAPIPE_FRAMEWORK_API3_CALCULATOR_H_

#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/api3/calculator_contract.h"
#include "mediapipe/framework/api3/internal/contract_validator.h"
#include "mediapipe/framework/api3/internal/has_update_contract.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe::api3 {

// Calculator class should be used to implement a defined node as a calculator.
//
// Let's say there's a node `FooNode` defined as below:
//
// foo_node.h:
// ```
//   inline constexpr absl::string_view kFooNodeName = "FooNode";
//   struct FooNode : Node<kFooNodeName> {
//     template <typename S>
//     struct Contract {
//       Input<S, int> input{"INPUT"};
//       Output<S, std::string> output{"OUTPUT"};
//       Options<S, FooOptions> options;
//     };
//   };
// ```
//
// The calculator implementation would subclass `Calculator` and specify a node
// it implements as a first template parameter and itself as a second template
// parameter for auto registration (so then it can be found and instantiated
// by provided registration name in the node definition).
//
// The subclass can implement Process() to do the processing, Open() to do the
// initialization, and Close() to do the cleanup. The subclass can also add a
// static UpdateContract() function to update the contract (e.g. request some
// platform specific graph service).
//
// For example:
//
// foo_node.cc:
// ```
//   class FooNodeImpl : public Calculator<FooNode, FooNodeImpl> {
//    public:
//     absl::Status Open(CalculatorContext<FooNode>& cc) final {
//       ...
//     }
//     absl::Status Process(CalculatorContext<FooNode>& cc) final {
//       RET_CHECK(cc.input);
//       int input = cc.input.GetOrDie();
//       ...
//     }
//     absl::Status Close(CalculatorContext<FooNode>& cc) final {
//       ...
//     }
//   };
// ```
//
// Now FooNodeImpl is registered as "FooNode" (taken from `class FooNode...`) in
// MediaPipe registry automatically and can be used in `CalculatorGraphConfig`
// by name.
//
// Below is the explanation of how framework calls various node functions:
//
//   static absl::Status UpdateContract(CalculatorContract<FooNode>& cc)
//     (Optional) Invoked on graph initialization if defined to update the
//     contract.
//
// Then, for each run of the graph on a set of input side packets, the
// following sequence will occur.
//
//   absl::Status Open(CalculatorContext<FooNode>&)
//     (Optional) To initialize the calculator.
//
//     NOTE : with the latest API, the default Timestamp
//       Offset of a calculator is 0. (Pay attention when migrating from older
//       calculator APIs (excluding API2), because the default there is
//       "arbitrary" Timestamp Offset.)
//
//       With 0 Timestamp Offset, calculator is expected to send an output
//       packet for every input packet at the input packet timestamp.
//
//       If the calculator returns from Process without adding an output to some
//       or all output streams:
//       - The framework will send a timestamp bound update to downstream
//         calculators that there won't be a packet for that particular
//         timestamp on output streams in question.
//       - Dependent downstream calculator(s) will execute on timestamp bound
//         update if they have other input streams with ready packets at that
//         particular timestamp. Input streams corresponding to output streams
//         in question (with timestamp bound update) will have empty packets, so
//         calculators need to use: IsEmpty before getting data.
//
//     You can disable default 0 Timestamp Offset in the node definition as
//     following:
//
//       foo_node.h:
//       ```
//         inline constexpr absl::string_view kFooNodeName = "Foo";
//         struct FooNode : Node<kFooName> {
//           template <typename S>
//           struct Contract {
//             // ...
//             static absl::Status UpdateContract(
//                  CalculatorContract<FooNode>& cc) {
//               cc.SetTimestampOffset(TimestampDiff::Unset());
//               return absl::OkStatus();
//             }
//           };
//         };
//       ```
//
//     NOTE: Clients can help optimize framework packet queueing by calling
//       SetNextTimestampBound on outputs if applicable (e.g.
//       cc.output.SetNextTimestampBound())
//
//   absl::Status Process(CalculatorContext<FooNode>&) (repeatedly)
//
//     For Non-Source Nodes (nodes that have input streams):
//
//     By default, invoked when every input stream either has a packet at
//     timestamp T or the framework knows the packet is not expected at that
//     timestamp. The latter occurs during timestamp bound update (Timestamp
//     Offset is 0 (default), explicit call to SetNextTimestampBound() on
//     calculator graph/upstream calculator or receiving a packet with a
//     timestamp > T), and this results in corresponding input stream being
//     empty during Process() call, so clients need to use: IsEmpty before
//     getting data.
//
//     This behavior may be adjusted, by utilizing different input stream
//     handlers (please consult corresponding documentation):
//     - DefaultInputStreamHandler (default)
//     - FixedSizeInputStreamHandler
//     - ImmediateInputStreamHandler
//     - etc. in mediapipe/third_party/framework/stream_handler
//
//     NOTE: strive to stick to the default handler (not specifying explicitly)
//       and only use custom one if you know for sure what the custom one does
//       and what effect it will have on your graph.
//
//     In the first place, consider setting it in the node definition if the
//     calculator is required to have a custom stream handler always:
//
//       foo_node.h:
//       ```
//         inline constexpr absl::string_view kFooNodeName = "Foo";
//         struct FooNode : Node<kFooName> {
//           template <typename S>
//           struct Contract {
//             // ...
//             static absl::Status UpdateContract(
//                  CalculatorContract<FooNode>& cc) {
//               cc.SetInputStreamHandler("FixedSizeInputStreamHandler");
//               return absl::OkStatus();
//             }
//           };
//         };
//       ```
//
//     Otherwise, you can set it in CalculatorGraphConfig:
//       node {
//         calculator: "CalculatorRunningAtOneFps"
//         input_stream: "packets_streaming_in_at_ten_fps"
//         input_stream_handler {
//           input_stream_handler: "FixedSizeInputStreamHandler"
//         }
//       }
//
//     or Graph builder:
//       Graph<...> graph;
//       auto& node = graph.AddNode<FooNode>();
//       node.SetInputStreamHandler("FixedSizeInputStreamHandler")
//
//     For Source Nodes (nodes that don't have input streams):
//
//     Continues to have Process() called as long as it returns an
//     absl::OkStatus(). Returning tool::StatusStop() indicates source node is
//     done producing data.
//
//   absl::Status Close(CalculatorContext<FooNode>&)
//
//     After all calls to Process() finish or when all input streams close, the
//     framework calls Close(). This function is always called if Open() was
//     called and succeeded and even if the graph run terminated because of an
//     error. No inputs are available via any input streams during Close(), but
//     it still has access to input side packets and therefore may write
//     outputs. After Close() returns, the calculator should be considered a
//     dead node. The calculator object is destroyed as soon as the graph
//     finishes running.
//
// NOTE: the entire calculator is constructed and destroyed for each graph run
// (set of input side packets, which could mean once per video, or once per
// image). Expensive operations and large objects should be input side packets
// or provided by graph services.
//
// Calculators must be thread-compatible.
// The framework does not call the non-const methods of a calculator from
// multiple threads at the same time. However, the thread that calls the methods
// of a calculator is not fixed. Therefore, calculators should not use
// ThreadLocal objects.
// TODO: get rid of api2 usage.
template <typename NodeT, typename ImplT>
class Calculator : public CalculatorBase,
                   private api2::internal::NodeRegistrator<ImplT>,
                   private ContractValidator<NodeT::template Contract> {
 public:
  // Invoked once to initialize the calculator. (More details available in
  // `Calculator` class documentation.)
  virtual absl::Status Open(CalculatorContext<NodeT>& cc) {
    return absl::OkStatus();
  }

  // Invoked repeatedly to process inputs/produce outputs. (More details
  // available in `Calculator` class documentation.)
  //
  // IMPORTANT: pay special attention for `TimestampOffset` - which is equal `0`
  //   by default - even if you don't send outputs, the framework will broadcast
  //   timestamp bound update for the current input timestamp - in short -
  //   notifying all downstream calculators that there won't be output for the
  //   current input timestamp for this calculator. (More details in
  //   `Calculator` class documentation.)
  //
  // IMPORTANT: even for non `Optional` inputs, you can receive an empty packet
  //   (situation with timestamp bound update from some upstream calculator), so
  //   always handle this situation according to your requirements. (More
  //   details in `Calculator` class documentation.)
  virtual absl::Status Process(CalculatorContext<NodeT>& cc) {
    return absl::UnimplementedError(
        "`Process` is not implemented. It is OK only for side packet "
        "generator calculators - no input/output streams, only input/output "
        "side packets. For all other cases, `Process` must be implemented.");
  }

  // Invoked once for a calculator cleanup. (More details in `Calculator` class
  // documentation.)
  virtual absl::Status Close(CalculatorContext<NodeT>& cc) {
    return absl::OkStatus();
  }

  static constexpr auto kCalculatorName = NodeT::GetRegistrationName();

  static absl::Status GetContract(mediapipe::CalculatorContract* cc) {
    RET_CHECK_EQ(cc->GetMaxInFlight(), 1)
        << "Only single invocation in flight is allowed.";

    std::vector<absl::Status> statuses;
    auto store_status = [&statuses](absl::Status status) {
      if (!status.ok()) statuses.push_back(std::move(status));
    };

    CalculatorContract<NodeT> specialized_contract(*cc, store_status);

    // Default to SetOffset(0);
    cc->SetTimestampOffset(TimestampDiff(0));

    // Optional contract update from node (interface) - e.g. unsetting 0
    // timestamp offset.
    if constexpr (kHasUpdateContract<CalculatorContract<NodeT>,
                                     CalculatorContract<NodeT>>) {
      store_status(
          CalculatorContract<NodeT>::UpdateContract(specialized_contract));
    }

    // Optional contract update from implementation - e.g. Web implementation
    // requesting WebGpuService, Android implementation requesting GpuService.
    if constexpr (kHasUpdateContract<ImplT, CalculatorContract<NodeT>>) {
      store_status(ImplT::UpdateContract(specialized_contract));
    }

    if (statuses.empty()) return absl::OkStatus();
    if (statuses.size() == 1) return statuses[0];
    return tool::CombinedStatus("Multiple errors", statuses);
  }

  absl::Status Open(mediapipe::CalculatorContext* cc) final {
    context_.emplace(*cc);
    absl::Status status = Open(*context_);
    context_->Clear();
    return status;
  }

  absl::Status Process(mediapipe::CalculatorContext* cc) final {
    context_->Reset(*cc);
    absl::Status status = Process(*context_);
    context_->Clear();
    return status;
  }

  absl::Status Close(mediapipe::CalculatorContext* cc) final {
    context_->Reset(*cc);
    absl::Status status = Close(*context_);
    context_->Clear();
    return status;
  }

 private:
  // Specialized `CalculatorContext<...>` to enable reuse across repeated
  // `Process` invocations.
  std::optional<CalculatorContext<NodeT>> context_;
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_CALCULATOR_H_
