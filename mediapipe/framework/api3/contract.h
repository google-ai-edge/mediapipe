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

#ifndef MEDIAPIPE_FRAMEWORK_API3_CONTRACT_H_
#define MEDIAPIPE_FRAMEWORK_API3_CONTRACT_H_

#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/api3/internal/contract_fields.h"
#include "mediapipe/framework/api3/internal/port_base.h"
#include "mediapipe/framework/api3/internal/specializers.h"

namespace mediapipe::api3 {

// Node contract is:
// - any struct template
// - with no parents and non virtual
// - which uses special types to define node inputs, outputs and options.
//
// For example:
//
// foo.h:
// ```
//   template <typename S>
//   struct Foo {
//     Input<S, int> input{"INPUT"};
//     Output<S, std::string> output{"OUTPUT"};
//     Options<S, FooOptions> options;
//   };
// ```
//
// S in `typename S` stands for SPECIALIZER and will be used to specialize
// contract inputs, outputs and options for various scenarios: calculator,
// subgraph, graph and runner.
//
// You can use the following types for contract fields:
//
// - Input (E.g. Input<S, int>)
// - Output (E.g. Output<S, int>)
// - SideInput (E.g. SideInput<S, int>)
// - SideOutput (E.g. SideOutput<S, int>)
// - Repeated<...> (E.g. Repeated<Input<S, int>>)
//   NOTE: Repeated<Optional<...>> is disabled.
// - Optional<...> (E.g. Optional<Input<S, int>>)
//   NOTE: Optional<Repeated<...>> is disabled.
//
// See more in node.h, calculator.h for how to use a contract for various
// scenarios.

// Defines an input (input stream) - which carries a sequence of packets, whose
// timestamps must be monotonically increasing.
//
// In node contract:
// ```
//   template <typename S>
//   struct Contract {
//     Input<S, Tensor> input_tensor{"TENSOR"};
//     ...
//   }
// ```
//
// In `CalculatorGraphConfig`:
// ```
//   node {
//     ...
//     input_stream: "TENSOR:tensor_in"
//     ...
//   }
// ```
//
// In calculator:
// ```
//   absl::Status Process(CalculatorContext<...>& cc) {
//     cc.input_tensor
//     ...
//   }
// ```
template <typename S, typename PayloadT>
class Input;

// Defines a side input - input side packet - which carries a single packet
// (with unspecified timestamp). It can be used to provide some data that will
// remain constant.
//
// In node contract:
// ```
//   template <typename S>
//   struct Contract {
//     SideInput<S, Model> model{"MODEL"};
//     ...
//   }
// ```
//
// In `CalculatorGraphConfig`:
// ```
//   node {
//     ...
//     input_side_packet: "MODEL:model"
//     ...
//   }
// ```
//
// In calculator:
// ```
//   absl::Status Open(CalculatorContext<...>& cc) {
//     cc.model
//     ...
//   }
// ```
template <typename S, typename PayloadT>
class SideInput;

// Defines an output - output stream - which carries a sequence of packets,
// whose timestamps must be monotonically increasing.
//
// In node contract:
// ```
//   template <typename S>
//   struct Contract {
//     Output<S, Tensor> output_tensor{"TENSOR"};
//     ...
//   }
// ```
//
// In `CalculatorGraphConfig`:
// ```
//   node {
//     ...
//     output_stream: "TENSOR:tensor_out"
//     ...
//   }
// ```
//
// In calculator:
// ```
//   absl::Status Process(CalculatorContext<...>& cc) {
//     cc.output_tensor
//     ...
//   }
// ```
template <typename S, typename PayloadT>
class Output;

// Defines a side output - output side packet - which carries a single packet
// (with unspecified timestamp). It can be used to provide some data that will
// remain constant.
//
// In node contract:
// ```
//   template <typename S>
//   struct Contract {
//     SideOutput<S, Model> model{"MODEL"};
//     ...
//   }
// ```
//
// In `CalculatorGraphConfig`:
// ```
//   node {
//     ...
//     output_side_packet: "MODEL:model"
//     ...
//   }
// ```
//
// In calculator:
// ```
//   absl::Status Open(CalculatorContext<...>& cc) {
//     cc.model
//     ...
//   }
// ```
template <typename S, typename PayloadT>
class SideOutput;

// Defines a repeated (side)input or (side)output as following:
//
// ```
//   Repeated<Input<S, int>> repeated_input{"REPEATED_IN"};
// ```
template <typename P, typename = void>
class Repeated;

// TODO: rename to OptionalConnection ? (cc.image.IsConnected())
//
// Defines an optional (side)input or (side)output as following:
//
// ```
//   Optional<Input<S, int>> optional_input{"OPTIONAL_IN"};
// ```
//
// IMPORTANT: only in rare situations you may need all your inputs and outputs
//   `Optional`. All `Optional` inputs and outputs may indicate you are putting
//   too much into a single calculator. Try to recognize this early and split
//   into multiple calculators intstead.
template <typename P, typename = void>
class Optional;

// Defines calculator options. (Calculator can have multiple options.)
//
// If specified, appear as literal values in the `node_options` field (`options`
// for proto2) of the `CalculatorGraphConfiguration.Node` message.
//
// In node contract:
// ```
//   template <typename S>
//   struct Contract {
//     Options<S, InferenceCalculatorOptions> options;
//     ...
//   }
// ```
//
// In `CalculatorGraphConfig`:
// ```
//   node {
//     ...
//     node_options: {
//       [type.googleapis.com/mediapipe.InferenceCalculatorOptions] {
//         model_path: "model/path"
//       }
//     }
//   }
// ```
//
// In calculator:
// ```
//   absl::Status Open(CalculatorContext<...>& cc) {
//     cc.options.Get().model_path
//     ...
//   }
// ```
template <typename S, typename ProtoT>
struct Options;

// Used in cases when inputs/outputs are provided in alternative ways.
// (E.g. `Build(Graph<>& graph, Stream<..> input) -> Stream<...>)`)
template <typename S>
struct GenericContract {};

// Repeated / Optional template implementation

template <typename P, typename>
class Repeated : public internal_port::RepeatedBase<typename P::Specializer,
                                                    typename P::Field> {
 public:
  static_assert(!std::is_same_v<typename P::Field, OptionalField>,
                "`Repeated` doesn't accept `Optional` ports (`Repeated` is "
                "already optional.)");

  using Field = RepeatedField;
  using Contained = P;
  using internal_port::RepeatedBase<typename P::Specializer,
                                    typename P::Field>::RepeatedBase;

  P& At(int index) const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    if (auto it = repeated_ports_.find(index); it != repeated_ports_.end()) {
      return *it->second;
    }
    auto port = std::unique_ptr<P>(
        new P(std::make_unique<internal_port::StrViewTag>(
                  internal_port::RepeatedBase<typename P::Specializer,
                                              typename P::Field>::Tag()),
              index));
    internal_port::RepeatedBase<typename P::Specializer,
                                typename P::Field>::InitPort(*port);
    P* repated_port_ptr = port.get();
    if (auto [it, inserted] = repeated_ports_.insert({index, std::move(port)});
        !inserted) {
      ABSL_LOG(DFATAL) << "Unexpected port at index: " << index;
    }
    return *repated_port_ptr;
  }

  P& operator[](int index) const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return At(index);
  }

  class Iterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using pointer = P*;
    using reference = P;  // allowed; see e.g. std::istreambuf_iterator

    explicit Iterator(const Repeated* repeated, int pos)
        : repeated_(repeated), pos_(pos) {}

    Iterator& operator++() {
      ++pos_;
      return *this;
    }

    Iterator operator++(int) {
      Iterator res = *this;
      ++(*this);
      return res;
    }

    bool operator==(const Iterator& other) const { return pos_ == other.pos_; }
    bool operator!=(const Iterator& other) const { return !(*this == other); }

    P& operator*() const { return repeated_->At(pos_); }

   private:
    const Repeated* repeated_;
    int pos_ = 0;
  };

  Iterator begin() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return Iterator(this, 0);
  }
  Iterator end() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return Iterator(this,
                    internal_port::RepeatedBase<typename P::Specializer,
                                                typename P::Field>::Count());
  }

 private:
  // TODO: can I get rid of "mutable" here?
  mutable absl::flat_hash_map<int, std::unique_ptr<P>> repeated_ports_;
};

template <typename P, typename>
class Optional : public P {
 public:
  static_assert(!std::is_same_v<typename P::Field, RepeatedField>,
                "`Optional` doesn't accept `Repeated` ports as `Repeated` is "
                "already optional.");

  static_assert(
      !std::is_same_v<typename P::Specializer, GraphNodeSpecializer> &&
          !std::is_same_v<typename P::Specializer, GraphSpecializer>,
      "Incorrect `Optional` specialization.");

  using Field = OptionalField;
  using Contained = P;
  using P::P;

  using P::IsConnected;
};

// `Optional` connection for `Graph` or `Graph` node doesn't have `IsConnected`
// and only indicates that it's optional to connect.
template <typename P>
class Optional<
    P, std::enable_if_t<
           std::is_same_v<typename P::Specializer, GraphNodeSpecializer> ||
           std::is_same_v<typename P::Specializer, GraphSpecializer>>>
    : public P {
 public:
  static_assert(!std::is_same_v<typename P::Field, RepeatedField>,
                "`Optional` doesn't accept `Repeated` ports as `Repeated` is "
                "already optional.");

  using Field = OptionalField;
  using Contained = P;
  using P::P;
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_CONTRACT_H_
