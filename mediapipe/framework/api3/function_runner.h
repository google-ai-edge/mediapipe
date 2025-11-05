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

#ifndef MEDIAPIPE_FRAMEWORK_API3_FUNCTION_RUNNER_H_
#define MEDIAPIPE_FRAMEWORK_API3_FUNCTION_RUNNER_H_

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api3/function_runner_internal.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/executor.h"
#include "mediapipe/framework/graph_service.h"
#include "mediapipe/framework/output_stream_poller.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/thread_pool_executor.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe::api3 {

template <typename BuildGraphFnT>
class FunctionRunnerBuilder;

namespace internal_function_runner {

// `FunctionRunnerImpl` is used to implement the `Run` method.
// This indirection is necessary to allow specializing on the variadic input
// types (`InputPacketTs...`) which are part of a class template parameter
// (`InputT`), rather than trying to deduce them within the `Run` method itself.
template <typename BuildGraphFnT, typename OutputT, typename InputT>
class FunctionRunnerImpl;

template <typename BuildGraphFnT, typename OutputT, typename... InputPacketTs>
class FunctionRunnerImpl<BuildGraphFnT, OutputT, std::tuple<InputPacketTs...>>
    : public FunctionRunnerBase {
 public:
  // - Adds all provided input packets
  // - Waits for graph work completion
  // - Polls and returns the output packet(s)
  absl::StatusOr<OutputT> Run(InputPacketTs... inputs) {
    mediapipe::Timestamp timestamp = this->NextTimestamp();
    MP_RETURN_IF_ERROR(AddInputPackets(*this->calculator_graph_,
                                       this->input_names_map_, timestamp,
                                       inputs...));
    MP_RETURN_IF_ERROR(this->calculator_graph_->WaitUntilIdle());

    if constexpr (kIsTupleV<OutputT>) {
      OutputT output;
      MP_RETURN_IF_ERROR(GetOutputPackets(
          std::make_index_sequence<std::tuple_size_v<OutputT>>(), output));
      return output;
    } else {
      MP_ASSIGN_OR_RETURN(OutputStreamPoller * poller,
                          this->GetOutputPoller(0));
      MP_ASSIGN_OR_RETURN(mediapipe::Packet packet,
                          GetOutputPacket(*poller, *this->calculator_graph_));
      return WrapLegacyPacket<typename OutputT::PayloadT>(std::move(packet));
    }
  }

 private:
  using FunctionRunnerBase::FunctionRunnerBase;

  template <size_t... Is, typename TupleT>
  absl::Status GetOutputPackets(std::index_sequence<Is...>, TupleT& output) {
    absl::Status status = absl::OkStatus();
    ((
         status = [&]() -> absl::Status {
           MP_ASSIGN_OR_RETURN(OutputStreamPoller * poller,
                               this->GetOutputPoller(Is));
           MP_ASSIGN_OR_RETURN(
               mediapipe::Packet packet,
               GetOutputPacket(*poller, *this->calculator_graph_));
           using CurrentOutputPacketT = std::tuple_element_t<Is, OutputT>;
           MP_ASSIGN_OR_RETURN(
               std::get<Is>(output),
               WrapLegacyPacket<typename CurrentOutputPacketT::PayloadT>(
                   std::move(packet)));
           return absl::OkStatus();
         }(),
         status.ok()) &&
     ...);
    return status;
  }

  friend class FunctionRunnerBuilder<BuildGraphFnT>;
};

}  // namespace internal_function_runner

// This runner enables running MediaPipe graph as a function.
//
// The intended usage is:
// ```
//   /* Creating the runner from graph builder lambda. */
//   MP_ASSIGN_OR_RETURN(
//      auto runner,
//      Runner::For([](GenericGraph& graph,
//                     Stream<ImageFrame> input_image) -> Stream<ImageFrame> {
//        Stream<Tensor> input_tensor = [&] {
//          auto& node = graph.AddNode<ImageToTensorNode>();
//          node.image.Set(input_image);
//          return node.tensor.Get();
//        }();
//
//        /* Inference node. */
//        Stream<Tensor> output_tensor = ...;
//
//        /* Tensor to image conversion node. */
//        Stream<ImageFrame> output_image = ...;
//
//        return output_image;
//      }).Create());
//
//  /* Running the graph. */
//  MP_ASSIGN_OR_RETURN(Packet<ImageFrame> output,
//                   runner.Run(MakePacket<ImageFrame>(...)));
// ```
//
// If you need to keep runner across invocations.
// ```
//   MP_ASSIGN_OR_RETURN(FunctionRunner<decltype(lambda)>, Runner::For...);
//   MP_ASSIGN_OR_RETURN(FunctionRunner<decltype(&FreeFunction)>,
//   Runner::For...); MP_ASSIGN_OR_RETURN(FunctionRunner<GraphBuilderObject>,
//   Runner::For...);
//
//   // Where GraphBuilderObject can be
//   struct GraphBuilderObject {
//     Stream<GpuBuffer> operator()(GenericGraph& graph,
//                                  Stream<GpuBuffer> input) { ... }
//   };
// ```
//
// - `Runner::For(...)` returns a `FunctionRunnerBuilder` that allows fine
//   tuning your runner.
// - `FunctionRunnerBuilder::Create()` returns the runner.
// - `FunctionRunner::Run(...)` runs the graph for provided input packets and
//   returns output packet(s).
//
// More details in `Runner` and `FunctionRunnerBuilder` classes.
template <
    typename BuildGraphFnT,
    typename BaseT = internal_function_runner::FunctionRunnerImpl<
        BuildGraphFnT,
        ToPacketType<typename BuildGraphFnRawSignature<BuildGraphFnT>::Out>,
        ToPacketType<typename BuildGraphFnRawSignature<BuildGraphFnT>::In>>>
class FunctionRunner : public BaseT {
 private:
  using BaseT::BaseT;
};

template <typename BuildGraphFnT>
class FunctionRunnerBuilder {
 public:
  using RawSignatureT = BuildGraphFnRawSignature<BuildGraphFnT>;

  // Set graph service for the MediaPipe graph.
  template <typename T>
  FunctionRunnerBuilder& SetService(const GraphService<T>& service,
                                    std::shared_ptr<T> object) {
    services_[&service] =
        mediapipe::MakePacket<std::shared_ptr<T>>(std::move(object));
    return *this;
  }

  // Sets the default executor for MediaPipe graph.
  //
  // NOTE: this is optional, and the default executor set for the function
  //   runner has just a single thread.
  FunctionRunnerBuilder& SetDefaultExecutor(
      std::shared_ptr<Executor> default_executor) {
    default_executor_ = std::move(default_executor);
    return *this;
  }

  // Creates the graph runner according to the provided graph builder function
  // and initializes using all provided parameters.
  //
  // The runner is ready to be used as following for single output:
  // ```
  //   MP_ASSIGN_OR_RETURN(Packet<...> p, runner.Run(...input packets...));
  // ```
  // and multiple outputs:
  // ```
  //   MP_ASSIGN_OR_RETURN((auto [p1, p2]), runner.Run(...input packets...));
  // ```
  //
  // Refer `Runner::For` for more details on builder graph function and
  // corresponding runners.
  absl::StatusOr<FunctionRunner<BuildGraphFnT>> Create() {
    // Build the graph using provided build graph function.
    // (The function can return StatusOr or not, std::tuple or not.)
    GenericGraph graph;
    MP_ASSIGN_OR_RETURN(
        auto raw_output,
        AsStatusOr(InvokeBuildGraphFn(std::move(build_graph_fn_), graph)));
    auto output = AsTuple(std::move(raw_output));

    // Connect output stream(s) to graph outputs, generate/collect output stream
    // names.
    absl::flat_hash_map<int, std::string> output_names_map;
    int output_index = 0;
    static constexpr absl::string_view kOutputPrefix = "__runner_out_";
    ForEachOnTuple(output, [&](auto stream) {
      if (stream.Name().empty()) {
        std::string name = absl::StrCat(kOutputPrefix, output_index);
        stream.SetName(name);
      }
      stream.GetBase()->ConnectTo(
          graph.builder_.Out(/* empty output tag */ "").At(output_index));
      output_names_map[output_index] = stream.Name();
      ++output_index;
    });

    // Generate/collect input stream names.
    absl::flat_hash_map<int, std::string> input_names_map;
    const int num_inputs = std::tuple_size_v<typename RawSignatureT::In>;
    static constexpr absl::string_view kInputPrefix = "__runner_in_";
    for (int i = 0; i < num_inputs; ++i) {
      auto& input = graph.builder_.In(/*empty input tag*/ "").At(i);
      if (input.name.empty()) {
        std::string name = absl::StrCat(kInputPrefix, i);
        input.SetName(name);
      }
      input_names_map[i] = input.name;
    }

    // Create graph config and ensure sync execution.
    MP_ASSIGN_OR_RETURN(CalculatorGraphConfig config, graph.GetConfig());
    VLOG(1) << "Graph config:\n" << config.DebugString();

    auto calculator_graph = std::make_unique<CalculatorGraph>();

    // Default to a single thread execution.
    std::shared_ptr<Executor> default_executor = std::move(default_executor_);
    if (!default_executor) {
#ifdef __EMSCRIPTEN__
      auto* executor = config.add_executor();
      executor->set_type("ApplicationThreadExecutor");
      executor->set_name("");
#else
      default_executor =
          std::make_shared<ThreadPoolExecutor>(/*num_threads*/ 1);
#endif
    }

    if (default_executor) {
      MP_RETURN_IF_ERROR(calculator_graph->SetExecutor(
          /*default executor name*/ "", std::move(default_executor)));
    }

    for (const auto& [key, value] : services_) {
      MP_RETURN_IF_ERROR(calculator_graph->SetServicePacket(*key, value));
    }
    MP_RETURN_IF_ERROR(calculator_graph->Initialize(std::move(config)));

    // Setup output pollers for the requested output streams.
    absl::flat_hash_map<int, OutputStreamPoller> output_pollers;
    for (const auto& [index, name] : output_names_map) {
      MP_ASSIGN_OR_RETURN(OutputStreamPoller poller,
                          calculator_graph->AddOutputStreamPoller(
                              name, /*observe_timestamp_bounds=*/true));
      const bool inserted =
          output_pollers.try_emplace(index, std::move(poller)).second;
      RET_CHECK(inserted);
    }

    MP_RETURN_IF_ERROR(calculator_graph->StartRun({}));

    return FunctionRunner<BuildGraphFnT>(
        std::move(graph), std::move(calculator_graph),
        std::move(input_names_map), std::move(output_names_map),
        std::move(output_pollers));
  }

 private:
  explicit FunctionRunnerBuilder(BuildGraphFnT fn)
      : build_graph_fn_(std::move(fn)) {}

  static auto InvokeBuildGraphFn(BuildGraphFnT build_graph_fn,
                                 GenericGraph& graph) {
    using InputsT = typename RawSignatureT::In;
    return InvokeBuildGraphFnImpl<InputsT>(
        std::move(build_graph_fn), graph,
        std::make_index_sequence<std::tuple_size_v<InputsT>>());
  }

  template <typename InputsT, size_t... Indices>
  static auto InvokeBuildGraphFnImpl(BuildGraphFnT build_graph_fn,
                                     GenericGraph& graph,
                                     std::index_sequence<Indices...>) {
    return build_graph_fn(
        graph, Stream<std::tuple_element_t<Indices, InputsT>>(
                   graph.builder_.In(/*empty input tag*/ "").At(Indices))...);
  }

  BuildGraphFnT build_graph_fn_;
  absl::flat_hash_map<const GraphServiceBase*, mediapipe::Packet> services_;
  std::shared_ptr<Executor> default_executor_;

  friend class Runner;
};

class Runner {
 public:
  // Creates a builder for synchronous runner according to the provided graph
  // builder function.
  //
  // For example:
  //
  // +----------------------------------------------------------------------+
  // |                                                                      |
  // |   1. Single input, single output use case.                           |
  // |                                                                      |
  // +----------------------------------------------------------------------+
  //
  // Graph builder function is:
  // ```
  //   [](GenericGraph& graph, Stream<int> input) -> Stream<int> {
  //       ...
  //   }
  // ```
  //
  // Returns builder for the runner with `Run` function:
  // ```
  //   absl::StatusOr<Packet<int>> Run(Packet<int> input);
  // ```
  //
  // +----------------------------------------------------------------------+
  // |                                                                      |
  // |   2. Multiple inputs, single output use case.                        |
  // |                                                                      |
  // +----------------------------------------------------------------------+
  //
  // Graph builder function is:
  // ```
  //   [](GenericGraph& graph, Stream<int> a, Stream<float> b) -> Stream<...> {
  //       ...
  //   }
  // ```
  // Returns builder for the runner with `Run` function:
  // ```
  //   absl::StatusOr<Packet<...>> Run(Packet<int> a, Packet<float> b);
  // ```
  //
  // +----------------------------------------------------------------------+
  // |                                                                      |
  // |   3. Multiple outputs.                                               |
  // |                                                                      |
  // +----------------------------------------------------------------------+
  //
  // Multiple outputs are supported with `std::tuple`. If graph builder function
  // is:
  // ```
  //   [](GenericGraph& graph, ...) -> std::tuple<Stream<int>, Stream<float>> {
  //       ...
  //   }
  // ```
  //
  // Returns builder for the runner with `Run` function:
  // ```
  //   absl::StatusOr<std::tuple<Packet<int>, Packet<float>> Run(...);
  // ```
  //
  // +----------------------------------------------------------------------+
  // |                                                                      |
  // |   3. absl::StatusOr<> support.                                       |
  // |                                                                      |
  // +----------------------------------------------------------------------+
  //
  // Your graph builder function can use absl::StatusOr as following:
  // ```
  //   [](GenericGraph& graph, ...) -> absl::StatusOr<Stream<>>
  //   [](GenericGraph& graph, ...) -> absl::StatusOr<std::tuple<Stream<>, ...>>
  // ```
  //
  // And in case of failure, the error status will be returned by builder's
  // `Create` function.
  template <typename BuildGraphFnT>
  static FunctionRunnerBuilder<BuildGraphFnT> For(BuildGraphFnT fn) {
    return FunctionRunnerBuilder(std::move(fn));
  }
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_FUNCTION_RUNNER_H_
