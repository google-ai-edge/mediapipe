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
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api3/function_runner_internal.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/packet.h"
#include "mediapipe/framework/api3/side_packet.h"
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
//   MP_ASSIGN_OR_RETURN(FunctionRunner<decltype(lambda)> runner,
//                    Runner::For(std::move(lambda)));
//   MP_ASSIGN_OR_RETURN(FunctionRunner<decltype(&FreeFunction)> runner,
//                    Runner::For(FreeFunction));
//   MP_ASSIGN_OR_RETURN(FunctionRunner<GraphBuilderObject> runner,
//                    Runner::For(GraphBuilderObject()));
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

// A dedicated class which can be used in graph builder classes / functions in
// place of `GenericGraph` to enable input side packets to the graph.
//
// For example:
// ```
//   [](FunctionGraphBuilder& builder, Stream<...> in) -> Stream<...> {
//     SidePacket<int> side_in =
//         builder.side_packets.AddSidePacket(MakePacket<int>(...));
//     GenericGraph& graph = builder.graph;
//     ...
//   }
// ```
struct FunctionGraphBuilder {
  class SidePackets {
   public:
    SidePackets(GenericGraph& graph,
                std::vector<mediapipe::Packet>& side_packets)
        : graph_(graph), side_packets_(side_packets) {};

    template <typename T>
    SidePacket<T> AddSidePacket(Packet<T> packet) {
      int index = side_packets_.size();
      auto& side_source = graph_.builder_.SideIn("").At(index);
      side_packets_.push_back(std::move(packet).ConsumeAsLegacyPacket());
      return SidePacket<T>(side_source);
    }

   private:
    GenericGraph& graph_;
    std::vector<mediapipe::Packet>& side_packets_;
  };

  FunctionGraphBuilder(GenericGraph& graph,
                       std::vector<mediapipe::Packet>& side_packets)
      : graph(graph), side_packets(SidePackets(graph, side_packets)) {};

  GenericGraph& graph;
  SidePackets side_packets;
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
    std::vector<mediapipe::Packet> side_packets;
    FunctionGraphBuilder builder(graph, side_packets);
    MP_ASSIGN_OR_RETURN(auto raw_output,
                        AsStatusOr(InvokeBuildGraphFn(builder)));
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
        input.SetName(absl::StrCat(kInputPrefix, i));
      }
      input_names_map[i] = input.name;
    }

    std::map<std::string, mediapipe::Packet> side_packets_mapping;
    if (!side_packets.empty()) {
      static constexpr absl::string_view kSideInputPrefix = "__runner_side_in_";
      for (int i = 0; i < side_packets.size(); ++i) {
        auto& side_in =
            graph.builder_.SideIn(/*empty side input tag*/ "").At(i);
        if (side_in.name.empty()) {
          side_in.SetName(absl::StrCat(kSideInputPrefix, i));
        }
        side_packets_mapping[side_in.name] = side_packets[i];
      }
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

    MP_RETURN_IF_ERROR(
        calculator_graph->StartRun(std::move(side_packets_mapping)));

    return FunctionRunner<BuildGraphFnT>(
        std::move(graph), std::move(calculator_graph),
        std::move(input_names_map), std::move(output_names_map),
        std::move(output_pollers));
  }

 private:
  explicit FunctionRunnerBuilder(BuildGraphFnT fn)
      : build_graph_fn_(std::move(fn)) {}

  auto InvokeBuildGraphFn(FunctionGraphBuilder& builder) {
    using InputsT = typename RawSignatureT::In;
    return InvokeBuildGraphFnImpl<InputsT>(
        build_graph_fn_, builder,
        std::make_index_sequence<std::tuple_size_v<InputsT>>());
  }

  template <typename InputsT, size_t... Indices>
  auto InvokeBuildGraphFnImpl(BuildGraphFnT& build_graph_fn,
                              FunctionGraphBuilder& builder,
                              std::index_sequence<Indices...>) {
    if constexpr (std::is_invocable_v<
                      BuildGraphFnT&, FunctionGraphBuilder&,
                      Stream<std::tuple_element_t<Indices, InputsT>>...>) {
      return build_graph_fn(
          builder, Stream<std::tuple_element_t<Indices, InputsT>>(
                       builder.graph.builder_.In(/*empty input tag*/ "")
                           .At(Indices))...);
    } else {
      return build_graph_fn(
          builder.graph, Stream<std::tuple_element_t<Indices, InputsT>>(
                             builder.graph.builder_.In(/*empty input tag*/ "")
                                 .At(Indices))...);
    }
  }

  BuildGraphFnT build_graph_fn_;
  absl::flat_hash_map<const GraphServiceBase*, mediapipe::Packet> services_;
  std::shared_ptr<Executor> default_executor_;

  friend class Runner;
};

class Runner {
 public:
  // Creates a builder for synchronous runner according to the provided graph
  // builder function/object.
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
  // Graph builder object is:
  // ```
  //   struct GraphBuilder {
  //     [](GenericGraph& graph, Stream<int> input) -> Stream<int> {
  //       ...
  //     }
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
  // +----------------------------------------------------------------------+
  // |                                                                      |
  // |   4. Input side packets                                              |
  // |                                                                      |
  // +----------------------------------------------------------------------+
  //
  // Input side packets are supported by using `FunctionGraphBuilder&` instead
  // of `GenericGraph&` as the first argument in builder function/object:
  // ```
  //   [](FunctionGraphBuilder& builder, Stream<...> in) -> Stream<...> {
  //     SidePacket<int> side_in =
  //         builder.side_packets.AddSidePacket(MakePacket<int>(...));
  //     GenericGraph& graph = builder.graph;
  //     ...
  //   }
  // ```
  // or for builder objects:
  // ```
  //   struct GraphBuilder {
  //     Stream<...> operator()(FunctionGraphBuilder& builder, Stream<...> in) {
  //       SidePacket<int> side_in =
  //           builder.side_packets.AddSidePacket(MakePacket<int>(...));
  //       GenericGraph& graph = builder.graph;
  //       ...
  //     }
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
  // |   5. absl::StatusOr<> support.                                       |
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
