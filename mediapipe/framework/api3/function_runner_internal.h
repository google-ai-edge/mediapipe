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

#ifndef MEDIAPIPE_FRAMEWORK_API3_FUNCTION_RUNNER_INTERNAL_H_
#define MEDIAPIPE_FRAMEWORK_API3_FUNCTION_RUNNER_INTERNAL_H_

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/output_stream_poller.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe::api3 {

template <typename TupleT, typename Func, size_t... I>
void ForEachOnTupleImpl(const TupleT& t, Func f, std::index_sequence<I...>) {
  (f(std::get<I>(t)), ...);
}

template <typename TupleT, typename F>
void ForEachOnTuple(const TupleT& t, F f) {
  return ForEachOnTupleImpl(
      t, f, std::make_index_sequence<std::tuple_size_v<TupleT>>{});
}

template <typename T>
constexpr bool kIsTupleV = false;

template <typename... Ts>
constexpr bool kIsTupleV<std::tuple<Ts...>> = true;

template <typename MaybeTupleT>
auto AsTuple(MaybeTupleT output) {
  if constexpr (kIsTupleV<decltype(output)>) {
    return output;
  } else {
    return std::tuple<decltype(output)>(std::move(output));
  }
}

template <typename T>
constexpr bool kIsStatusOr = false;

template <typename T>
constexpr bool kIsStatusOr<absl::StatusOr<T>> = true;

template <typename T>
auto AsStatusOr(T output) {
  if constexpr (kIsStatusOr<T>) {
    return output;
  } else {
    return absl::StatusOr<T>(std::move(output));
  }
}

template <typename T>
struct ToPacketTypeImpl {
  using type = Packet<T>;
};

template <typename... Ts>
struct ToPacketTypeImpl<std::tuple<Ts...>> {
  using type = std::tuple<Packet<Ts>...>;
};

template <typename T>
using ToPacketType = typename ToPacketTypeImpl<T>::type;

template <typename T>
struct UnwrapStatusOrImpl {
  using type = T;
};

template <typename T>
struct UnwrapStatusOrImpl<absl::StatusOr<T>> {
  using type = T;
};

template <typename T>
using UnwrapStatusOrType = typename UnwrapStatusOrImpl<T>::type;

template <typename T>
struct UnwrapStreamTypeImpl;

template <typename T>
struct UnwrapStreamTypeImpl<Stream<T>> {
  using type = T;
};

template <typename... Ts>
struct UnwrapStreamTypeImpl<std::tuple<Ts...>> {
  using type = std::tuple<typename UnwrapStreamTypeImpl<Ts>::type...>;
};

template <typename T>
using UnwrapStreamType = typename UnwrapStreamTypeImpl<T>::type;

template <typename T>
struct RemoveGenericGraphArgTypeImpl;

// Removes the first element from a tuple type.
template <typename Head, typename... Tail>
struct RemoveGenericGraphArgTypeImpl<std::tuple<Head, Tail...>> {
  using type = std::tuple<Tail...>;
};
template <typename T>
using RemoveGenericGraphArgType =
    typename RemoveGenericGraphArgTypeImpl<T>::type;

template <typename Signature>
struct BuildGraphFnRawSignature
    : BuildGraphFnRawSignature<decltype(&Signature::operator())> {};

// Specialization for free functions
template <typename R, typename... Args>
struct BuildGraphFnRawSignature<R (*)(Args...)> {
  using Out = UnwrapStreamType<UnwrapStatusOrType<R>>;
  using In = UnwrapStreamType<RemoveGenericGraphArgType<std::tuple<Args...>>>;
};

// Specialization for const member functions
template <typename C, typename R, typename... Args>
struct BuildGraphFnRawSignature<R (C::*)(Args...) const> {
  using Out = UnwrapStreamType<UnwrapStatusOrType<R>>;
  using In = UnwrapStreamType<RemoveGenericGraphArgType<std::tuple<Args...>>>;
};

// Specialization for non-const member functions
template <typename C, typename R, typename... Args>
struct BuildGraphFnRawSignature<R (C::*)(Args...)> {
  using Out = UnwrapStreamType<UnwrapStatusOrType<R>>;
  using In = UnwrapStreamType<RemoveGenericGraphArgType<std::tuple<Args...>>>;
};

// Specialization for std::function
template <typename R, typename... Args>
struct BuildGraphFnRawSignature<std::function<R(Args...)>> {
  using Out = UnwrapStreamType<UnwrapStatusOrType<R>>;
  using In = UnwrapStreamType<RemoveGenericGraphArgType<std::tuple<Args...>>>;
};

class ErrorCallback {
 public:
  void OnError(const absl::Status& error_status) {
    absl::MutexLock lock(&mutex_);
    errors_.push_back(error_status);
  }

  bool HasErrors() const {
    absl::MutexLock lock(&mutex_);
    return !errors_.empty();
  }

  std::vector<absl::Status> GetErrors() const {
    absl::MutexLock lock(&mutex_);
    return errors_;
  }

 private:
  mutable absl::Mutex mutex_;
  std::vector<absl::Status> errors_ ABSL_GUARDED_BY(mutex_);
};

class FunctionRunnerBase {
 public:
  FunctionRunnerBase(
      GenericGraph graph, std::unique_ptr<CalculatorGraph> calculator_graph,
      absl::flat_hash_map<int, std::string> input_names_map,
      absl::flat_hash_map<int, std::string> output_names_map,
      absl::flat_hash_map<int, OutputStreamPoller> output_pollers,
      std::shared_ptr<ErrorCallback> error_callback)
      : graph_(std::move(graph)),
        calculator_graph_(std::move(calculator_graph)),
        input_names_map_(std::move(input_names_map)),
        output_names_map_(std::move(output_names_map)),
        output_pollers_(std::move(output_pollers)),
        error_callback_(std::move(error_callback)) {}

  FunctionRunnerBase(FunctionRunnerBase&& other) = default;
  FunctionRunnerBase& operator=(FunctionRunnerBase&& other) = delete;

  ~FunctionRunnerBase() {
    if (calculator_graph_) {
      if (error_callback_->HasErrors()) {
        calculator_graph_->Cancel();
      } else {
        absl::Status status = calculator_graph_->CloseAllPacketSources();
        if (!status.ok()) ABSL_LOG(DFATAL) << status;
        status = calculator_graph_->WaitUntilDone();
        if (!status.ok()) ABSL_LOG(DFATAL) << status;
      }
    }
  }

 protected:
  mediapipe::Timestamp NextTimestamp() { return ++timestamp_; }

  absl::StatusOr<OutputStreamPoller*> GetOutputPoller(int index) {
    auto poller_iter = this->output_pollers_.find(index);
    RET_CHECK(poller_iter != this->output_pollers_.end());
    return &poller_iter->second;
  }

  GenericGraph graph_;
  std::unique_ptr<mediapipe::CalculatorGraph> calculator_graph_;
  absl::flat_hash_map<int, std::string> input_names_map_;
  absl::flat_hash_map<int, std::string> output_names_map_;
  absl::flat_hash_map<int, OutputStreamPoller> output_pollers_;
  std::shared_ptr<ErrorCallback> error_callback_;
  mediapipe::Timestamp timestamp_ = mediapipe::Timestamp(0);
};

template <typename... PacketTs>
absl::Status AddInputPackets(
    CalculatorGraph& calculator_graph,
    absl::flat_hash_map<int, std::string> input_names_map,
    const Timestamp& timestamp, PacketTs... inputs) {
  int input_index = 0;
  absl::Status status;
  // clang-format off
  ((
       status = [&]() -> absl::Status {
         // NOTE: currently supporting timestamp-less execution only.
         if (inputs.Timestamp() != Timestamp::Unset()) {
           return absl::InvalidArgumentError(absl::StrCat(
               "Timestamp for input [", input_index, "] is [",
               inputs.Timestamp().DebugString(), "], but must be Unset"));
         }
         return calculator_graph.AddPacketToInputStream(
             input_names_map[input_index++],
             inputs.AsLegacyPacket().At(timestamp));
       }(),
       status.ok()) &&
    ...);
  // clang-format on
  return status;
}

absl::StatusOr<mediapipe::Packet> GetOutputPacket(
    OutputStreamPoller& poller, const ErrorCallback& error_callback);

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_FUNCTION_RUNNER_INTERNAL_H_
