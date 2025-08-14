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

#include "mediapipe/framework/api3/function_runner.h"

#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/api3/calculator_contract.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/api3/packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/graph_service.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/thread_pool_executor.h"

namespace mediapipe::api3 {
namespace {

constexpr absl::string_view kMultiplyBy2NodeName = "MultiplyBy2";
struct MultiplyBy2Node : Node<kMultiplyBy2NodeName> {
  template <typename S>
  struct Contract {
    Input<S, int> in{"IN"};
    Output<S, int> out{"OUT"};
  };
};

class MultiplyBy2NodeImpl
    : public Calculator<MultiplyBy2Node, MultiplyBy2NodeImpl> {
 public:
  absl::Status Process(CalculatorContext<MultiplyBy2Node>& cc) final {
    RET_CHECK(cc.in);
    cc.out.Send(cc.in.GetOrDie() * 2);
    return absl::OkStatus();
  }
};

constexpr absl::string_view kAdd10Node = "Add10";
struct Add10Node : Node<kAdd10Node> {
  template <typename S>
  struct Contract {
    Input<S, int> in{"IN"};
    Output<S, int> out{"OUT"};
  };
};

class Add10NodeImpl : public Calculator<Add10Node, Add10NodeImpl> {
 public:
  absl::Status Process(CalculatorContext<Add10Node>& cc) final {
    RET_CHECK(cc.in);
    cc.out.Send(cc.in.GetOrDie() + 10);
    return absl::OkStatus();
  }
};

TEST(FunctionRunnerTest, RunsSingleGraphToMultiplyBy2Add10) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([](GenericGraph& graph, Stream<int> in) -> Stream<int> {
        Stream<int> multiplication = [&]() {
          auto& node = graph.AddNode<MultiplyBy2Node>();
          node.in.Set(in);
          return node.out.Get();
        }();

        Stream<int> addition = [&]() {
          auto& node = graph.AddNode<Add10Node>();
          node.in.Set(multiplication);
          return node.out.Get();
        }();

        return addition;
      }).Create());

  for (auto [input, expected] :
       {std::pair(20, 50), std::pair(40, 90), std::pair(100, 210)}) {
    MP_ASSERT_OK_AND_ASSIGN(Packet<int> output,
                            runner.Run(MakePacket<int>(input)));
    EXPECT_EQ(output.GetOrDie(), expected);
  }
}

TEST(FunctionRunnerTest, WorksForMultiGraphUseCase) {
  auto shared_executor =
      std::make_shared<ThreadPoolExecutor>(/*num_threads*/ 1);

  MP_ASSERT_OK_AND_ASSIGN(
      auto multiply_by_2,
      Runner::For([](GenericGraph& graph, Stream<int> in) -> Stream<int> {
        auto& node = graph.AddNode<MultiplyBy2Node>();
        node.in.Set(in);
        return node.out.Get();
      })
          .SetDefaultExecutor(shared_executor)
          .Create());

  MP_ASSERT_OK_AND_ASSIGN(
      auto add_10,
      Runner::For([](GenericGraph& graph, Stream<int> in) -> Stream<int> {
        auto& node = graph.AddNode<Add10Node>();
        node.in.Set(in);
        return node.out.Get();
      })
          .SetDefaultExecutor(shared_executor)
          .Create());

  for (int value : {20, 40, 100}) {
    MP_ASSERT_OK_AND_ASSIGN(Packet<int> p,
                            multiply_by_2.Run(MakePacket<int>(value)));
    constexpr int kNumAdditions = 5;
    for (int i = 0; i < kNumAdditions; ++i) {
      MP_ASSERT_OK_AND_ASSIGN(p, add_10.Run(std::move(p)));
    }
    EXPECT_EQ(p.GetOrDie(), value * 2 + kNumAdditions * 10);
  }
}

// The default mode is no timestamps.
TEST(FunctionRunnerTest, FailsWhenTimestampIsProvided) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([](GenericGraph& graph, Stream<int> in) -> Stream<int> {
        auto& node = graph.AddNode<MultiplyBy2Node>();
        node.in.Set(in);
        return node.out.Get();
      }).Create());

  EXPECT_THAT(runner.Run(MakePacket<int>(20).At(Timestamp(10))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("must be Unset")));
}

TEST(FunctionRunnerTest, WorksForMultipleOutputs) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([](GenericGraph& graph,
                     Stream<int> x) -> std::tuple<Stream<int>, Stream<int>> {
        Stream<int> multiplication = [&]() {
          auto& node = graph.AddNode<MultiplyBy2Node>();
          node.in.Set(x);
          return node.out.Get();
        }();

        Stream<int> addition = [&]() {
          auto& node = graph.AddNode<Add10Node>();
          node.in.Set(x);
          return node.out.Get();
        }();

        return {multiplication, addition};
      }).Create());

  for (int value : {20, 40}) {
    MP_ASSERT_OK_AND_ASSIGN((auto [multiplied, adjusted]),
                            runner.Run(MakePacket<int>(value)));
    EXPECT_EQ(multiplied.GetOrDie(), value * 2);
    EXPECT_EQ(adjusted.GetOrDie(), value + 10);
  }
}

TEST(FunctionRunnerTest, WorksForMultipleInputsAndOutputs) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([](GenericGraph& graph, Stream<int> a,
                     Stream<int> b) -> std::tuple<Stream<int>, Stream<int>> {
        // NOTE: Setting input names is available, but not required.
        a.SetName("a");
        b.SetName("b");

        Stream<int> multiplied_a = [&]() {
          auto& node = graph.AddNode<MultiplyBy2Node>();
          node.in.Set(a);
          return node.out.Get();
        }();

        Stream<int> adjusted_b = [&]() {
          auto& node = graph.AddNode<Add10Node>();
          node.in.Set(b);
          return node.out.Get();
        }();

        // NOTE: Setting output names is available, but not required.
        return {multiplied_a.SetName("multiplied_a"),
                adjusted_b.SetName("adjusted_b")};
      }).Create());

  for (auto [a, b] : {std::pair(20, 100), std::pair(40, 50)}) {
    MP_ASSERT_OK_AND_ASSIGN((auto [multiplied_a, adjusted_b]),
                            runner.Run(MakePacket<int>(a), MakePacket<int>(b)));
    EXPECT_EQ(multiplied_a.GetOrDie(), a * 2);
    EXPECT_EQ(adjusted_b.GetOrDie(), b + 10);
  }
}

constexpr GraphService<std::string> kTestService{"kTestService"};

constexpr absl::string_view kServiceValueOnTickNodeName =
    "ServiceValueOnTickNode";
struct ServiceValueOnTickNode : Node<kServiceValueOnTickNodeName> {
  template <typename S>
  struct Contract {
    Input<S, int> tick{"TICK"};
    Output<S, std::string> service_value{"SERVICE_VALUE"};

    static absl::Status UpdateContract(
        CalculatorContract<ServiceValueOnTickNode>& cc) {
      cc.UseService(kTestService);
      return absl::OkStatus();
    }
  };
};

class ServiceValueOnTickNodeImpl
    : public Calculator<ServiceValueOnTickNode, ServiceValueOnTickNodeImpl> {
 public:
  absl::Status Process(CalculatorContext<ServiceValueOnTickNode>& cc) final {
    RET_CHECK(cc.Service(kTestService).IsAvailable());
    cc.service_value.Send(cc.Service(kTestService).GetObject());
    return absl::OkStatus();
  }
};

TEST(FunctionRunnerTest, SupportsServices) {
  constexpr absl::string_view kServiceValue = "service_value";
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For([](GenericGraph& graph,
                                  Stream<int> tick) -> Stream<std::string> {
                     auto& node = graph.AddNode<ServiceValueOnTickNode>();
                     node.tick.Set(tick);
                     return node.service_value.Get();
                   })
                       .SetService(kTestService,
                                   std::make_shared<std::string>(kServiceValue))
                       .Create());

  Packet<std::string> service_value;
  MP_ASSERT_OK_AND_ASSIGN(service_value, runner.Run(MakePacket<int>(0)));
  EXPECT_EQ(service_value.GetOrDie(), kServiceValue);
}

TEST(FunctionRunnerTest, SupportsAnyInvocableAsBuildGraphFunction) {
  constexpr absl::string_view kServiceValue = "service_value";
  absl::AnyInvocable<Stream<std::string>(GenericGraph&, Stream<int>)>
      build_graph_fn =
          [](GenericGraph& graph, Stream<int> tick) -> Stream<std::string> {
    auto& node = graph.AddNode<ServiceValueOnTickNode>();
    node.tick.Set(tick);
    return node.service_value.Get();
  };
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For(std::move(build_graph_fn))
                       .SetService(kTestService,
                                   std::make_shared<std::string>(kServiceValue))
                       .Create());

  Packet<std::string> service_value;
  MP_ASSERT_OK_AND_ASSIGN(service_value, runner.Run(MakePacket<int>(0)));
  EXPECT_EQ(service_value.GetOrDie(), kServiceValue);
}

Stream<std::string> GetServiceValue(GenericGraph& graph, Stream<int> tick) {
  auto& node = graph.AddNode<ServiceValueOnTickNode>();
  node.tick.Set(tick);
  return node.service_value.Get();
}

TEST(FunctionRunnerTest, SupportsFreeGraphBuilderFunction) {
  constexpr absl::string_view kServiceValue = "service_value";
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For(GetServiceValue)
                       .SetService(kTestService,
                                   std::make_shared<std::string>(kServiceValue))
                       .Create());

  Packet<std::string> service_value;
  MP_ASSERT_OK_AND_ASSIGN(service_value, runner.Run(MakePacket<int>(0)));
  EXPECT_EQ(service_value.GetOrDie(), kServiceValue);
}

TEST(FunctionRunnerTest, SupportsStatusOrBuilderFunctions) {
  constexpr absl::string_view kServiceValue = "service_value";
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([](GenericGraph& graph,
                     Stream<int> tick) -> absl::StatusOr<Stream<std::string>> {
        auto& node = graph.AddNode<ServiceValueOnTickNode>();
        node.tick.Set(tick);
        return node.service_value.Get();
      })
          .SetService(kTestService,
                      std::make_shared<std::string>(kServiceValue))
          .Create());

  Packet<std::string> service_value;
  MP_ASSERT_OK_AND_ASSIGN(service_value, runner.Run(MakePacket<int>(0)));
  EXPECT_EQ(service_value.GetOrDie(), kServiceValue);
}

TEST(FunctionRunnerTest, SupportsStatusOrBuilderFunctionsWhenNotOk) {
  constexpr absl::string_view kServiceValue = "service_value";
  EXPECT_THAT(
      Runner::For([](GenericGraph& graph,
                     Stream<int> tick) -> absl::StatusOr<Stream<std::string>> {
        return absl::InternalError("expected");
      })
          .SetService(kTestService,
                      std::make_shared<std::string>(kServiceValue))
          .Create(),
      StatusIs(absl::StatusCode::kInternal, testing::HasSubstr("expected")));
}

TEST(FunctionRunnerTest, SupportsStatusOrMultiOutputBuilderFunctions) {
  constexpr absl::string_view kServiceValue = "service_value";
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For(
          [](GenericGraph& graph, Stream<int> tick)
              -> absl::StatusOr<
                  std::tuple<Stream<std::string>, Stream<std::string>>> {
            Stream<std::string> service_value1 = [&]() {
              auto& node = graph.AddNode<ServiceValueOnTickNode>();
              node.tick.Set(tick);
              return node.service_value.Get();
            }();

            Stream<std::string> service_value2 = [&]() {
              auto& node = graph.AddNode<ServiceValueOnTickNode>();
              node.tick.Set(tick);
              return node.service_value.Get();
            }();

            return {{service_value1, service_value2}};
          })
          .SetService(kTestService,
                      std::make_shared<std::string>(kServiceValue))
          .Create());

  MP_ASSERT_OK_AND_ASSIGN((auto [value1, value2]),
                          runner.Run(MakePacket<int>(0)));
  EXPECT_EQ(value1.GetOrDie(), kServiceValue);
  EXPECT_EQ(value2.GetOrDie(), kServiceValue);
}

TEST(FunctionRunnerTest,
     SupportsStatusOrMultiOutputBuilderFunctionsWhenNotOkAnd) {
  EXPECT_THAT(
      Runner::For(
          [](GenericGraph& graph, Stream<int> tick)
              -> absl::StatusOr<
                  std::tuple<Stream<std::string>, Stream<std::string>>> {
            return absl::InternalError("expected");
          })
          .SetService(kTestService,
                      std::make_shared<std::string>("service_value"))
          .Create(),
      StatusIs(absl::StatusCode::kInternal, testing::HasSubstr("expected")));
}

constexpr absl::string_view kAlwaysFailingNode = "AlwaysFailingNode";
struct AlwaysFailingNode : Node<kAlwaysFailingNode> {
  template <typename S>
  struct Contract {
    Input<S, int> tick{"TICK"};
    Output<S, int> output{"UNUSED"};
  };
};

class AlwaysFailingNodeImpl
    : public Calculator<AlwaysFailingNode, AlwaysFailingNodeImpl> {
 public:
  absl::Status Process(CalculatorContext<AlwaysFailingNode>& cc) final {
    return absl::UnimplementedError("unimplemented is expected");
  }
};

TEST(FunctionRunnerTest, ReturnsCorrectFailureStatus) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([](GenericGraph& graph, Stream<int> tick) -> Stream<int> {
        auto& node = graph.AddNode<AlwaysFailingNode>();
        node.tick.Set(tick);
        return node.output.Get();
      }).Create());

  EXPECT_THAT(runner.Run(MakePacket<int>(0)),
              StatusIs(absl::StatusCode::kUnimplemented,
                       testing::HasSubstr("unimplemented is expected")));
}

}  // namespace
}  // namespace mediapipe::api3
