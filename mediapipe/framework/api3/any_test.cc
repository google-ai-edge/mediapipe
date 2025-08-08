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

#include "mediapipe/framework/api3/any.h"

#include <string>
#include <tuple>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/api3/calculator_contract.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/api3/side_packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/tool/options_map.h"

namespace mediapipe::api3 {
namespace {

constexpr absl::string_view kTickNodeName = "TemplatedTickNode";
// Intentionally using template for Tick. For all types, the implementation will
// be the same - through Any type.
template <typename TickT>
struct TemplatedTickNode : Node<kTickNodeName> {
  template <typename S>
  struct Contract {
    Input<S, TickT> tick{"TICK"};
    Output<S, int> out{"OUT"};
  };
};

// Common implementation for the calculator.
class TickNodeImpl : public Calculator<TemplatedTickNode<Any>, TickNodeImpl> {
 public:
  absl::Status Process(CalculatorContext<TemplatedTickNode<Any>>& cc) final {
    cc.out.Send(42);
    return absl::OkStatus();
  };
};

struct SomeTick {};

TEST(AnyTest, CanUseAnyForNodeTickInputImplementation) {
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, []() {
    Graph<TemplatedTickNode<SomeTick>::Contract> graph;

    Stream<SomeTick> tick = graph.tick.Get().SetName("tick");

    Stream<int> out = [&]() {
      auto& node = graph.AddNode<TemplatedTickNode<SomeTick>>();
      node.tick.Set(tick);
      return node.out.Get();
    }();

    graph.out.Set(out.SetName("out"));

    return graph.GetConfig();
  }());

  CalculatorGraph calculator_graph;
  MP_ASSERT_OK(calculator_graph.Initialize(std::move(config)));
  mediapipe::Packet output;
  MP_ASSERT_OK(calculator_graph.ObserveOutputStream(
      "out", [&](const mediapipe::Packet& p) {
        output = p;
        return absl::OkStatus();
      }));
  MP_ASSERT_OK(calculator_graph.StartRun({}));

  // Using tick of type SomeTick as in the graph builder.
  MP_ASSERT_OK(calculator_graph.AddPacketToInputStream(
      "tick", mediapipe::MakePacket<SomeTick>().At(Timestamp(1000))));
  MP_ASSERT_OK(calculator_graph.WaitUntilIdle());

  ASSERT_FALSE(output.IsEmpty());
  EXPECT_EQ(output.Get<int>(), 42);

  output = mediapipe::Packet();  // Reset packet.

  // Using a tick of a different type - should still work as Any is the
  // underlying implementation and `CalcualtorGraphConfig` doesn't preserve
  // builder's type restrictions.
  MP_ASSERT_OK(calculator_graph.AddPacketToInputStream(
      "tick", mediapipe::MakePacket<std::string>(
                  "not the same type as in graph builder")
                  .At(Timestamp(2000))));
  MP_ASSERT_OK(calculator_graph.WaitUntilIdle());

  ASSERT_FALSE(output.IsEmpty());
  EXPECT_EQ(output.Get<int>(), 42);
}

constexpr absl::string_view kAnyTickNodeName = "AnyTickNode";
struct AnyTickNode : Node<kAnyTickNodeName> {
  template <typename S>
  struct Contract {
    Input<S, Any> tick{"TICK"};
    Output<S, int> out{"OUT"};
  };
};

// Common implementation for the calculator.
class AnyTickNodeImpl : public Calculator<AnyTickNode, AnyTickNodeImpl> {
 public:
  absl::Status Process(CalculatorContext<AnyTickNode>& cc) final {
    cc.out.Send(42);
    return absl::OkStatus();
  };
};

template <typename S>
struct SomeTickGraphContract {
  Input<S, SomeTick> tick{"IN"};
  Output<S, int> out{"OUT"};
};

TEST(AnyTest, CanUseAnyForNodeTickInputInterfaceAndImplementation) {
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, []() {
    Graph<SomeTickGraphContract> graph;

    Stream<SomeTick> tick = graph.tick.Get().SetName("tick");

    Stream<int> out = [&]() {
      auto& node = graph.AddNode<AnyTickNode>();
      node.tick.Set(tick.Cast<Any>());
      return node.out.Get();
    }();

    graph.out.Set(out.SetName("out"));

    return graph.GetConfig();
  }());

  CalculatorGraph calculator_graph;
  MP_ASSERT_OK(calculator_graph.Initialize(std::move(config)));
  mediapipe::Packet output;
  MP_ASSERT_OK(calculator_graph.ObserveOutputStream(
      "out", [&](const mediapipe::Packet& p) {
        output = p;
        return absl::OkStatus();
      }));
  MP_ASSERT_OK(calculator_graph.StartRun({}));

  // Using tick of type SomeTick as in the graph builder.
  MP_ASSERT_OK(calculator_graph.AddPacketToInputStream(
      "tick", mediapipe::MakePacket<SomeTick>().At(Timestamp(1000))));
  MP_ASSERT_OK(calculator_graph.WaitUntilIdle());

  ASSERT_FALSE(output.IsEmpty());
  EXPECT_EQ(output.Get<int>(), 42);

  output = mediapipe::Packet();  // Reset packet.

  // Using a tick of a different type - should still work as Any is the
  // underlying implementation and `CalcualtorGraphConfig` doesn't preserve
  // builder's type restrictions.
  MP_ASSERT_OK(calculator_graph.AddPacketToInputStream(
      "tick", mediapipe::MakePacket<std::string>(
                  "not the same type as in graph builder")
                  .At(Timestamp(2000))));
  MP_ASSERT_OK(calculator_graph.WaitUntilIdle());

  ASSERT_FALSE(output.IsEmpty());
  EXPECT_EQ(output.Get<int>(), 42);
}

constexpr absl::string_view kPassThroughNode = "PassThroughNode";

struct PassThroughNode : Node<kPassThroughNode> {
  template <typename S>
  struct Contract {
    Repeated<Input<S, Any>> in{"IN"};
    Repeated<Output<S, Any>> out{"OUT"};

    Repeated<SideInput<S, Any>> side_in{"SIDE_IN"};
    Repeated<SideOutput<S, Any>> side_out{"SIDE_OUT"};
  };
};

// Common implementation for the calculator.
class PassThroughNodeImpl
    : public Calculator<PassThroughNode, PassThroughNodeImpl> {
 public:
  static absl::Status UpdateContract(CalculatorContract<PassThroughNode>& cc) {
    RET_CHECK_EQ(cc.in.Count(), cc.out.Count());
    for (int i = 0; i < cc.in.Count(); ++i) {
      cc.out.At(i).SetSameAs(cc.in.At(i));
    }

    RET_CHECK_EQ(cc.side_in.Count(), cc.side_out.Count());
    for (int i = 0; i < cc.side_in.Count(); ++i) {
      cc.side_out.At(i).SetSameAs(cc.side_in.At(i));
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext<PassThroughNode>& cc) final {
    for (int i = 0; i < cc.side_in.Count(); ++i) {
      cc.side_out.At(i).Set(cc.side_in.At(i).Packet());
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext<PassThroughNode>& cc) final {
    for (int i = 0; i < cc.in.Count(); ++i) {
      cc.out.At(i).Send(cc.in.At(i).Packet());
    }
    return absl::OkStatus();
  };
};

template <typename S>
struct PassThroughGraphContract {
  Input<S, int> in_a{"A"};
  Input<S, std::string> in_b{"B"};
  Output<S, int> out_a{"PASSED_A"};
  Output<S, std::string> out_b{"PASSED_B"};

  SideInput<S, int> in_side{"IN_SIDE"};
  SideOutput<S, int> out_side{"OUT_SIDE"};
};

TEST(AnyTest, CalculatorsCanSupportSameAsAny) {
  MP_ASSERT_OK_AND_ASSIGN(
      CalculatorGraphConfig config, ([]() {
        Graph<PassThroughGraphContract> graph;

        Stream<int> a = graph.in_a.Get().SetName("a");
        Stream<std::string> b = graph.in_b.Get().SetName("b");
        SidePacket<int> side = graph.in_side.Get().SetName("side");

        auto [passed_side, passed_a, passed_b] = [&]() {
          auto& node = graph.AddNode<PassThroughNode>();
          node.side_in.Add(side.Cast<Any>());
          node.in.Add(a.Cast<Any>());
          node.in.Add(b.Cast<Any>());
          return std::tuple(node.side_out.Add().Cast<int>(),
                            node.out.Add().Cast<int>(),
                            node.out.Add().Cast<std::string>());
        }();

        graph.out_side.Set(passed_side.SetName("passed_side"));
        graph.out_a.Set(passed_a.SetName("passed_a"));
        graph.out_b.Set(passed_b.SetName("passed_b"));

        return graph.GetConfig();
      }()));

  CalculatorGraph calculator_graph;
  MP_ASSERT_OK(calculator_graph.Initialize(std::move(config)));
  mediapipe::Packet output_a;
  MP_ASSERT_OK(calculator_graph.ObserveOutputStream(
      "passed_a", [&](const mediapipe::Packet& p) {
        output_a = p;
        return absl::OkStatus();
      }));
  mediapipe::Packet output_b;
  MP_ASSERT_OK(calculator_graph.ObserveOutputStream(
      "passed_b", [&](const mediapipe::Packet& p) {
        output_b = p;
        return absl::OkStatus();
      }));
  MP_ASSERT_OK(
      calculator_graph.StartRun({{"side", mediapipe::MakePacket<int>(256)}}));

  // Using input of the same type as in the graph builder.
  MP_ASSERT_OK(calculator_graph.AddPacketToInputStream(
      "a", mediapipe::MakePacket<int>(42).At(Timestamp(1000))));
  MP_ASSERT_OK(calculator_graph.AddPacketToInputStream(
      "b", mediapipe::MakePacket<std::string>("str").At(Timestamp(1000))));
  MP_ASSERT_OK(calculator_graph.WaitUntilIdle());

  MP_ASSERT_OK_AND_ASSIGN(mediapipe::Packet passed_side,
                          calculator_graph.GetOutputSidePacket("passed_side"));
  ASSERT_FALSE(passed_side.IsEmpty());
  EXPECT_EQ(passed_side.Get<int>(), 256);

  ASSERT_FALSE(output_a.IsEmpty());
  EXPECT_EQ(output_a.Get<int>(), 42);
  ASSERT_FALSE(output_b.IsEmpty());
  EXPECT_EQ(output_b.Get<std::string>(), "str");
}

}  // namespace
}  // namespace mediapipe::api3
