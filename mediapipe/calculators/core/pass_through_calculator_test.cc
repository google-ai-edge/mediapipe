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

#include "mediapipe/calculators/core/pass_through_calculator.h"

#include <string>
#include <tuple>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api3/any.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/tool/options_map.h"

namespace mediapipe::api3 {
namespace {

template <typename S>
struct PassThroughGraphContract {
  Input<S, int> in_a{"A"};
  Input<S, std::string> in_b{"B"};
  Output<S, int> out_a{"PASSED_A"};
  Output<S, std::string> out_b{"PASSED_B"};
};

TEST(AnyTest, CalculatorsCanSupportSameAsAny) {
  MP_ASSERT_OK_AND_ASSIGN(
      CalculatorGraphConfig config, ([]() {
        Graph<PassThroughGraphContract> graph;

        Stream<int> a = graph.in_a.Get().SetName("a");
        Stream<std::string> b = graph.in_b.Get().SetName("b");

        auto [passed_a, passed_b] = [&]() {
          auto& node = graph.AddNode<PassThroughNode>();
          node.in.Add(a.Cast<Any>());
          node.in.Add(b.Cast<Any>());
          return std::tuple(node.out.Add().Cast<int>(),
                            node.out.Add().Cast<std::string>());
        }();

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
  MP_ASSERT_OK(calculator_graph.StartRun({}));

  // Using input of the same type as in the graph builder.
  MP_ASSERT_OK(calculator_graph.AddPacketToInputStream(
      "a", mediapipe::MakePacket<int>(42).At(Timestamp(1000))));
  MP_ASSERT_OK(calculator_graph.AddPacketToInputStream(
      "b", mediapipe::MakePacket<std::string>("str").At(Timestamp(1000))));
  MP_ASSERT_OK(calculator_graph.WaitUntilIdle());

  ASSERT_FALSE(output_a.IsEmpty());
  EXPECT_EQ(output_a.Get<int>(), 42);
  ASSERT_FALSE(output_b.IsEmpty());
  EXPECT_EQ(output_b.Get<std::string>(), "str");
}

}  // namespace
}  // namespace mediapipe::api3
