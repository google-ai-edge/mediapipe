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

#include "mediapipe/framework/api3/any.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/function_runner.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::api3 {
namespace {

TEST(PassThroughCalculatorTest, PassesPackets) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([](GenericGraph& graph, Stream<int> a, Stream<std::string> b)
                      -> std::tuple<Stream<int>, Stream<std::string>> {
        auto& node = graph.AddNode<PassThroughNode>();
        node.in.Add(a.Cast<Any>());
        node.in.Add(b.Cast<Any>());
        return std::tuple(node.out.Add().Cast<int>(),
                          node.out.Add().Cast<std::string>());
      }).Create());

  MP_ASSERT_OK_AND_ASSIGN(
      (auto [output_a, output_b]),
      runner.Run(MakePacket<int>(42), MakePacket<std::string>("str")));

  ASSERT_TRUE(output_a);
  EXPECT_EQ(output_a.GetOrDie(), 42);
  ASSERT_TRUE(output_b);
  EXPECT_EQ(output_b.GetOrDie(), "str");
}

}  // namespace
}  // namespace mediapipe::api3
