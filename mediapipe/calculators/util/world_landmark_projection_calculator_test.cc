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

#include "mediapipe/calculators/util/world_landmark_projection_calculator.h"

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/packet_test_util.h"

namespace mediapipe::api3 {
namespace {

struct WorldLandmarkProjectionTestCase {
  std::string test_name;
  LandmarkList input_landmarks;
  LandmarkList expected_output_landmarks;
};

using WorldLandmarkProjectionTest =
    ::testing::TestWithParam<WorldLandmarkProjectionTestCase>;

TEST_P(WorldLandmarkProjectionTest, Passes) {
  MP_ASSERT_OK_AND_ASSIGN(CalculatorGraphConfig config, []() {
    Graph<WorldLandmarkProjectionNode::Contract> graph;

    Stream<LandmarkList> in_landmarks =
        graph.input_landmarks.Get().SetName("in_landmarks");

    Stream<LandmarkList> out_landmarks = [&]() {
      auto& node = graph.AddNode<WorldLandmarkProjectionNode>();
      node.input_landmarks.Set(in_landmarks);
      return node.output_landmarks.Get();
    }();

    graph.output_landmarks.Set(out_landmarks.SetName("out_landmarks"));

    return graph.GetConfig();
  }());

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(std::move(config)));
  mediapipe::Packet out_packet;
  MP_ASSERT_OK(graph.ObserveOutputStream("out_landmarks", [&](const Packet& p) {
    out_packet = p;
    return absl::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "in_landmarks",
      MakePacket<LandmarkList>(GetParam().input_landmarks).At(Timestamp(0))));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  EXPECT_THAT(out_packet, mediapipe::PacketContains<LandmarkList>(EqualsProto(
                              GetParam().expected_output_landmarks)));

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

LandmarkList GetSingleLandmark() {
  LandmarkList list;
  auto& lm = *list.add_landmark();
  lm.set_x(1.0f);
  lm.set_y(2.0f);
  lm.set_z(3.0f);
  lm.set_presence(0.5f);
  lm.set_visibility(1.5f);
  return list;
}
LandmarkList GetMultiLandmarks() {
  LandmarkList list;
  for (int j = 0; j < 10; ++j) {
    auto& lm = *list.add_landmark();
    lm.set_x(1.0f + j);
    lm.set_y(2.0f + j);
    lm.set_z(3.0f + j);
    lm.set_presence(0.5f + j);
    lm.set_visibility(1.5f + j);
  }
  return list;
}

INSTANTIATE_TEST_SUITE_P(
    WorldLandmarkProjectionTestInstantiation, WorldLandmarkProjectionTest,
    testing::ValuesIn<WorldLandmarkProjectionTestCase>(
        {{.test_name = "EmptyInputEmptyOutput",
          .input_landmarks = LandmarkList(),
          .expected_output_landmarks = LandmarkList()},
         {.test_name = "SingleLandmarkSameOutput",
          .input_landmarks = GetSingleLandmark(),
          .expected_output_landmarks = GetSingleLandmark()},
         {.test_name = "MultiLandmarksSameOutput",
          .input_landmarks = GetMultiLandmarks(),
          .expected_output_landmarks = GetMultiLandmarks()}}),
    [](const testing::TestParamInfo<WorldLandmarkProjectionTest::ParamType>&
           info) { return info.param.test_name; });

}  // namespace
}  // namespace mediapipe::api3
