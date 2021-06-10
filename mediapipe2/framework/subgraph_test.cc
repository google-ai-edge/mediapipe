// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/framework/subgraph.h"

#include <string>

#include "absl/strings/str_format.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/graph_service_manager.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

// Because of portability issues, we include this directly.
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"  // NOLINT(build/deprecated)

namespace mediapipe {
namespace {

class SubgraphTest : public ::testing::Test {
 protected:
  void TestGraphEnclosing(const std::string& subgraph_type_name) {
    EXPECT_TRUE(SubgraphRegistry::IsRegistered(subgraph_type_name));

    CalculatorGraphConfig config;
    config.add_input_stream("in");
    CalculatorGraphConfig::Node* node = config.add_node();
    node->set_calculator(subgraph_type_name);
    node->add_input_stream("INTS:in");
    node->add_output_stream("DUBS:dubs_tmp");
    node->add_output_stream("QUADS:quads");
    node = config.add_node();
    node->set_calculator("PassThroughCalculator");
    node->add_input_stream("dubs_tmp");
    node->add_output_stream("dubs");

    std::vector<Packet> dubs;
    tool::AddVectorSink("dubs", &config, &dubs);

    std::vector<Packet> quads;
    tool::AddVectorSink("quads", &config, &quads);

    CalculatorGraph graph;
    MP_ASSERT_OK(graph.Initialize(config));
    MP_ASSERT_OK(graph.StartRun({}));

    constexpr int kCount = 5;
    for (int i = 0; i < kCount; ++i) {
      MP_ASSERT_OK(graph.AddPacketToInputStream(
          "in", MakePacket<int>(i).At(Timestamp(i))));
    }

    MP_ASSERT_OK(graph.CloseInputStream("in"));
    MP_ASSERT_OK(graph.WaitUntilDone());

    EXPECT_EQ(dubs.size(), kCount);
    EXPECT_EQ(quads.size(), kCount);
    for (int i = 0; i < kCount; ++i) {
      EXPECT_EQ(i * 2, dubs[i].Get<int>());
      EXPECT_EQ(i * 4, quads[i].Get<int>());
    }
  }
};

// Tests registration of subgraph named "DubQuadTestSubgraph" using target
// "dub_quad_test_subgraph" from macro "mediapipe_simple_subgraph".
TEST_F(SubgraphTest, LinkedSubgraph) {
  TestGraphEnclosing("DubQuadTestSubgraph");
}

const mediapipe::GraphService<std::string> kStringTestService{
    "mediapipe::StringTestService"};
class EmitSideServiceStringTestSubgraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      mediapipe::SubgraphContext* sc) override {
    auto string_service = sc->Service(kStringTestService);
    RET_CHECK(string_service.IsAvailable()) << "Service not available";
    CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
            absl::StrFormat(R"(
          output_side_packet: "string"
          node {
            calculator: "ConstantSidePacketCalculator"
            output_side_packet: "PACKET:string"
            options: {
              [mediapipe.ConstantSidePacketCalculatorOptions.ext]: {
                packet { string_value: "%s" }
              }
            }
          }
        )",
                            string_service.GetObject()));
    return config;
  }
};
REGISTER_MEDIAPIPE_GRAPH(EmitSideServiceStringTestSubgraph);

TEST(SubgraphServicesTest, EmitStringFromTestService) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        output_side_packet: "str"
        node {
          calculator: "EmitSideServiceStringTestSubgraph"
          output_side_packet: "str"
        }
      )pb");

  Packet side_string;
  tool::AddSidePacketSink("str", &config, &side_string);

  CalculatorGraph graph;
  // It's important that service object is set before Initialize()
  MP_ASSERT_OK(graph.SetServiceObject(
      kStringTestService, std::make_shared<std::string>("Expected STRING")));
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  MP_ASSERT_OK(graph.WaitUntilDone());

  EXPECT_EQ(side_string.Get<std::string>(), "Expected STRING");
}

}  // namespace
}  // namespace mediapipe
