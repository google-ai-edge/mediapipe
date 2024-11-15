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

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/core/constant_side_packet_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/graph_service.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"

// Because of portability issues, we include this directly.
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"  // NOLINT(build/deprecated)
#include "mediapipe/framework/resources.h"

namespace mediapipe {
namespace {

using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::SidePacket;

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

TEST(SubgraphServicesTest, CanUseGraphServiceToReceiveString) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        output_side_packet: "str"
        node {
          calculator: "EmitSideServiceStringTestSubgraph"
          output_side_packet: "str"
        }
      )pb");

  CalculatorGraph graph;
  // It's important that service object is set before Initialize()
  MP_ASSERT_OK(graph.SetServiceObject(
      kStringTestService, std::make_shared<std::string>("Expected STRING")));
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilDone());

  MP_ASSERT_OK_AND_ASSIGN(Packet str_packet, graph.GetOutputSidePacket("str"));
  ASSERT_FALSE(str_packet.IsEmpty());
  EXPECT_EQ(str_packet.Get<std::string>(), "Expected STRING");
}

class EmitFileStringTestSubgraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      mediapipe::SubgraphContext* sc) override {
    MP_ASSIGN_OR_RETURN(
        std::unique_ptr<mediapipe::Resource> data,
        sc->GetResources().Get(
            "mediapipe/framework/testdata/resource_subgraph.data"));
    Graph graph;
    auto& node = graph.AddNode("ConstantSidePacketCalculator");
    node.GetOptions<mediapipe::ConstantSidePacketCalculatorOptions>()
        .add_packet()
        ->set_string_value(std::string(data->ToStringView()));
    SidePacket<std::string> side_string =
        node.SideOut("PACKET").Cast<std::string>();
    side_string.SetName("string") >> graph.SideOut(0);
    return graph.GetConfig();
  }
};
REGISTER_MEDIAPIPE_GRAPH(EmitFileStringTestSubgraph);

TEST(SubgraphServicesTest, CanLoadResourcesThroughSubgraphContext) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        output_side_packet: "str"
        node {
          calculator: "EmitFileStringTestSubgraph"
          output_side_packet: "str"
        }
      )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilDone());

  MP_ASSERT_OK_AND_ASSIGN(Packet str_packet, graph.GetOutputSidePacket("str"));
  ASSERT_FALSE(str_packet.IsEmpty());
  EXPECT_EQ(str_packet.Get<std::string>(), "File system subgraph contents\n");
}

class OptionsCheckingSubgraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      mediapipe::SubgraphContext* sc) override {
    std::string subgraph_side_packet_val;
    if (sc->HasOptions<ConstantSidePacketCalculatorOptions>()) {
      subgraph_side_packet_val =
          sc->Options<ConstantSidePacketCalculatorOptions>()
              .packet(0)
              .string_value();
    }
    Graph graph;
    auto& node = graph.AddNode("ConstantSidePacketCalculator");
    node.GetOptions<mediapipe::ConstantSidePacketCalculatorOptions>()
        .add_packet()
        ->set_string_value(subgraph_side_packet_val);
    SidePacket<std::string> side_string =
        node.SideOut("PACKET").Cast<std::string>();
    side_string.SetName("string") >> graph.SideOut(0);
    return graph.GetConfig();
  }
};
REGISTER_MEDIAPIPE_GRAPH(OptionsCheckingSubgraph);

TEST_F(SubgraphTest, CheckSubgraphOptionsPassedIn) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        output_side_packet: "str"
        node {
          calculator: "OptionsCheckingSubgraph"
          output_side_packet: "str"
          options: {
            [mediapipe.ConstantSidePacketCalculatorOptions.ext]: {
              packet { string_value: "test" }
            }
          }
        }
      )pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilDone());
  auto packet = graph.GetOutputSidePacket("str");
  MP_ASSERT_OK(packet);
  EXPECT_EQ(packet.value().Get<std::string>(), "test");
}

}  // namespace
}  // namespace mediapipe
