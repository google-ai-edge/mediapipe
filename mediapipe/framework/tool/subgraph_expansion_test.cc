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
#include "mediapipe/framework/tool/subgraph_expansion.h"

#include <memory>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/graph_service_manager.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/packet_set.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/proto_ns.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/status_handler.h"
#include "mediapipe/framework/subgraph.h"
#include "mediapipe/framework/tool/node_chain_subgraph.pb.h"

namespace mediapipe {

namespace {

using ::testing::HasSubstr;

class SimpleTestCalculator : public CalculatorBase {
 public:
  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }
  static absl::Status GetContract(CalculatorContract* cc) {
    for (PacketType& type : cc->Inputs()) {
      type.Set<int>();
    }
    for (PacketType& type : cc->Outputs()) {
      type.Set<int>();
    }
    for (PacketType& type : cc->InputSidePackets()) {
      type.Set<int>();
    }
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(SimpleTestCalculator);
typedef SimpleTestCalculator SomeSourceCalculator;
typedef SimpleTestCalculator SomeSinkCalculator;
typedef SimpleTestCalculator SomeRegularCalculator;
typedef SimpleTestCalculator SomeAggregator;
REGISTER_CALCULATOR(SomeSourceCalculator);
REGISTER_CALCULATOR(SomeSinkCalculator);
REGISTER_CALCULATOR(SomeRegularCalculator);
REGISTER_CALCULATOR(SomeAggregator);

class TestSubgraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      const SubgraphOptions& /*options*/) override {
    CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
          input_stream: "DATA:input_1"
          node {
            name: "regular_node"
            calculator: "SomeRegularCalculator"
            input_stream: "input_1"
            output_stream: "stream_a"
            input_side_packet: "side"
          }
          node {
            name: "simple_sink"
            calculator: "SomeSinkCalculator"
            input_stream: "stream_a"
          }
          packet_generator {
            packet_generator: "SomePacketGenerator"
            output_side_packet: "side"
          }
        )pb");
    return config;
  }
};
REGISTER_MEDIAPIPE_GRAPH(TestSubgraph);

class PacketFactoryTestSubgraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      const SubgraphOptions& /*options*/) override {
    CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
          input_stream: "DATA:input_1"
          node {
            name: "regular_node"
            calculator: "SomeRegularCalculator"
            input_stream: "input_1"
            output_stream: "stream_a"
            input_side_packet: "side"
          }
          node {
            name: "simple_sink"
            calculator: "SomeSinkCalculator"
            input_stream: "stream_a"
          }
          packet_factory {
            packet_factory: "SomePacketFactory"
            output_side_packet: "side"
          }
        )pb");
    return config;
  }
};
REGISTER_MEDIAPIPE_GRAPH(PacketFactoryTestSubgraph);

// This subgraph chains copies of the specified node in series. The node type
// and the number of copies of the node are specified in subgraph options.
class NodeChainSubgraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      const SubgraphOptions& options) override {
    auto opts =
        Subgraph::GetOptions<mediapipe::NodeChainSubgraphOptions>(options);
    const ProtoString& node_type = opts.node_type();
    int chain_length = opts.chain_length();
    RET_CHECK(!node_type.empty());
    RET_CHECK_GT(chain_length, 0);
    CalculatorGraphConfig config;
    config.add_input_stream("INPUT:stream_0");
    config.add_output_stream(absl::StrCat("OUTPUT:stream_", chain_length));
    for (int i = 0; i < chain_length; ++i) {
      CalculatorGraphConfig::Node* node = config.add_node();
      node->set_calculator(node_type);
      node->add_input_stream(absl::StrCat("stream_", i));
      node->add_output_stream(absl::StrCat("stream_", i + 1));
    }
    return config;
  }
};
REGISTER_MEDIAPIPE_GRAPH(NodeChainSubgraph);

// A subgraph used in the ExecutorFieldOfNodeInSubgraphPreserved test. The
// subgraph contains a node with the executor field "custom_thread_pool".
class NodeWithExecutorSubgraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      const SubgraphOptions& options) override {
    CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
          input_stream: "INPUT:foo"
          output_stream: "OUTPUT:bar"
          node {
            calculator: "PassThroughCalculator"
            input_stream: "foo"
            output_stream: "bar"
            executor: "custom_thread_pool"
          }
        )pb");
    return config;
  }
};
REGISTER_MEDIAPIPE_GRAPH(NodeWithExecutorSubgraph);

// A subgraph used in the ExecutorFieldOfNodeInSubgraphPreserved test. The
// subgraph contains a NodeWithExecutorSubgraph.
class EnclosingSubgraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      const SubgraphOptions& options) override {
    CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
          input_stream: "IN:in"
          output_stream: "OUT:out"
          node {
            calculator: "NodeWithExecutorSubgraph"
            input_stream: "INPUT:in"
            output_stream: "OUTPUT:out"
          }
        )pb");
    return config;
  }
};
REGISTER_MEDIAPIPE_GRAPH(EnclosingSubgraph);

TEST(SubgraphExpansionTest, TransformStreamNames) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SomeSinkCalculator"
          input_stream: "input_1"
          input_stream: "VIDEO:input_2"
          input_stream: "AUDIO:0:input_3"
          input_stream: "AUDIO:1:input_4"
        }
      )pb");
  CalculatorGraphConfig expected_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SomeSinkCalculator"
          input_stream: "input_1_foo"
          input_stream: "VIDEO:input_2_foo"
          input_stream: "AUDIO:0:input_3_foo"
          input_stream: "AUDIO:1:input_4_foo"
        }
      )pb");
  auto add_foo = [](absl::string_view s) { return absl::StrCat(s, "_foo"); };
  MP_EXPECT_OK(tool::TransformStreamNames(
      (*config.mutable_node())[0].mutable_input_stream(), add_foo));
  EXPECT_THAT(config, mediapipe::EqualsProto(expected_config));
}

TEST(SubgraphExpansionTest, TransformNames) {
  CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input_1"
        node {
          calculator: "SomeRegularCalculator"
          name: "bob"
          input_stream: "input_1"
          input_stream: "VIDEO:input_2"
          input_stream: "AUDIO:0:input_3"
          input_stream: "AUDIO:1:input_4"
          output_stream: "output_1"
        }
        node {
          calculator: "SomeRegularCalculator"
          input_stream: "output_1"
          output_stream: "output_2"
        }
      )pb");
  CalculatorGraphConfig expected_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "__sg0_input_1"
        node {
          calculator: "SomeRegularCalculator"
          name: "__sg0_bob"
          input_stream: "__sg0_input_1"
          input_stream: "VIDEO:__sg0_input_2"
          input_stream: "AUDIO:0:__sg0_input_3"
          input_stream: "AUDIO:1:__sg0_input_4"
          output_stream: "__sg0_output_1"
        }
        node {
          name: "__sg0_SomeRegularCalculator"
          calculator: "SomeRegularCalculator"
          input_stream: "__sg0_output_1"
          output_stream: "__sg0_output_2"
        }
      )pb");
  auto add_prefix = [](absl::string_view s) {
    return absl::StrCat("__sg0_", s);
  };
  MP_EXPECT_OK(tool::TransformNames(&config, add_prefix));
  EXPECT_THAT(config, mediapipe::EqualsProto(expected_config));
}

TEST(SubgraphExpansionTest, FindCorrespondingStreams) {
  CalculatorGraphConfig config1 =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input_1"
        input_stream: "VIDEO:input_2"
        input_stream: "AUDIO:0:input_3"
        input_stream: "AUDIO:1:input_4"
      )pb");
  CalculatorGraphConfig config2 =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SomeSubgraph"
          input_stream: "foo"
          input_stream: "VIDEO:bar"
          input_stream: "AUDIO:0:baz"
          input_stream: "AUDIO:1:qux"
        }
      )pb");
  std::map<std::string, std::string> stream_map;
  MP_EXPECT_OK(tool::FindCorrespondingStreams(
      &stream_map, config1.input_stream(), config2.node()[0].input_stream()));
  EXPECT_THAT(stream_map,
              testing::UnorderedElementsAre(testing::Pair("input_1", "foo"),
                                            testing::Pair("input_2", "bar"),
                                            testing::Pair("input_3", "baz"),
                                            testing::Pair("input_4", "qux")));
}

TEST(SubgraphExpansionTest, FindCorrespondingStreamsNonexistentTag) {
  // The VIDEO tag does not exist in the subgraph.
  CalculatorGraphConfig config1 =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input_1"
        input_stream: "AUDIO:0:input_3"
        input_stream: "AUDIO:1:input_4"
      )pb");
  CalculatorGraphConfig config2 =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SomeSubgraph"
          input_stream: "foo"
          input_stream: "VIDEO:bar"
          input_stream: "AUDIO:0:baz"
          input_stream: "AUDIO:1:qux"
        }
      )pb");
  std::map<std::string, std::string> stream_map;
  auto status = tool::FindCorrespondingStreams(
      &stream_map, config1.input_stream(), config2.node()[0].input_stream());
  EXPECT_THAT(status.message(),

              testing::AllOf(
                  // Problematic tag.
                  testing::HasSubstr("VIDEO"),
                  // Error.
                  testing::HasSubstr("does not exist")));
}

TEST(SubgraphExpansionTest, FindCorrespondingStreamsTooFewIndexes) {
  // The AUDIO tag has too few indexes in the subgraph (1 vs. 2).
  CalculatorGraphConfig config1 =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input_1"
        input_stream: "VIDEO:input_2"
        input_stream: "AUDIO:0:input_3"
      )pb");
  CalculatorGraphConfig config2 =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SomeSubgraph"
          input_stream: "foo"
          input_stream: "VIDEO:bar"
          input_stream: "AUDIO:0:baz"
          input_stream: "AUDIO:1:qux"
        }
      )pb");
  std::map<std::string, std::string> stream_map;
  auto status = tool::FindCorrespondingStreams(
      &stream_map, config1.input_stream(), config2.node()[0].input_stream());

  EXPECT_THAT(status.message(),
              testing::AllOf(
                  // Problematic tag.
                  testing::HasSubstr("AUDIO"),
                  // Error.
                  testing::HasSubstr(" 2 "), testing::HasSubstr(" 1 ")));
}

TEST(SubgraphExpansionTest, ConnectSubgraphStreams) {
  CalculatorGraphConfig subgraph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "A:input_1"
        input_stream: "B:input_2"
        output_stream: "O:output_2"
        input_side_packet: "SI:side_input"
        output_side_packet: "SO:side_output"
        node {
          calculator: "SomeRegularCalculator"
          input_stream: "input_1"
          input_stream: "VIDEO:input_2"
          input_side_packet: "side_input"
          output_stream: "output_1"
        }
        node {
          calculator: "SomeRegularCalculator"
          input_stream: "input_1"
          input_stream: "output_1"
          output_stream: "output_2"
        }
        packet_generator {
          packet_generator: "SomeGenerator"
          input_side_packet: "side_input"
          output_side_packet: "side_output"
        }
      )pb");
  CalculatorGraphConfig supergraph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "SomeSubgraph"
          input_stream: "A:foo"
          input_stream: "B:bar"
          output_stream: "O:foobar"
          input_side_packet: "SI:flip"
          output_side_packet: "SO:flop"
        }
      )pb");
  // Note: graph input streams, output streams, and side packets on the
  // subgraph are not changed because they are going to be discarded anyway.
  CalculatorGraphConfig expected_subgraph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "A:input_1"
        input_stream: "B:input_2"
        output_stream: "O:output_2"
        input_side_packet: "SI:side_input"
        output_side_packet: "SO:side_output"
        node {
          calculator: "SomeRegularCalculator"
          input_stream: "foo"
          input_stream: "VIDEO:bar"
          input_side_packet: "flip"
          output_stream: "output_1"
        }
        node {
          calculator: "SomeRegularCalculator"
          input_stream: "foo"
          input_stream: "output_1"
          output_stream: "foobar"
        }
        packet_generator {
          packet_generator: "SomeGenerator"
          input_side_packet: "flip"
          output_side_packet: "flop"
        }
      )pb");
  MP_EXPECT_OK(tool::ConnectSubgraphStreams(supergraph.node()[0], &subgraph));
  EXPECT_THAT(subgraph, mediapipe::EqualsProto(expected_subgraph));
}

TEST(SubgraphExpansionTest, ExpandSubgraphs) {
  CalculatorGraphConfig supergraph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          name: "simple_source"
          calculator: "SomeSourceCalculator"
          output_stream: "foo"
        }
        node { calculator: "TestSubgraph" input_stream: "DATA:foo" }
      )pb");
  CalculatorGraphConfig expected_graph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          name: "simple_source"
          calculator: "SomeSourceCalculator"
          output_stream: "foo"
        }
        node {
          name: "testsubgraph__regular_node"
          calculator: "SomeRegularCalculator"
          input_stream: "foo"
          output_stream: "testsubgraph__stream_a"
          input_side_packet: "testsubgraph__side"
        }
        node {
          name: "testsubgraph__simple_sink"
          calculator: "SomeSinkCalculator"
          input_stream: "testsubgraph__stream_a"
        }
        packet_generator {
          packet_generator: "SomePacketGenerator"
          output_side_packet: "testsubgraph__side"
        }
      )pb");
  MP_EXPECT_OK(tool::ExpandSubgraphs(&supergraph));
  EXPECT_THAT(supergraph, mediapipe::EqualsProto(expected_graph));
}

TEST(SubgraphExpansionTest, ValidateSubgraphFields) {
  CalculatorGraphConfig supergraph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          name: "simple_source"
          calculator: "SomeSourceCalculator"
          output_stream: "foo"
        }
        node {
          name: "foo_subgraph"
          calculator: "TestSubgraph"
          input_stream: "DATA:foo"
          buffer_size_hint: -1  # This field is only applicable to calculators.
        }
      )pb");
  absl::Status s1 = tool::ValidateSubgraphFields(supergraph.node(1));
  EXPECT_EQ(s1.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(s1.message(), testing::HasSubstr("foo_subgraph"));

  absl::Status s2 = tool::ExpandSubgraphs(&supergraph);
  EXPECT_EQ(s2.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(s2.message(), testing::HasSubstr("foo_subgraph"));
}

// A test that captures the use case of CL 191001940. The "executor" field of
// a node inside a subgraph should be preserved, not mapped or mangled. This
// test will help us detect breakage of this use case when we implement
// subgraph executor support in the future.
TEST(SubgraphExpansionTest, ExecutorFieldOfNodeInSubgraphPreserved) {
  CalculatorGraphConfig supergraph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input"
        executor {
          name: "custom_thread_pool"
          type: "ThreadPoolExecutor"
          options {
            [mediapipe.ThreadPoolExecutorOptions.ext] { num_threads: 4 }
          }
        }
        node {
          calculator: "EnclosingSubgraph"
          input_stream: "IN:input"
          output_stream: "OUT:output"
        }
      )pb");
  CalculatorGraphConfig expected_graph = mediapipe::ParseTextProtoOrDie<
      CalculatorGraphConfig>(R"pb(
    input_stream: "input"
    executor {
      name: "custom_thread_pool"
      type: "ThreadPoolExecutor"
      options {
        [mediapipe.ThreadPoolExecutorOptions.ext] { num_threads: 4 }
      }
    }
    node {
      calculator: "PassThroughCalculator"
      name: "enclosingsubgraph__nodewithexecutorsubgraph__PassThroughCalculator"
      input_stream: "input"
      output_stream: "output"
      executor: "custom_thread_pool"
    }
  )pb");
  MP_EXPECT_OK(tool::ExpandSubgraphs(&supergraph));
  EXPECT_THAT(supergraph, mediapipe::EqualsProto(expected_graph));
}

const mediapipe::GraphService<std::string> kStringTestService{
    "mediapipe::StringTestService"};
class GraphServicesClientTestSubgraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    auto string_service = sc->Service(kStringTestService);
    RET_CHECK(string_service.IsAvailable()) << "Service not available";
    CalculatorGraphConfig config;
    config.add_node()->set_calculator(string_service.GetObject());
    return config;
  }
};
REGISTER_MEDIAPIPE_GRAPH(GraphServicesClientTestSubgraph);

TEST(SubgraphExpansionTest, GraphServicesUsage) {
  CalculatorGraphConfig supergraph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node { calculator: "GraphServicesClientTestSubgraph" }
      )pb");

  CalculatorGraphConfig expected_graph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          name: "graphservicesclienttestsubgraph__ExpectedNode"
          calculator: "ExpectedNode"
        }
      )pb");
  GraphServiceManager service_manager;
  MP_ASSERT_OK(service_manager.SetServiceObject(
      kStringTestService, std::make_shared<std::string>("ExpectedNode")));
  MP_EXPECT_OK(tool::ExpandSubgraphs(&supergraph, /*graph_registry=*/nullptr,
                                     /*graph_options=*/nullptr,
                                     &service_manager));
  EXPECT_THAT(supergraph, mediapipe::EqualsProto(expected_graph));
}

// Shows SubgraphOptions consumed by GraphRegistry::CreateByName.
TEST(SubgraphExpansionTest, SubgraphOptionsUsage) {
  EXPECT_TRUE(SubgraphRegistry::IsRegistered("NodeChainSubgraph"));
  GraphRegistry graph_registry;

  // CalculatorGraph::Initialize passes the SubgraphOptions into:
  // (1) GraphRegistry::CreateByName("NodeChainSubgraph", options)
  // (2) tool::ExpandSubgraphs(&config, options)
  auto graph_options =
      mediapipe::ParseTextProtoOrDie<Subgraph::SubgraphOptions>(R"pb(
        options {
          [mediapipe.NodeChainSubgraphOptions.ext] {
            node_type: "DoubleIntCalculator"
            chain_length: 3
          }
        })pb");
  SubgraphContext context(&graph_options, /*service_manager=*/nullptr);

  // "NodeChainSubgraph" consumes graph_options only in CreateByName.
  auto subgraph_status =
      graph_registry.CreateByName("", "NodeChainSubgraph", &context);
  MP_ASSERT_OK(subgraph_status);
  auto subgraph = std::move(subgraph_status).value();
  MP_ASSERT_OK(
      tool::ExpandSubgraphs(&subgraph, &graph_registry, &graph_options));

  CalculatorGraphConfig expected_graph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "DoubleIntCalculator"
          input_stream: "stream_0"
          output_stream: "stream_1"
        }
        node {
          calculator: "DoubleIntCalculator"
          input_stream: "stream_1"
          output_stream: "stream_2"
        }
        node {
          calculator: "DoubleIntCalculator"
          input_stream: "stream_2"
          output_stream: "stream_3"
        }
        input_stream: "INPUT:stream_0"
        output_stream: "OUTPUT:stream_3"
      )pb");

  EXPECT_THAT(subgraph, mediapipe::EqualsProto(expected_graph));
}

// Shows SubgraphOptions consumed by tool::ExpandSubgraphs.
TEST(SubgraphExpansionTest, SimpleSubgraphOptionsUsage) {
  EXPECT_TRUE(SubgraphRegistry::IsRegistered("NodeChainSubgraph"));
  GraphRegistry graph_registry;
  auto moon_options =
      mediapipe::ParseTextProtoOrDie<Subgraph::SubgraphOptions>(R"pb(
        options {
          [mediapipe.NodeChainSubgraphOptions.ext] {
            node_type: "DoubleIntCalculator"
            chain_length: 3
          }
        })pb");
  auto moon_subgraph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        type: "MoonSubgraph"
        graph_options: {
          [type.googleapis.com/mediapipe.NodeChainSubgraphOptions] {}
        }
        node: {
          calculator: "MoonCalculator"
          node_options: {
            [type.googleapis.com/mediapipe.NodeChainSubgraphOptions] {}
          }
          option_value: "chain_length:options/chain_length"
        }
      )pb");

  // The moon_options are copied into the graph_options of moon_subgraph.
  MP_ASSERT_OK(
      tool::ExpandSubgraphs(&moon_subgraph, &graph_registry, &moon_options));

  // The field chain_length is copied from moon_options into MoonCalculator.
  CalculatorGraphConfig expected_graph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "MoonCalculator"
          node_options {
            [type.googleapis.com/mediapipe.NodeChainSubgraphOptions] {
              chain_length: 3
            }
          }
        }
        type: "MoonSubgraph"
        graph_options {
          [type.googleapis.com/mediapipe.NodeChainSubgraphOptions] {}
        }
      )pb");
  EXPECT_THAT(moon_subgraph, mediapipe::EqualsProto(expected_graph));
}

// Shows ExpandSubgraphs applied twice. "option_value" fields are evaluated
// and removed on the first ExpandSubgraphs call.  If "option_value" fields
// are not removed during ExpandSubgraphs, they evaluate incorrectly on the
// second ExpandSubgraphs call and this test fails on "expected_node_options".
TEST(SubgraphExpansionTest, SimpleSubgraphOptionsTwice) {
  GraphRegistry graph_registry;

  // Register a simple-subgraph that accepts graph options.
  auto moon_subgraph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        type: "MoonSubgraph"
        graph_options: {
          [type.googleapis.com/mediapipe.NodeChainSubgraphOptions] {}
        }
        node: {
          calculator: "MoonCalculator"
          node_options: {
            [type.googleapis.com/mediapipe.NodeChainSubgraphOptions] {}
          }
          option_value: "chain_length:options/chain_length"
        }
      )pb");
  graph_registry.Register("MoonSubgraph", moon_subgraph);

  // Invoke the simple-subgraph with graph options.
  // The empty NodeChainSubgraphOptions below allows "option_value" fields
  // on "MoonCalculator" to evaluate incorrectly, if not removed.
  auto sky_graph = mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    graph_options: {
      [type.googleapis.com/mediapipe.NodeChainSubgraphOptions] {}
    }
    node: {
      calculator: "MoonSubgraph"
      options: {
        [mediapipe.NodeChainSubgraphOptions.ext] {
          node_type: "DoubleIntCalculator"
          chain_length: 3
        }
      }
    }
  )pb");

  // The first ExpandSubgraphs call evaluates and removes "option_value" fields.
  MP_ASSERT_OK(tool::ExpandSubgraphs(&sky_graph, &graph_registry));
  auto expanded_1 = sky_graph;

  // The second ExpandSubgraphs call has no effect on the expanded graph.
  MP_ASSERT_OK(tool::ExpandSubgraphs(&sky_graph, &graph_registry));

  // Validate the expected node_options for the "MoonSubgraph".
  // If the "option_value" fields are not removed during ExpandSubgraphs,
  // this test fails with an incorrect value for "chain_length".
  auto expected_node_options =
      mediapipe::ParseTextProtoOrDie<mediapipe::NodeChainSubgraphOptions>(
          "chain_length: 3");
  mediapipe::NodeChainSubgraphOptions node_options;
  sky_graph.node(0).node_options(0).UnpackTo(&node_options);
  ASSERT_THAT(node_options, mediapipe::EqualsProto(expected_node_options));

  // Validate the results from both ExpandSubgraphs() calls.
  CalculatorGraphConfig expected_graph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        graph_options {
          [type.googleapis.com/mediapipe.NodeChainSubgraphOptions] {}
        }
        node {
          name: "moonsubgraph__MoonCalculator"
          calculator: "MoonCalculator"
          node_options {
            [type.googleapis.com/mediapipe.NodeChainSubgraphOptions] {
              chain_length: 3
            }
          }
        }
      )pb");
  EXPECT_THAT(expanded_1, mediapipe::EqualsProto(expected_graph));
  EXPECT_THAT(sky_graph, mediapipe::EqualsProto(expected_graph));
}

// A subgraph that defines and uses an internal executor with name "xyz".
class InternalExecutorSubgraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      const SubgraphOptions& options) override {
    return mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
      input_stream: "IN:foo"
      output_stream: "OUT:bar"
      executor {
        name: "xyz"
        type: "ThreadPoolExecutor"
        options {
          [mediapipe.ThreadPoolExecutorOptions.ext] { num_threads: 1 }
        }
      }
      node {
        calculator: "PassThroughCalculator"
        executor: "xyz"
        input_stream: "foo"
        output_stream: "bar"
      }
    )pb");
  }
};
REGISTER_MEDIAPIPE_GRAPH(InternalExecutorSubgraph);

// This test confirms that none of existing subgraphs can actually create an
// executor when used as subgraphs and not like a final graph.
TEST(SubgraphExpansionTest, SubgraphExecutorIsIgnored) {
  CalculatorGraphConfig supergraph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input"
        node {
          calculator: "InternalExecutorSubgraph"
          input_stream: "IN:input"
          output_stream: "OUT:output"
        }
      )pb");
  CalculatorGraphConfig expected_graph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input"
        node {
          name: "internalexecutorsubgraph__PassThroughCalculator"
          calculator: "PassThroughCalculator"
          input_stream: "input"
          output_stream: "output"
          executor: "xyz"
        }
      )pb");
  MP_EXPECT_OK(tool::ExpandSubgraphs(&supergraph));
  EXPECT_THAT(supergraph, mediapipe::EqualsProto(expected_graph));

  CalculatorGraph calculator_graph;
  EXPECT_THAT(calculator_graph.Initialize(supergraph),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("The executor \"xyz\" is "
                                 "not declared in an ExecutorConfig.")));
}

class NestedInternalExecutorsSubgraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      const SubgraphOptions& options) override {
    return mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
      input_stream: "IN:foo"
      output_stream: "OUT:bar"
      node {
        calculator: "InternalExecutorSubgraph"
        input_stream: "IN:foo"
        output_stream: "OUT:bar_0"
      }
      executor {
        name: "xyz"
        type: "ThreadPoolExecutor"
        options {
          [mediapipe.ThreadPoolExecutorOptions.ext] { num_threads: 1 }
        }
      }
      node {
        calculator: "PassThroughCalculator"
        executor: "xyz"
        input_stream: "bar_0"
        output_stream: "bar_1"
      }
      executor {
        name: "abc"
        type: "ThreadPoolExecutor"
        options {
          [mediapipe.ThreadPoolExecutorOptions.ext] { num_threads: 1 }
        }
      }
      node {
        calculator: "PassThroughCalculator"
        executor: "abc"
        input_stream: "bar_1"
        output_stream: "bar"
      }
    )pb");
  }
};
REGISTER_MEDIAPIPE_GRAPH(NestedInternalExecutorsSubgraph);

TEST(SubgraphExpansionTest, NestedSubgraphExecutorsAreIgnored) {
  CalculatorGraphConfig supergraph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input"
        node {
          calculator: "NestedInternalExecutorsSubgraph"
          input_stream: "IN:input"
          output_stream: "OUT:output"
        }
      )pb");
  CalculatorGraphConfig expected_graph = mediapipe::ParseTextProtoOrDie<
      CalculatorGraphConfig>(R"pb(
    node {
      name: "nestedinternalexecutorssubgraph__PassThroughCalculator_1"
      calculator: "PassThroughCalculator"
      input_stream: "nestedinternalexecutorssubgraph__bar_0"
      output_stream: "nestedinternalexecutorssubgraph__bar_1"
      executor: "xyz"
    }
    node {
      name: "nestedinternalexecutorssubgraph__PassThroughCalculator_2"
      calculator: "PassThroughCalculator"
      input_stream: "nestedinternalexecutorssubgraph__bar_1"
      output_stream: "output"
      executor: "abc"
    }
    node {
      name: "nestedinternalexecutorssubgraph__internalexecutorsubgraph__PassThroughCalculator"
      calculator: "PassThroughCalculator"
      input_stream: "input"
      output_stream: "nestedinternalexecutorssubgraph__bar_0"
      executor: "xyz"
    }
    input_stream: "input"
  )pb");
  MP_EXPECT_OK(tool::ExpandSubgraphs(&supergraph));
  EXPECT_THAT(supergraph, mediapipe::EqualsProto(expected_graph));

  CalculatorGraph calculator_graph;
  EXPECT_THAT(calculator_graph.Initialize(supergraph),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("The executor \"xyz\" is "
                                 "not declared in an ExecutorConfig.")));
}

TEST(SubgraphExpansionTest, GraphExecutorsSubstituteSubgraphExecutors) {
  CalculatorGraphConfig supergraph =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "input"
        executor {
          name: "xyz"
          type: "ThreadPoolExecutor"
          options {
            [mediapipe.ThreadPoolExecutorOptions.ext] { num_threads: 1 }
          }
        }
        executor {
          name: "abc"
          type: "ThreadPoolExecutor"
          options {
            [mediapipe.ThreadPoolExecutorOptions.ext] { num_threads: 1 }
          }
        }
        node {
          calculator: "NestedInternalExecutorsSubgraph"
          input_stream: "IN:input"
          output_stream: "OUT:output"
        }
      )pb");
  CalculatorGraphConfig expected_graph = mediapipe::ParseTextProtoOrDie<
      CalculatorGraphConfig>(R"pb(
    node {
      name: "nestedinternalexecutorssubgraph__PassThroughCalculator_1"
      calculator: "PassThroughCalculator"
      input_stream: "nestedinternalexecutorssubgraph__bar_0"
      output_stream: "nestedinternalexecutorssubgraph__bar_1"
      executor: "xyz"
    }
    node {
      name: "nestedinternalexecutorssubgraph__PassThroughCalculator_2"
      calculator: "PassThroughCalculator"
      input_stream: "nestedinternalexecutorssubgraph__bar_1"
      output_stream: "output"
      executor: "abc"
    }
    node {
      name: "nestedinternalexecutorssubgraph__internalexecutorsubgraph__PassThroughCalculator"
      calculator: "PassThroughCalculator"
      input_stream: "input"
      output_stream: "nestedinternalexecutorssubgraph__bar_0"
      executor: "xyz"
    }
    input_stream: "input"
    executor {
      name: "xyz"
      type: "ThreadPoolExecutor"
      options {
        [mediapipe.ThreadPoolExecutorOptions.ext] { num_threads: 1 }
      }
    }
    executor {
      name: "abc"
      type: "ThreadPoolExecutor"
      options {
        [mediapipe.ThreadPoolExecutorOptions.ext] { num_threads: 1 }
      }
    }
  )pb");
  MP_EXPECT_OK(tool::ExpandSubgraphs(&supergraph));
  EXPECT_THAT(supergraph, mediapipe::EqualsProto(expected_graph));

  CalculatorGraph calculator_graph;
  MP_EXPECT_OK(calculator_graph.Initialize(supergraph));
}

}  // namespace
}  // namespace mediapipe
