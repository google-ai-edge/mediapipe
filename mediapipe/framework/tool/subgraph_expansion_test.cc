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
#include "mediapipe/framework/deps/message_matchers.h"
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

class SimpleTestCalculator : public CalculatorBase {
 public:
  ::mediapipe::Status Process(CalculatorContext* cc) override {
    return ::mediapipe::OkStatus();
  }
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    for (PacketType& type : cc->Inputs()) {
      type.Set<int>();
    }
    for (PacketType& type : cc->Outputs()) {
      type.Set<int>();
    }
    for (PacketType& type : cc->InputSidePackets()) {
      type.Set<int>();
    }
    return ::mediapipe::OkStatus();
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
  ::mediapipe::StatusOr<CalculatorGraphConfig> GetConfig(
      const SubgraphOptions& /*options*/) override {
    CalculatorGraphConfig config =
        ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
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
        )");
    return config;
  }
};
REGISTER_MEDIAPIPE_GRAPH(TestSubgraph);

class PacketFactoryTestSubgraph : public Subgraph {
 public:
  ::mediapipe::StatusOr<CalculatorGraphConfig> GetConfig(
      const SubgraphOptions& /*options*/) override {
    CalculatorGraphConfig config =
        ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
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
        )");
    return config;
  }
};
REGISTER_MEDIAPIPE_GRAPH(PacketFactoryTestSubgraph);

// This subgraph chains copies of the specified node in series. The node type
// and the number of copies of the node are specified in subgraph options.
class NodeChainSubgraph : public Subgraph {
 public:
  ::mediapipe::StatusOr<CalculatorGraphConfig> GetConfig(
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
  ::mediapipe::StatusOr<CalculatorGraphConfig> GetConfig(
      const SubgraphOptions& options) override {
    CalculatorGraphConfig config =
        ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
          input_stream: "INPUT:foo"
          output_stream: "OUTPUT:bar"
          node {
            calculator: "PassThroughCalculator"
            input_stream: "foo"
            output_stream: "bar"
            executor: "custom_thread_pool"
          }
        )");
    return config;
  }
};
REGISTER_MEDIAPIPE_GRAPH(NodeWithExecutorSubgraph);

// A subgraph used in the ExecutorFieldOfNodeInSubgraphPreserved test. The
// subgraph contains a NodeWithExecutorSubgraph.
class EnclosingSubgraph : public Subgraph {
 public:
  ::mediapipe::StatusOr<CalculatorGraphConfig> GetConfig(
      const SubgraphOptions& options) override {
    CalculatorGraphConfig config =
        ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
          input_stream: "IN:in"
          output_stream: "OUT:out"
          node {
            calculator: "NodeWithExecutorSubgraph"
            input_stream: "INPUT:in"
            output_stream: "OUTPUT:out"
          }
        )");
    return config;
  }
};
REGISTER_MEDIAPIPE_GRAPH(EnclosingSubgraph);

TEST(SubgraphExpansionTest, TransformStreamNames) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: "SomeSinkCalculator"
          input_stream: "input_1"
          input_stream: "VIDEO:input_2"
          input_stream: "AUDIO:0:input_3"
          input_stream: "AUDIO:1:input_4"
        }
      )");
  CalculatorGraphConfig expected_config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: "SomeSinkCalculator"
          input_stream: "input_1_foo"
          input_stream: "VIDEO:input_2_foo"
          input_stream: "AUDIO:0:input_3_foo"
          input_stream: "AUDIO:1:input_4_foo"
        }
      )");
  auto add_foo = [](absl::string_view s) { return absl::StrCat(s, "_foo"); };
  MP_EXPECT_OK(tool::TransformStreamNames(
      (*config.mutable_node())[0].mutable_input_stream(), add_foo));
  EXPECT_THAT(config, mediapipe::EqualsProto(expected_config));
}

TEST(SubgraphExpansionTest, TransformNames) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
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
      )");
  CalculatorGraphConfig expected_config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
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
      )");
  auto add_prefix = [](absl::string_view s) {
    return absl::StrCat("__sg0_", s);
  };
  MP_EXPECT_OK(tool::TransformNames(&config, add_prefix));
  EXPECT_THAT(config, mediapipe::EqualsProto(expected_config));
}

TEST(SubgraphExpansionTest, FindCorrespondingStreams) {
  CalculatorGraphConfig config1 =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "input_1"
        input_stream: "VIDEO:input_2"
        input_stream: "AUDIO:0:input_3"
        input_stream: "AUDIO:1:input_4"
      )");
  CalculatorGraphConfig config2 =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: "SomeSubgraph"
          input_stream: "foo"
          input_stream: "VIDEO:bar"
          input_stream: "AUDIO:0:baz"
          input_stream: "AUDIO:1:qux"
        }
      )");
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
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "input_1"
        input_stream: "AUDIO:0:input_3"
        input_stream: "AUDIO:1:input_4"
      )");
  CalculatorGraphConfig config2 =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: "SomeSubgraph"
          input_stream: "foo"
          input_stream: "VIDEO:bar"
          input_stream: "AUDIO:0:baz"
          input_stream: "AUDIO:1:qux"
        }
      )");
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
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "input_1"
        input_stream: "VIDEO:input_2"
        input_stream: "AUDIO:0:input_3"
      )");
  CalculatorGraphConfig config2 =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: "SomeSubgraph"
          input_stream: "foo"
          input_stream: "VIDEO:bar"
          input_stream: "AUDIO:0:baz"
          input_stream: "AUDIO:1:qux"
        }
      )");
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
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
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
      )");
  CalculatorGraphConfig supergraph =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          calculator: "SomeSubgraph"
          input_stream: "A:foo"
          input_stream: "B:bar"
          output_stream: "O:foobar"
          input_side_packet: "SI:flip"
          output_side_packet: "SO:flop"
        }
      )");
  // Note: graph input streams, output streams, and side packets on the
  // subgraph are not changed because they are going to be discarded anyway.
  CalculatorGraphConfig expected_subgraph =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
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
      )");
  MP_EXPECT_OK(tool::ConnectSubgraphStreams(supergraph.node()[0], &subgraph));
  EXPECT_THAT(subgraph, mediapipe::EqualsProto(expected_subgraph));
}

TEST(SubgraphExpansionTest, ExpandSubgraphs) {
  CalculatorGraphConfig supergraph =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        node {
          name: "simple_source"
          calculator: "SomeSourceCalculator"
          output_stream: "foo"
        }
        node { calculator: "TestSubgraph" input_stream: "DATA:foo" }
      )");
  CalculatorGraphConfig expected_graph =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
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
      )");
  MP_EXPECT_OK(tool::ExpandSubgraphs(&supergraph));
  EXPECT_THAT(supergraph, mediapipe::EqualsProto(expected_graph));
}

TEST(SubgraphExpansionTest, ValidateSubgraphFields) {
  CalculatorGraphConfig supergraph =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
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
      )");
  ::mediapipe::Status s1 = tool::ValidateSubgraphFields(supergraph.node(1));
  EXPECT_EQ(s1.code(), ::mediapipe::StatusCode::kInvalidArgument);
  EXPECT_THAT(s1.message(), testing::HasSubstr("foo_subgraph"));

  ::mediapipe::Status s2 = tool::ExpandSubgraphs(&supergraph);
  EXPECT_EQ(s2.code(), ::mediapipe::StatusCode::kInvalidArgument);
  EXPECT_THAT(s2.message(), testing::HasSubstr("foo_subgraph"));
}

// A test that captures the use case of CL 191001940. The "executor" field of
// a node inside a subgraph should be preserved, not mapped or mangled. This
// test will help us detect breakage of this use case when we implement
// subgraph executor support in the future.
TEST(SubgraphExpansionTest, ExecutorFieldOfNodeInSubgraphPreserved) {
  CalculatorGraphConfig supergraph =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
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
      )");
  CalculatorGraphConfig expected_graph = ::mediapipe::ParseTextProtoOrDie<
      CalculatorGraphConfig>(R"(
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
  )");
  MP_EXPECT_OK(tool::ExpandSubgraphs(&supergraph));
  EXPECT_THAT(supergraph, mediapipe::EqualsProto(expected_graph));
}

}  // namespace
}  // namespace mediapipe
