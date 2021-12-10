#include "mediapipe/framework/validated_graph_config.h"

#include <string_view>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/graph_service.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

class NoOp : public mediapipe::api2::Node {
 public:
  static constexpr mediapipe::api2::Input<int>::Optional kInputNotNeeded{"NN"};
  static constexpr mediapipe::api2::Output<int>::Optional kOutputNotNeeded{
      "NN"};
  MEDIAPIPE_NODE_CONTRACT(kInputNotNeeded, kOutputNotNeeded);
  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }
};

using CalculatorA = NoOp;
MEDIAPIPE_REGISTER_NODE(CalculatorA);
using CalculatorB = NoOp;
MEDIAPIPE_REGISTER_NODE(CalculatorB);
using CalculatorC = NoOp;
MEDIAPIPE_REGISTER_NODE(CalculatorC);

CalculatorGraphConfig ExpectedConfig(const std::string& node_name) {
  CalculatorGraphConfig config;
  config.add_node()->set_calculator(node_name);
  config.add_executor();
  return config;
}

CalculatorGraphConfig ExpectedConfigExpandedFromGraph(
    const std::string& graph_name, const std::string& node_name) {
  CalculatorGraphConfig config;
  auto* node = config.add_node();
  node->set_calculator(node_name);
  node->set_name(
      absl::StrCat(absl::AsciiStrToLower(graph_name), "__", node_name));
  config.add_executor();
  return config;
}

class AlwaysCalculatorALegacySubgraph : public Subgraph {
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      const SubgraphOptions& options) override {
    return ExpectedConfig("CalculatorA");
  }
};
REGISTER_MEDIAPIPE_GRAPH(AlwaysCalculatorALegacySubgraph);

TEST(ValidatedGraphConfigTest, InitializeByTypeLegacySubgraphHardcoded) {
  ValidatedGraphConfig config;
  MP_EXPECT_OK(config.Initialize("AlwaysCalculatorALegacySubgraph",
                                 /*options=*/nullptr,
                                 /*graph_registry=*/nullptr,
                                 /*service_manager=*/nullptr));
  ASSERT_TRUE(config.Initialized());
  EXPECT_THAT(config.Config(), EqualsProto(ExpectedConfig("CalculatorA")));
}

TEST(ValidatedGraphConfigTest, InitializeLegacySubgraphHardcoded) {
  CalculatorGraphConfig graph;
  graph.add_node()->set_calculator("AlwaysCalculatorALegacySubgraph");

  ValidatedGraphConfig config;
  MP_EXPECT_OK(config.Initialize(graph,
                                 /*graph_registry=*/nullptr,
                                 /*service_manager=*/nullptr));
  ASSERT_TRUE(config.Initialized());
  EXPECT_THAT(config.Config(),
              EqualsProto(ExpectedConfigExpandedFromGraph(
                  "AlwaysCalculatorALegacySubgraph", "CalculatorA")));
}

class AlwaysCalculatorASubgraph : public Subgraph {
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    return ExpectedConfig("CalculatorA");
  }
};
REGISTER_MEDIAPIPE_GRAPH(AlwaysCalculatorASubgraph);

TEST(ValidatedGraphConfigTest, InitializeByTypeSubgraphHardcoded) {
  ValidatedGraphConfig config;
  MP_EXPECT_OK(config.Initialize("AlwaysCalculatorASubgraph",
                                 /*options=*/nullptr,
                                 /*graph_registry=*/nullptr,
                                 /*service_manager=*/nullptr));
  ASSERT_TRUE(config.Initialized());
  EXPECT_THAT(config.Config(), EqualsProto(ExpectedConfig("CalculatorA")));
}

TEST(ValidatedGraphConfigTest, InitializeSubgraphHardcoded) {
  CalculatorGraphConfig graph;
  graph.add_node()->set_calculator("AlwaysCalculatorASubgraph");

  ValidatedGraphConfig config;
  MP_EXPECT_OK(config.Initialize(graph,
                                 /*graph_registry=*/nullptr,
                                 /*service_manager=*/nullptr));
  ASSERT_TRUE(config.Initialized());
  EXPECT_THAT(config.Config(),
              EqualsProto(ExpectedConfigExpandedFromGraph(
                  "AlwaysCalculatorASubgraph", "CalculatorA")));
}

const mediapipe::GraphService<std::string> kStringTestService{
    "mediapipe::StringTestService"};

class TestServiceSubgraph : public Subgraph {
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    return ExpectedConfig(sc->Service(kStringTestService).GetObject());
  }
};
REGISTER_MEDIAPIPE_GRAPH(TestServiceSubgraph);

TEST(ValidatedGraphConfigTest, InitializeByTypeSubgraphWithServiceCalculatorB) {
  for (const std::string& calculator_name :
       {"CalculatorA", "CalculatorB", "CalculatorC"}) {
    ValidatedGraphConfig config;
    GraphServiceManager service_manager;
    MP_ASSERT_OK(service_manager.SetServiceObject(
        kStringTestService, std::make_shared<std::string>(calculator_name)));
    MP_EXPECT_OK(config.Initialize("TestServiceSubgraph",
                                   /*options=*/nullptr,
                                   /*graph_registry=*/nullptr,
                                   /*service_manager=*/&service_manager));
    ASSERT_TRUE(config.Initialized());
    EXPECT_THAT(config.Config(), EqualsProto(ExpectedConfig(calculator_name)));
  }
}

TEST(ValidatedGraphConfigTest, InitializeSubgraphWithServiceCalculatorB) {
  for (const std::string& calculator_name :
       {"CalculatorA", "CalculatorB", "CalculatorC"}) {
    CalculatorGraphConfig graph;
    graph.add_node()->set_calculator("TestServiceSubgraph");

    ValidatedGraphConfig config;
    GraphServiceManager service_manager;
    MP_ASSERT_OK(service_manager.SetServiceObject(
        kStringTestService, std::make_shared<std::string>(calculator_name)));
    MP_EXPECT_OK(config.Initialize(graph,
                                   /*graph_registry=*/nullptr,
                                   /*subgraph_options=*/nullptr,
                                   /*service_manager=*/&service_manager));
    ASSERT_TRUE(config.Initialized());
    EXPECT_THAT(config.Config(), EqualsProto(ExpectedConfigExpandedFromGraph(
                                     "TestServiceSubgraph", calculator_name)));
  }
}

}  // namespace mediapipe
