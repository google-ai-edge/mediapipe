
#include <memory>
#include <string>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/core/constant_side_packet_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/resources.h"
#include "mediapipe/framework/resources_service.h"

namespace mediapipe {
namespace {

using ::mediapipe::api2::Node;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::SideOutput;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::SidePacket;

constexpr absl::string_view kSubgraphResource =
    "mediapipe/framework/"
    "testdata/resource_subgraph.data";

constexpr absl::string_view kCalculatorResource =
    "mediapipe/framework/"
    "testdata/resource_calculator.data";

class TestResourcesCalculator : public Node {
 public:
  static constexpr SideOutput<std::string> kSideOut{"SIDE_OUT"};
  static constexpr Output<std::string> kOut{"OUT"};
  MEDIAPIPE_NODE_CONTRACT(kSideOut, kOut);

  absl::Status Open(CalculatorContext* cc) override {
    std::string data;
    MP_RETURN_IF_ERROR(
        cc->GetResources().ReadContents(kCalculatorResource, data));
    absl::StripAsciiWhitespace(&data);
    kSideOut(cc).Set(std::move(data));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    std::string data;
    MP_RETURN_IF_ERROR(
        cc->GetResources().ReadContents(kCalculatorResource, data));
    absl::StripAsciiWhitespace(&data);
    kOut(cc).Send(std::move(data));
    return tool::StatusStop();
  }
};
MEDIAPIPE_REGISTER_NODE(TestResourcesCalculator);

class TestResourcesSubgraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    std::string data;
    MP_RETURN_IF_ERROR(
        sc->GetResources().ReadContents(kSubgraphResource, data));
    absl::StripAsciiWhitespace(&data);

    Graph graph;
    auto& constants_node = graph.AddNode("ConstantSidePacketCalculator");
    auto& constants_options =
        constants_node
            .GetOptions<mediapipe::ConstantSidePacketCalculatorOptions>();
    constants_options.add_packet()->set_string_value(data);
    SidePacket<std::string> side_out =
        constants_node.SideOut("PACKET").Cast<std::string>();

    side_out.ConnectTo(graph.SideOut("SIDE_OUT"));

    return graph.GetConfig();
  }
};
REGISTER_MEDIAPIPE_GRAPH(TestResourcesSubgraph);

struct ResourceContentsPackets {
  Packet subgraph_side_out;
  Packet calculator_out;
  Packet calculator_side_out;
};

CalculatorGraphConfig BuildGraphProducingResourceContentsPackets() {
  Graph graph;

  auto& subgraph = graph.AddNode("TestResourcesSubgraph");
  subgraph.SideOut("SIDE_OUT").SetName("subgraph_side_out");

  auto& calculator = graph.AddNode("TestResourcesCalculator");
  calculator.SideOut("SIDE_OUT").SetName("calculator_side_out");
  calculator.Out("OUT").SetName("calculator_out");

  return graph.GetConfig();
}

absl::StatusOr<ResourceContentsPackets>
RunGraphAndCollectResourceContentsPackets(CalculatorGraph& calculator_graph) {
  Packet calculator_out;
  MP_RETURN_IF_ERROR(calculator_graph.ObserveOutputStream(
      "calculator_out", [&calculator_out](const Packet& packet) {
        ABSL_CHECK(calculator_out.IsEmpty());
        calculator_out = packet;
        return absl::OkStatus();
      }));
  MP_RETURN_IF_ERROR(calculator_graph.StartRun({}));
  MP_RETURN_IF_ERROR(calculator_graph.WaitUntilDone());

  MP_ASSIGN_OR_RETURN(
      Packet subgraph_side_out,
      calculator_graph.GetOutputSidePacket("subgraph_side_out"));
  MP_ASSIGN_OR_RETURN(
      Packet calculator_side_out,
      calculator_graph.GetOutputSidePacket("calculator_side_out"));
  return ResourceContentsPackets{
      .subgraph_side_out = std::move(subgraph_side_out),
      .calculator_out = std::move(calculator_out),
      .calculator_side_out = std::move(calculator_side_out)};
}

TEST(CalculatorGraphResourcesTest, GraphAndContextsHaveDefaultResources) {
  CalculatorGraph calculator_graph;
  MP_ASSERT_OK(calculator_graph.Initialize(
      BuildGraphProducingResourceContentsPackets()));
  MP_ASSERT_OK_AND_ASSIGN(
      ResourceContentsPackets packets,
      RunGraphAndCollectResourceContentsPackets(calculator_graph));

  EXPECT_EQ(packets.subgraph_side_out.Get<std::string>(),
            "File system subgraph contents");
  EXPECT_EQ(packets.calculator_out.Get<std::string>(),
            "File system calculator contents");
  EXPECT_EQ(packets.calculator_side_out.Get<std::string>(),
            "File system calculator contents");
}

class CustomResources : public Resources {
 public:
  absl::Status ReadContents(absl::string_view resource_id, std::string& output,
                            const Resources::Options& options) const final {
    if (resource_id == kSubgraphResource) {
      output = "Custom subgraph contents";
    } else if (resource_id == kCalculatorResource) {
      output = "Custom calculator contents";
    } else {
      return absl::NotFoundError(
          absl::StrCat("Resource [", resource_id, "] not found."));
    }
    return absl::OkStatus();
  }
};

TEST(CalculatorGraphResourcesTest, CustomResourcesCanBeSetOnGraph) {
  CalculatorGraph calculator_graph;
  std::shared_ptr<Resources> resources = std::make_shared<CustomResources>();
  MP_ASSERT_OK(calculator_graph.SetServiceObject(kResourcesService,
                                                 std::move(resources)));
  MP_ASSERT_OK(calculator_graph.Initialize(
      BuildGraphProducingResourceContentsPackets()));
  MP_ASSERT_OK_AND_ASSIGN(
      ResourceContentsPackets packets,
      RunGraphAndCollectResourceContentsPackets(calculator_graph));

  EXPECT_EQ(packets.subgraph_side_out.Get<std::string>(),
            "Custom subgraph contents");
  EXPECT_EQ(packets.calculator_out.Get<std::string>(),
            "Custom calculator contents");
  EXPECT_EQ(packets.calculator_side_out.Get<std::string>(),
            "Custom calculator contents");
}

class CustomizedDefaultResources : public Resources {
 public:
  absl::Status ReadContents(absl::string_view resource_id, std::string& output,
                            const Resources::Options& options) const final {
    MP_RETURN_IF_ERROR(
        default_resources_->ReadContents(resource_id, output, options));
    output.insert(0, "Customized: ");
    return absl::OkStatus();
  }

 private:
  std::unique_ptr<Resources> default_resources_ = CreateDefaultResources();
};

TEST(CalculatorGraphResourcesTest,
     CustomResourcesUsingDefaultResourcesCanBeSetOnGraph) {
  CalculatorGraph calculator_graph;
  std::shared_ptr<Resources> resources =
      std::make_shared<CustomizedDefaultResources>();
  MP_ASSERT_OK(calculator_graph.SetServiceObject(kResourcesService,
                                                 std::move(resources)));
  MP_ASSERT_OK(calculator_graph.Initialize(
      BuildGraphProducingResourceContentsPackets()));
  MP_ASSERT_OK_AND_ASSIGN(
      ResourceContentsPackets packets,
      RunGraphAndCollectResourceContentsPackets(calculator_graph));

  EXPECT_EQ(packets.subgraph_side_out.Get<std::string>(),
            "Customized: File system subgraph contents");
  EXPECT_EQ(packets.calculator_out.Get<std::string>(),
            "Customized: File system calculator contents");
  EXPECT_EQ(packets.calculator_side_out.Get<std::string>(),
            "Customized: File system calculator contents");
}

}  // namespace
}  // namespace mediapipe
