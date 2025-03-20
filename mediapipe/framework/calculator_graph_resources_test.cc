#include <memory>
#include <string>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/core/constant_side_packet_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/resources.h"
#include "mediapipe/framework/resources_service.h"
#include "mediapipe/framework/testdata/resource_path.pb.h"

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
  static constexpr SideOutput<Resource> kSideOut{"SIDE_OUT"};
  static constexpr Output<Resource> kOut{"OUT"};
  MEDIAPIPE_NODE_CONTRACT(kSideOut, kOut);

  absl::Status Open(CalculatorContext* cc) override {
    MP_ASSIGN_OR_RETURN(
        std::unique_ptr<Resource> resource,
        cc->GetResources().Get(
            cc->Options<mediapipe::ResourcePathOptions>().path()));
    kSideOut(cc).Set(api2::PacketAdopting(std::move(resource)));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    MP_ASSIGN_OR_RETURN(
        std::unique_ptr<Resource> resource,
        cc->GetResources().Get(
            cc->Options<mediapipe::ResourcePathOptions>().path()));
    kOut(cc).Send(std::move(resource));
    return tool::StatusStop();
  }
};
MEDIAPIPE_REGISTER_NODE(TestResourcesCalculator);

class TestResourcesSubgraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    MP_ASSIGN_OR_RETURN(
        std::unique_ptr<Resource> resource,
        sc->GetResources().Get(
            sc->Options<mediapipe::ResourcePathOptions>().path()));
    Graph graph;
    auto& constants_node = graph.AddNode("ConstantSidePacketCalculator");
    auto& constants_options =
        constants_node
            .GetOptions<mediapipe::ConstantSidePacketCalculatorOptions>();
    constants_options.add_packet()->mutable_string_value()->append(
        resource->ToStringView());
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

CalculatorGraphConfig BuildGraphProducingResourceContentsPackets(
    absl::string_view calculator_path, absl::string_view subgraph_path) {
  Graph graph;

  auto& subgraph = graph.AddNode("TestResourcesSubgraph");
  subgraph.GetOptions<mediapipe::ResourcePathOptions>().set_path(
      std::string(subgraph_path));
  subgraph.SideOut("SIDE_OUT").SetName("subgraph_side_out");

  auto& calculator = graph.AddNode("TestResourcesCalculator");
  calculator.GetOptions<mediapipe::ResourcePathOptions>().set_path(
      std::string(calculator_path));
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
  MP_ASSERT_OK(
      calculator_graph.Initialize(BuildGraphProducingResourceContentsPackets(
          kCalculatorResource, kSubgraphResource)));
  MP_ASSERT_OK_AND_ASSIGN(
      ResourceContentsPackets packets,
      RunGraphAndCollectResourceContentsPackets(calculator_graph));

  EXPECT_EQ(packets.subgraph_side_out.Get<std::string>(),
            "File system subgraph contents\n");
  EXPECT_EQ(packets.calculator_out.Get<Resource>().ToStringView(),
            "File system calculator contents\n");
  EXPECT_EQ(packets.calculator_side_out.Get<Resource>().ToStringView(),
            "File system calculator contents\n");
}

constexpr absl::string_view kCustomSubgraphContents =
    "Custom subgraph contents";
constexpr absl::string_view kCustomCalculatorContents =
    "Custom calculator contents";

class CustomResources : public Resources {
 public:
  // Avoids copy of kCustomSubtraph/CalculatorContents - while it's not that
  // beneficial for these specific strings, but it showcases one can avoid
  // copying whole ML models.
  absl::StatusOr<std::unique_ptr<Resource>> Get(
      absl::string_view resource_id,
      const Resources::Options& options) const final {
    std::unique_ptr<Resource> resource;
    if (resource_id == kSubgraphResource) {
      resource = MakeNoCleanupResource(kCustomSubgraphContents.data(),
                                       kCustomSubgraphContents.size());
    } else if (resource_id == kCalculatorResource) {
      resource = MakeNoCleanupResource(kCustomCalculatorContents.data(),
                                       kCustomCalculatorContents.size());
    } else {
      return absl::NotFoundError(
          absl::StrCat("Resource [", resource_id, "] not found."));
    }
    return resource;
  }
};

TEST(CalculatorGraphResourcesTest, CustomResourcesCanBeSetOnGraph) {
  CalculatorGraph calculator_graph;
  std::shared_ptr<Resources> resources = std::make_shared<CustomResources>();
  MP_ASSERT_OK(calculator_graph.SetServiceObject(kResourcesService,
                                                 std::move(resources)));
  MP_ASSERT_OK(
      calculator_graph.Initialize(BuildGraphProducingResourceContentsPackets(
          kCalculatorResource, kSubgraphResource)));
  MP_ASSERT_OK_AND_ASSIGN(
      ResourceContentsPackets packets,
      RunGraphAndCollectResourceContentsPackets(calculator_graph));

  EXPECT_EQ(packets.subgraph_side_out.Get<std::string>(),
            "Custom subgraph contents");
  EXPECT_EQ(packets.calculator_out.Get<Resource>().ToStringView(),
            "Custom calculator contents");
  EXPECT_EQ(packets.calculator_side_out.Get<Resource>().ToStringView(),
            "Custom calculator contents");
}

class CustomizedDefaultResources : public Resources {
 public:
  absl::StatusOr<std::unique_ptr<Resource>> Get(
      absl::string_view resource_id,
      const Resources::Options& options) const final {
    MP_ASSIGN_OR_RETURN(std::unique_ptr<Resource> output,
                        default_resources_->Get(resource_id, options));
    std::string contents = std::move(*output).ReleaseOrCopyAsString();
    contents.insert(0, "Customized: ");
    return MakeStringResource(std::move(contents));
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
  MP_ASSERT_OK(
      calculator_graph.Initialize(BuildGraphProducingResourceContentsPackets(
          kCalculatorResource, kSubgraphResource)));
  MP_ASSERT_OK_AND_ASSIGN(
      ResourceContentsPackets packets,
      RunGraphAndCollectResourceContentsPackets(calculator_graph));

  EXPECT_EQ(packets.subgraph_side_out.Get<std::string>(),
            "Customized: File system subgraph contents\n");
  EXPECT_EQ(packets.calculator_out.Get<Resource>().ToStringView(),
            "Customized: File system calculator contents\n");
  EXPECT_EQ(packets.calculator_side_out.Get<Resource>().ToStringView(),
            "Customized: File system calculator contents\n");
}

TEST(CalculatorGraphResourcesTest,
     DefaultResourcesWithMappingCanBeSetAndUsedOnGraph) {
  CalculatorGraph calculator_graph;
  std::shared_ptr<Resources> resources = CreateDefaultResourcesWithMapping(
      {{"$CALCULATOR_PATH", std::string(kCalculatorResource)},
       {"$SUBGRAPH_PATH", std::string(kSubgraphResource)}});
  MP_ASSERT_OK(calculator_graph.SetServiceObject(kResourcesService,
                                                 std::move(resources)));
  MP_ASSERT_OK(
      calculator_graph.Initialize(BuildGraphProducingResourceContentsPackets(
          "$CALCULATOR_PATH", "$SUBGRAPH_PATH")));
  MP_ASSERT_OK_AND_ASSIGN(
      ResourceContentsPackets packets,
      RunGraphAndCollectResourceContentsPackets(calculator_graph));

  EXPECT_EQ(packets.subgraph_side_out.Get<std::string>(),
            "File system subgraph contents\n");
  EXPECT_EQ(packets.calculator_out.Get<Resource>().ToStringView(),
            "File system calculator contents\n");
  EXPECT_EQ(packets.calculator_side_out.Get<Resource>().ToStringView(),
            "File system calculator contents\n");
}

}  // namespace
}  // namespace mediapipe
