// Copyright 2024 The MediaPipe Authors.
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

#include "mediapipe/calculators/util/resource_provider_calculator.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/util/resource_provider_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/resources.h"
#include "mediapipe/framework/resources_service.h"
#include "mediapipe/util/resources_test_util.h"

namespace mediapipe::api2 {
namespace {

using ::mediapipe::Packet;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::SidePacket;

TEST(ResourceProviderCalculatorTest, CanGetSingleResourceUsingOptions) {
  Graph graph;

  auto& res_node = graph.AddNode<ResourceProviderCalculator>();
  auto& res_opts = res_node.GetOptions<ResourceProviderCalculatorOptions>();
  res_opts.add_resource_id("$RES_ID");

  SidePacket<Resource> res =
      res_node[ResourceProviderCalculator::kResources][0];
  res.SetName("resource");

  // Run graph.
  CalculatorGraph calculator_graph;
  std::shared_ptr<Resources> resources =
      CreateInMemoryResources({{"$RES_ID", "Some file blob"}});
  MP_ASSERT_OK(calculator_graph.SetServiceObject(kResourcesService,
                                                 std::move(resources)));
  MP_ASSERT_OK(calculator_graph.Initialize(graph.GetConfig()));
  MP_ASSERT_OK(calculator_graph.Run());

  // Check results.
  MP_ASSERT_OK_AND_ASSIGN(Packet resource_packet,
                          calculator_graph.GetOutputSidePacket("resource"));
  ASSERT_FALSE(resource_packet.IsEmpty());
  EXPECT_EQ(resource_packet.Get<Resource>().ToStringView(), "Some file blob");
}

TEST(ResourceProviderCalculatorTest, CanGetMultipleResourcesUsingOptions) {
  constexpr int kNumResources = 3;
  absl::flat_hash_map<std::string, std::string> resources_in_memory;

  Graph graph;

  auto& res_node = graph.AddNode<ResourceProviderCalculator>();
  auto& res_opts = res_node.GetOptions<ResourceProviderCalculatorOptions>();
  for (int i = 0; i < kNumResources; ++i) {
    std::string res_id = absl::StrCat("$RES_ID", i);
    res_opts.add_resource_id(res_id);
    SidePacket<Resource> res =
        res_node[ResourceProviderCalculator::kResources][i];
    res.SetName(absl::StrCat("resource", i));

    // Put corresponding resource.
    resources_in_memory[res_id] = absl::StrCat("Some file blob ", i);
  }

  // Run graph.
  CalculatorGraph calculator_graph;
  std::shared_ptr<Resources> resources =
      CreateInMemoryResources(std::move(resources_in_memory));
  MP_ASSERT_OK(calculator_graph.SetServiceObject(kResourcesService,
                                                 std::move(resources)));
  CalculatorGraphConfig config = graph.GetConfig();
  MP_ASSERT_OK(calculator_graph.Initialize(config));
  MP_ASSERT_OK(calculator_graph.Run());

  // Check results.
  for (int i = 0; i < kNumResources; ++i) {
    MP_ASSERT_OK_AND_ASSIGN(
        Packet resource_packet,
        calculator_graph.GetOutputSidePacket(absl::StrCat("resource", i)));
    ASSERT_FALSE(resource_packet.IsEmpty());
    EXPECT_EQ(resource_packet.Get<Resource>().ToStringView(),
              absl::StrCat("Some file blob ", i));
  }
}

TEST(ResourceProviderCalculatorTest, CanGetSingleResourceUsingSidePacket) {
  Graph graph;

  SidePacket<std::string> resource_id =
      graph.SideIn(0).SetName("res_id").Cast<std::string>();

  auto& res_node = graph.AddNode<ResourceProviderCalculator>();
  resource_id.ConnectTo(res_node[ResourceProviderCalculator::kIds]);
  SidePacket<Resource> res =
      res_node[ResourceProviderCalculator::kResources][0];
  res.SetName("resource");

  // Run graph.
  CalculatorGraph calculator_graph;
  std::shared_ptr<Resources> resources =
      CreateInMemoryResources({{"$RES_ID", "Some file blob"}});
  MP_ASSERT_OK(calculator_graph.SetServiceObject(kResourcesService,
                                                 std::move(resources)));
  MP_ASSERT_OK(calculator_graph.Initialize(graph.GetConfig()));
  MP_ASSERT_OK(
      calculator_graph.Run({{"res_id", MakePacket<std::string>("$RES_ID")}}));

  // Check results.
  MP_ASSERT_OK_AND_ASSIGN(Packet resource_packet,
                          calculator_graph.GetOutputSidePacket("resource"));
  ASSERT_FALSE(resource_packet.IsEmpty());
  EXPECT_EQ(resource_packet.Get<Resource>().ToStringView(), "Some file blob");
}

TEST(ResourceProviderCalculatorTest, CanGetMultipleResourcesUsingSidePackets) {
  constexpr int kNumResources = 3;
  absl::flat_hash_map<std::string, std::string> resources_in_memory;
  std::map<std::string, mediapipe::Packet> resource_ids_side_packets;

  Graph graph;

  std::vector<SidePacket<std::string>> side_packets;
  for (int i = 0; i < kNumResources; ++i) {
    std::string res_id_side_name = absl::StrCat("res_id", i);
    side_packets.push_back(
        graph.SideIn(i).SetName(res_id_side_name).Cast<std::string>());

    std::string res_id = absl::StrCat("$RES_ID", i);
    resource_ids_side_packets[res_id_side_name] =
        MakePacket<std::string>(res_id);
    resources_in_memory[res_id] = absl::StrCat("Some file blob ", i);
  }

  auto& res_node = graph.AddNode<ResourceProviderCalculator>();
  for (int i = 0; i < side_packets.size(); ++i) {
    side_packets[i].ConnectTo(res_node[ResourceProviderCalculator::kIds][i]);
    SidePacket<Resource> res =
        res_node[ResourceProviderCalculator::kResources][i];
    res.SetName(absl::StrCat("resource", i));
  }

  // Run graph.
  CalculatorGraph calculator_graph;
  std::shared_ptr<Resources> resources =
      CreateInMemoryResources(std::move(resources_in_memory));
  MP_ASSERT_OK(calculator_graph.SetServiceObject(kResourcesService,
                                                 std::move(resources)));
  CalculatorGraphConfig config = graph.GetConfig();
  MP_ASSERT_OK(calculator_graph.Initialize(config));
  MP_ASSERT_OK(calculator_graph.Run(resource_ids_side_packets));

  // Check results.
  for (int i = 0; i < kNumResources; ++i) {
    MP_ASSERT_OK_AND_ASSIGN(
        Packet resource_packet,
        calculator_graph.GetOutputSidePacket(absl::StrCat("resource", i)));
    ASSERT_FALSE(resource_packet.IsEmpty());
    EXPECT_EQ(resource_packet.Get<Resource>().ToStringView(),
              absl::StrCat("Some file blob ", i));
  }
}

}  // namespace
}  // namespace mediapipe::api2
