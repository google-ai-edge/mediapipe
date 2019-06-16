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

#include "mediapipe/framework/deps/topologicalsorter.h"

#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {

TEST(TopologicalSorterTest, NoConnection) {
  TopologicalSorter sorter(3);
  std::vector<int> expected_result({0, 1, 2});

  int visited = 0;
  int node_index;
  bool cyclic;
  std::vector<int> cycle_nodes;
  while (sorter.GetNext(&node_index, &cyclic, &cycle_nodes)) {
    EXPECT_EQ(expected_result[visited], node_index);
    ++visited;
  }
  ASSERT_FALSE(cyclic);
  EXPECT_EQ(3, visited);
}

TEST(TopologicalSorterTest, SimpleDAG) {
  TopologicalSorter sorter(5);
  sorter.AddEdge(4, 0);
  sorter.AddEdge(4, 1);
  sorter.AddEdge(4, 2);
  sorter.AddEdge(0, 3);
  sorter.AddEdge(1, 3);
  sorter.AddEdge(3, 2);
  std::vector<int> expected_result({4, 0, 1, 3, 2});

  int visited = 0;
  int node_index;
  bool cyclic;
  std::vector<int> cycle_nodes;
  while (sorter.GetNext(&node_index, &cyclic, &cycle_nodes)) {
    EXPECT_EQ(expected_result[visited], node_index);
    ++visited;
  }
  ASSERT_FALSE(cyclic);
  EXPECT_EQ(5, visited);
}

TEST(TopologicalSorterTest, DuplicatedEdges) {
  TopologicalSorter sorter(5);
  sorter.AddEdge(3, 2);
  sorter.AddEdge(4, 0);
  sorter.AddEdge(4, 2);
  sorter.AddEdge(4, 1);
  sorter.AddEdge(3, 2);
  sorter.AddEdge(4, 2);
  sorter.AddEdge(1, 3);
  sorter.AddEdge(0, 3);
  sorter.AddEdge(1, 3);
  sorter.AddEdge(3, 2);
  std::vector<int> expected_result({4, 0, 1, 3, 2});

  int visited = 0;
  int node_index;
  bool cyclic;
  std::vector<int> cycle_nodes;
  while (sorter.GetNext(&node_index, &cyclic, &cycle_nodes)) {
    EXPECT_EQ(expected_result[visited], node_index);
    ++visited;
  }
  ASSERT_FALSE(cyclic);
  EXPECT_EQ(5, visited);
}

TEST(TopologicalSorterTest, Cycle) {
  // Cycle: 1->3->2->1
  TopologicalSorter sorter(5);
  sorter.AddEdge(4, 0);
  sorter.AddEdge(4, 1);
  sorter.AddEdge(4, 2);
  sorter.AddEdge(0, 3);
  sorter.AddEdge(1, 3);
  sorter.AddEdge(3, 2);
  sorter.AddEdge(2, 1);

  int node_index;
  bool cyclic;
  std::vector<int> cycle_nodes;
  while (sorter.GetNext(&node_index, &cyclic, &cycle_nodes)) {
  }

  EXPECT_TRUE(cyclic);
  std::vector<int> expected_cycle({1, 3, 2});
  ASSERT_EQ(3, cycle_nodes.size());
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(expected_cycle[i], cycle_nodes[i]);
  }
}

}  // namespace mediapipe
