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

#include <algorithm>

#include "mediapipe/framework/port/logging.h"

namespace mediapipe {

TopologicalSorter::TopologicalSorter(int num_nodes) : num_nodes_(num_nodes) {
  CHECK_GE(num_nodes_, 0);
  adjacency_lists_.resize(num_nodes_);
}

void TopologicalSorter::AddEdge(int from, int to) {
  CHECK(!traversal_started_ && from < num_nodes_ && to < num_nodes_ &&
        from >= 0 && to >= 0);
  adjacency_lists_[from].push_back(to);
}

bool TopologicalSorter::GetNext(int* node_index, bool* cyclic,
                                std::vector<int>* output_cycle_nodes) {
  if (!traversal_started_) {
    // Iterates over all adjacency lists, and fills the indegree_ vector.
    indegree_.assign(num_nodes_, 0);
    for (int from = 0; from < num_nodes_; ++from) {
      std::vector<int>& adjacency_list = adjacency_lists_[from];
      // Eliminates duplicate edges.
      std::sort(adjacency_list.begin(), adjacency_list.end());
      adjacency_list.erase(
          std::unique(adjacency_list.begin(), adjacency_list.end()),
          adjacency_list.end());
      for (int to : adjacency_list) {
        ++indegree_[to];
      }
    }

    // Fills the nodes_with_zero_indegree_ vector.
    for (int i = 0; i < num_nodes_; ++i) {
      if (indegree_[i] == 0) {
        nodes_with_zero_indegree_.push(i);
      }
    }
    num_nodes_left_ = num_nodes_;
    traversal_started_ = true;
  }

  *cyclic = false;
  if (num_nodes_left_ == 0) {
    // Done the traversal.
    return false;
  }
  if (nodes_with_zero_indegree_.empty()) {
    *cyclic = true;
    FindCycle(output_cycle_nodes);
    return false;
  }

  // Gets the least node.
  --num_nodes_left_;
  *node_index = nodes_with_zero_indegree_.top();
  nodes_with_zero_indegree_.pop();
  // Swap out the adjacency list, since we won't need it afterwards,
  // to decrease memory usage.
  std::vector<int> adjacency_list;
  adjacency_list.swap(adjacency_lists_[*node_index]);

  // Updates the indegree_ vector and nodes_with_zero_indegree_ queue.
  for (int i = 0; i < adjacency_list.size(); ++i) {
    if (--indegree_[adjacency_list[i]] == 0) {
      nodes_with_zero_indegree_.push(adjacency_list[i]);
    }
  }
  return true;
}

void TopologicalSorter::FindCycle(std::vector<int>* cycle_nodes) {
  cycle_nodes->clear();
  // To find a cycle, we start a DFS from each yet-unvisited node and
  // try to find a cycle, if we don't find it then we know for sure that
  // no cycle is reachable from any of the explored nodes (so, we don't
  // explore them in later DFSs).
  std::vector<bool> no_cycle_reachable_from(num_nodes_, false);
  // The DFS stack will contain a chain of nodes, from the root of the
  // DFS to the current leaf.
  struct DfsState {
    int node;
    // Points at the first child node that we did *not* yet look at.
    int adjacency_list_index;
    explicit DfsState(int _node) : node(_node), adjacency_list_index(0) {}
  };
  std::vector<DfsState> dfs_stack;
  std::vector<bool> in_cur_stack(num_nodes_, false);

  for (int start_node = 0; start_node < num_nodes_; ++start_node) {
    if (no_cycle_reachable_from[start_node]) {
      continue;
    }
    // Starts the DFS.
    dfs_stack.push_back(DfsState(start_node));
    in_cur_stack[start_node] = true;
    while (!dfs_stack.empty()) {
      DfsState* cur_state = &dfs_stack.back();
      if (cur_state->adjacency_list_index >=
          adjacency_lists_[cur_state->node].size()) {
        no_cycle_reachable_from[cur_state->node] = true;
        in_cur_stack[cur_state->node] = false;
        dfs_stack.pop_back();
        continue;
      }
      // Looks at the current child, and increases the current state's
      // adjacency_list_index.
      const int child =
          adjacency_lists_[cur_state->node][cur_state->adjacency_list_index];
      ++(cur_state->adjacency_list_index);
      if (no_cycle_reachable_from[child]) {
        continue;
      }
      if (in_cur_stack[child]) {
        // We detected a cycle! Fills it and return.
        for (;;) {
          cycle_nodes->push_back(dfs_stack.back().node);
          if (dfs_stack.back().node == child) {
            std::reverse(cycle_nodes->begin(), cycle_nodes->end());
            return;
          }
          dfs_stack.pop_back();
        }
      }
      // Pushs the child onto the stack.
      dfs_stack.push_back(DfsState(child));
      in_cur_stack[child] = true;
    }
  }
  // If we're here, then all the DFS stopped, and they never encountered
  // a cycle (otherwise, we would have returned). Just exit; the output
  // vector has been cleared already.
}

}  // namespace mediapipe
