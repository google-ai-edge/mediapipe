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

#ifndef MEDIAPIPE_FRAMEWORK_GRAPH_SERVICE_H_
#define MEDIAPIPE_FRAMEWORK_GRAPH_SERVICE_H_

#include <memory>

namespace mediapipe {

// The GraphService API can be used to define extensions to a graph's execution
// environment. These are, essentially, graph-level singletons, and are
// available to all calculators in the graph (and in any subgraphs) without
// requiring a manual connection.
//
// IMPORTANT: this is an experimental API. Get in touch with the MediaPipe team
// if you want to use it. In most cases, you should use a side packet instead.

struct GraphServiceBase {
  constexpr GraphServiceBase(const char* key) : key(key) {}

  const char* key;
};

template <typename T>
struct GraphService : public GraphServiceBase {
  using type = T;
  using packet_type = std::shared_ptr<T>;

  constexpr GraphService(const char* key) : GraphServiceBase(key) {}
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_GRAPH_SERVICE_H_
