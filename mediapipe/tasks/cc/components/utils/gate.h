/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_CC_COMPONENTS_UTILS_GATE_H_
#define MEDIAPIPE_TASKS_CC_COMPONENTS_UTILS_GATE_H_

#include <utility>

#include "mediapipe/calculators/core/gate_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"

namespace mediapipe {
namespace tasks {
namespace components {
namespace utils {

using ::mediapipe::api2::builder::SideSource;
using ::mediapipe::api2::builder::Source;

// Utility class that simplifies allowing (gating) multiple streams.
class AllowGate {
 public:
  AllowGate(Source<bool> allow, mediapipe::api2::builder::Graph& graph)
      : node_(AddSourceGate(allow, graph)) {}
  AllowGate(SideSource<bool> allow, mediapipe::api2::builder::Graph& graph)
      : node_(AddSideSourceGate(allow, graph)) {}

  // Move-only
  AllowGate(AllowGate&& allow_gate) = default;
  AllowGate& operator=(AllowGate&& allow_gate) = default;

  template <typename T>
  Source<T> Allow(Source<T> source) {
    source >> node_.In(index_);
    return node_.Out(index_++).Cast<T>();
  }

 private:
  template <typename T>
  static mediapipe::api2::builder::GenericNode& AddSourceGate(
      T allow, mediapipe::api2::builder::Graph& graph) {
    auto& gate_node = graph.AddNode("GateCalculator");
    allow >> gate_node.In("ALLOW");
    return gate_node;
  }

  template <typename T>
  static mediapipe::api2::builder::GenericNode& AddSideSourceGate(
      T allow, mediapipe::api2::builder::Graph& graph) {
    auto& gate_node = graph.AddNode("GateCalculator");
    allow >> gate_node.SideIn("ALLOW");
    return gate_node;
  }

  mediapipe::api2::builder::GenericNode& node_;
  int index_ = 0;
};

// Utility class that simplifies disallowing (gating) multiple streams.
class DisallowGate {
 public:
  DisallowGate(Source<bool> disallow, mediapipe::api2::builder::Graph& graph)
      : node_(AddSourceGate(disallow, graph)) {}
  DisallowGate(SideSource<bool> disallow,
               mediapipe::api2::builder::Graph& graph)
      : node_(AddSideSourceGate(disallow, graph)) {}

  // Move-only
  DisallowGate(DisallowGate&& disallow_gate) = default;
  DisallowGate& operator=(DisallowGate&& disallow_gate) = default;

  template <typename T>
  Source<T> Disallow(Source<T> source) {
    source >> node_.In(index_);
    return node_.Out(index_++).Cast<T>();
  }

 private:
  template <typename T>
  static mediapipe::api2::builder::GenericNode& AddSourceGate(
      T disallow, mediapipe::api2::builder::Graph& graph) {
    auto& gate_node = graph.AddNode("GateCalculator");
    auto& gate_node_opts =
        gate_node.GetOptions<mediapipe::GateCalculatorOptions>();
    // Supposedly, the most popular configuration for MediaPipe Tasks team
    // graphs. Hence, intentionally hard coded to catch and verify any other use
    // case (should help to workout a common approach and have a recommended way
    // of blocking streams).
    gate_node_opts.set_empty_packets_as_allow(true);
    disallow >> gate_node.In("DISALLOW");
    return gate_node;
  }

  template <typename T>
  static mediapipe::api2::builder::GenericNode& AddSideSourceGate(
      T disallow, mediapipe::api2::builder::Graph& graph) {
    auto& gate_node = graph.AddNode("GateCalculator");
    auto& gate_node_opts =
        gate_node.GetOptions<mediapipe::GateCalculatorOptions>();
    gate_node_opts.set_empty_packets_as_allow(true);
    disallow >> gate_node.SideIn("DISALLOW");
    return gate_node;
  }

  mediapipe::api2::builder::GenericNode& node_;
  int index_ = 0;
};

// Updates graph to drop @value stream packet if corresponding @condition stream
// packet holds true.
template <class T>
Source<T> DisallowIf(Source<T> value, Source<bool> condition,
                     mediapipe::api2::builder::Graph& graph) {
  return DisallowGate(condition, graph).Disallow(value);
}

// Updates graph to drop @value stream packet if corresponding @condition stream
// packet holds true.
template <class T>
Source<T> DisallowIf(Source<T> value, SideSource<bool> condition,
                     mediapipe::api2::builder::Graph& graph) {
  return DisallowGate(condition, graph).Disallow(value);
}

// Updates graph to pass through @value stream packet if corresponding
// @condition stream packet holds true.
template <class T>
Source<T> AllowIf(Source<T> value, Source<bool> allow,
                  mediapipe::api2::builder::Graph& graph) {
  return AllowGate(allow, graph).Allow(value);
}

// Updates graph to pass through @value stream packet if corresponding
// @condition side stream packet holds true.
template <class T>
Source<T> AllowIf(Source<T> value, SideSource<bool> allow,
                  mediapipe::api2::builder::Graph& graph) {
  return AllowGate(allow, graph).Allow(value);
}

}  // namespace utils
}  // namespace components
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_COMPONENTS_UTILS_GATE_H_
