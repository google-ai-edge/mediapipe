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
//
// Facility for registering subgraphs that can be included in other graphs.

#ifndef MEDIAPIPE_FRAMEWORK_SUBGRAPH_H_
#define MEDIAPIPE_FRAMEWORK_SUBGRAPH_H_

#include "absl/base/macros.h"
#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/deps/registration.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/framework/tool/calculator_graph_template.pb.h"
#include "mediapipe/framework/tool/options_util.h"

namespace mediapipe {

// Instances of this class are responsible for providing a subgraph config.
// They are only used during graph construction. They do not stay alive once
// the graph is running.
class Subgraph {
 public:
  using SubgraphOptions = CalculatorGraphConfig::Node;
  Subgraph();
  virtual ~Subgraph();
  // Returns the config to use for one instantiation of the subgraph. The
  // nodes and generators in this config will replace the subgraph node in
  // the parent graph.
  // Subclasses may use the options argument to parameterize the config.
  // TODO: make this static?
  virtual ::mediapipe::StatusOr<CalculatorGraphConfig> GetConfig(
      const SubgraphOptions& options) = 0;

  // Returns options of a specific type.
  template <typename T>
  static T GetOptions(Subgraph::SubgraphOptions supgraph_options) {
    return tool::OptionsMap().Initialize(supgraph_options).Get<T>();
  }
};

using SubgraphRegistry = GlobalFactoryRegistry<std::unique_ptr<Subgraph>>;

#define REGISTER_MEDIAPIPE_GRAPH(name)                               \
  REGISTER_FACTORY_FUNCTION_QUALIFIED(::mediapipe::SubgraphRegistry, \
                                      subgraph_registration, name,   \
                                      absl::make_unique<name>)

// A graph factory holding a literal CalculatorGraphConfig.
class ProtoSubgraph : public Subgraph {
 public:
  ProtoSubgraph(const CalculatorGraphConfig& config);
  virtual ~ProtoSubgraph();
  virtual ::mediapipe::StatusOr<CalculatorGraphConfig> GetConfig(
      const Subgraph::SubgraphOptions& options);

 private:
  CalculatorGraphConfig config_;
};

// A graph factory holding a literal CalculatorGraphTemplate.
class TemplateSubgraph : public Subgraph {
 public:
  TemplateSubgraph(const CalculatorGraphTemplate& templ);
  virtual ~TemplateSubgraph();
  virtual ::mediapipe::StatusOr<CalculatorGraphConfig> GetConfig(
      const Subgraph::SubgraphOptions& options);

 private:
  CalculatorGraphTemplate templ_;
};

// A local registry of CalculatorGraphConfig definitions.
class GraphRegistry {
 public:
  // Creates a GraphRegistry derived from the GlobalFactoryRegistry,
  // which stores statically linked Subgraphs.
  GraphRegistry();

  // Creates a GraphRegistry derived from the specified FunctionRegistry,
  // which is used in place of the GlobalFactoryRegistry.
  // Ownership of the specified FunctionRegistry is not transferred.
  GraphRegistry(FunctionRegistry<std::unique_ptr<Subgraph>>* factories);

  // Registers a graph config builder type, using a factory function.
  void Register(const std::string& type_name,
                std::function<std::unique_ptr<Subgraph>()> factory);

  // Registers a graph config by name.
  void Register(const std::string& type_name,
                const CalculatorGraphConfig& config);

  // Registers a template graph config by name.
  void Register(const std::string& type_name,
                const CalculatorGraphTemplate& templ);

  // Returns true if the specified graph config is registered.
  bool IsRegistered(const std::string& ns, const std::string& type_name) const;

  // Returns the specified graph config.
  ::mediapipe::StatusOr<CalculatorGraphConfig> CreateByName(
      const std::string& ns, const std::string& type_name,
      const Subgraph::SubgraphOptions* options = nullptr) const;

  static GraphRegistry global_graph_registry;

 private:
  // The FunctionRegistry for dynamically loaded Subgraphs.
  mutable FunctionRegistry<std::unique_ptr<Subgraph>> local_factories_;
  // The FunctionRegistry for statically linked Subgraphs.
  // The global_factories_ registry is overridden by local_factories_.
  mutable FunctionRegistry<std::unique_ptr<Subgraph>>* global_factories_;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_SUBGRAPH_H_
