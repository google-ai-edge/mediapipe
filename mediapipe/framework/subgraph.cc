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

#include "mediapipe/framework/subgraph.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/tool/template_expander.h"

namespace mediapipe {

Subgraph::Subgraph() {}

Subgraph::~Subgraph() {}

ProtoSubgraph::ProtoSubgraph(const CalculatorGraphConfig& config)
    : config_(config) {}

ProtoSubgraph::~ProtoSubgraph() {}

absl::StatusOr<CalculatorGraphConfig> ProtoSubgraph::GetConfig(
    const Subgraph::SubgraphOptions& options) {
  return config_;
}

TemplateSubgraph::TemplateSubgraph(const CalculatorGraphTemplate& templ)
    : templ_(templ) {}

TemplateSubgraph::~TemplateSubgraph() {}

absl::StatusOr<CalculatorGraphConfig> TemplateSubgraph::GetConfig(
    const Subgraph::SubgraphOptions& options) {
  TemplateDict arguments =
      Subgraph::GetOptions<mediapipe::TemplateSubgraphOptions>(options).dict();
  tool::TemplateExpander expander;
  CalculatorGraphConfig config;
  MP_RETURN_IF_ERROR(expander.ExpandTemplates(arguments, templ_, &config));
  return config;
}

GraphRegistry GraphRegistry::global_graph_registry;

GraphRegistry::GraphRegistry()
    : global_factories_(SubgraphRegistry::functions()) {}

GraphRegistry::GraphRegistry(
    FunctionRegistry<std::unique_ptr<Subgraph>>* factories)
    : global_factories_(factories) {}

void GraphRegistry::Register(
    const std::string& type_name,
    std::function<std::unique_ptr<Subgraph>()> factory) {
  local_factories_.Register(type_name, factory);
}

// TODO: Remove this convenience function.
void GraphRegistry::Register(const std::string& type_name,
                             const CalculatorGraphConfig& config) {
  local_factories_.Register(type_name, [config] {
    auto result = absl::make_unique<ProtoSubgraph>(config);
    return std::unique_ptr<Subgraph>(result.release());
  });
}

// TODO: Remove this convenience function.
void GraphRegistry::Register(const std::string& type_name,
                             const CalculatorGraphTemplate& templ) {
  local_factories_.Register(type_name, [templ] {
    auto result = absl::make_unique<TemplateSubgraph>(templ);
    return std::unique_ptr<Subgraph>(result.release());
  });
}

bool GraphRegistry::IsRegistered(const std::string& ns,
                                 const std::string& type_name) const {
  return local_factories_.IsRegistered(ns, type_name) ||
         global_factories_->IsRegistered(ns, type_name);
}

absl::StatusOr<CalculatorGraphConfig> GraphRegistry::CreateByName(
    absl::string_view ns, absl::string_view type_name,
    SubgraphContext* context) const {
  absl::StatusOr<std::unique_ptr<Subgraph>> maker =
      local_factories_.IsRegistered(ns, type_name)
          ? local_factories_.Invoke(ns, type_name)
          : global_factories_->Invoke(ns, type_name);
  MP_RETURN_IF_ERROR(maker.status());
  if (context != nullptr) {
    return maker.value()->GetConfig(context);
  }
  SubgraphContext default_context;
  return maker.value()->GetConfig(&default_context);
}

}  // namespace mediapipe
