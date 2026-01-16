/* Copyright 2022 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/core/model_task_graph.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/core/model_asset_bundle_resources.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_resources_cache.h"
#include "mediapipe/tasks/cc/core/proto/acceleration.pb.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/core/proto/model_resources_calculator.pb.h"

namespace mediapipe {
namespace tasks {
namespace core {
namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::SideInput;
using ::mediapipe::api2::SideOutput;
using ::mediapipe::api2::builder::GenericNode;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::tasks::core::proto::Acceleration;
using ::mediapipe::tasks::core::proto::InferenceSubgraphOptions;
using ::mediapipe::tasks::core::proto::ModelResourcesCalculatorOptions;

constexpr char kMetadataExtractorTag[] = "METADATA_EXTRACTOR";
constexpr char kModelTag[] = "MODEL";
constexpr char kOpResolverTag[] = "OP_RESOLVER";
constexpr char kTensorsTag[] = "TENSORS";

std::string CreateModelResourcesTag(const CalculatorGraphConfig::Node& node) {
  std::vector<std::string> names = absl::StrSplit(node.name(), "__");
  std::string node_type = node.calculator();
  std::replace(node_type.begin(), node_type.end(), '.', '_');
  absl::AsciiStrToLower(&node_type);
  return absl::StrFormat("%s_%s_model_resources",
                         names.back().empty() ? "unnamed" : names.back(),
                         node_type);
}

std::string CreateModelAssetBundleResourcesTag(
    const CalculatorGraphConfig::Node& node) {
  std::vector<std::string> names = absl::StrSplit(node.name(), "__");
  std::string node_type = node.calculator();
  std::replace(node_type.begin(), node_type.end(), '.', '_');
  absl::AsciiStrToLower(&node_type);
  return absl::StrFormat("%s_%s_model_asset_bundle_resources",
                         names.back().empty() ? "unnamed" : names.back(),
                         node_type);
}

}  // namespace

// Defines the mediapipe task inference unit as a MediaPipe subgraph that
// contains a ModelResourcesCalculator (for model resources management) and
// an InferenceCalculator (for single model inference).
class InferenceSubgraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    auto* subgraph_options = sc->MutableOptions<InferenceSubgraphOptions>();
    MP_ASSIGN_OR_RETURN(auto inference_delegate,
                        DecideInferenceSettings(*subgraph_options));
    Graph graph;
    auto& model_resources_node = graph.AddNode("ModelResourcesCalculator");
    auto& model_resources_opts =
        model_resources_node.GetOptions<ModelResourcesCalculatorOptions>();
    if (!subgraph_options->model_resources_tag().empty()) {
      model_resources_opts.set_model_resources_tag(
          subgraph_options->model_resources_tag());
    } else {
      model_resources_opts.mutable_model_file()->Swap(
          subgraph_options->mutable_base_options()->mutable_model_asset());
    }
    model_resources_node.SideOut(kMetadataExtractorTag) >>
        graph.SideOut(kMetadataExtractorTag);

    auto& inference_node = graph.AddNode("InferenceCalculator");
    inference_node.GetOptions<mediapipe::InferenceCalculatorOptions>()
        .mutable_delegate()
        ->CopyFrom(inference_delegate);
    model_resources_node.SideOut(kModelTag) >> inference_node.SideIn(kModelTag);
    model_resources_node.SideOut(kOpResolverTag) >>
        inference_node.SideIn(kOpResolverTag);
    graph.In(kTensorsTag) >> inference_node.In(kTensorsTag);
    inference_node.Out(kTensorsTag) >> graph.Out(kTensorsTag);
    return graph.GetConfig();
  }

 private:
  absl::StatusOr<mediapipe::InferenceCalculatorOptions::Delegate>
  DecideInferenceSettings(const InferenceSubgraphOptions& options) {
    // TODO: Fills in the inference delegate options based on the
    // model, acceleration settings, and device hardware info.
    mediapipe::InferenceCalculatorOptions::Delegate delegate;
    const Acceleration& acceleration = options.base_options().acceleration();
    switch (acceleration.delegate_case()) {
      case Acceleration::kXnnpack:
        *delegate.mutable_xnnpack() = acceleration.xnnpack();
        break;
      case Acceleration::kGpu:
        *delegate.mutable_gpu() = acceleration.gpu();
        break;
      case Acceleration::kNnapi:
        *delegate.mutable_nnapi() = acceleration.nnapi();
        break;
      case Acceleration::kTflite:
        *delegate.mutable_tflite() = acceleration.tflite();
        break;
      case Acceleration::DELEGATE_NOT_SET:
        // Default inference calculator setting.
        break;
    }
    return delegate;
  }
};

REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tasks::core::InferenceSubgraph)

absl::StatusOr<CalculatorGraphConfig> ModelTaskGraph::GetConfig(
    SubgraphContext* sc) {
  return CreateStatusWithPayload(
      absl::StatusCode::kUnimplemented,
      "The task graph is not implemented. Please override the GetConfig() "
      "method in the subclass.",
      MediaPipeTasksStatus::kTaskGraphNotImplementedError);
}

absl::StatusOr<const ModelResources*> ModelTaskGraph::CreateModelResources(
    SubgraphContext* sc, std::unique_ptr<proto::ExternalFile> external_file,
    const std::string tag_suffix) {
  auto model_resources_cache_service = sc->Service(kModelResourcesCacheService);
  if (!model_resources_cache_service.IsAvailable()) {
    MP_ASSIGN_OR_RETURN(auto local_model_resource,
                        ModelResources::Create("", std::move(external_file)));
    ABSL_LOG(WARNING)
        << "A local ModelResources object is created. Please consider using "
           "ModelResourcesCacheService to cache the created ModelResources "
           "object in the CalculatorGraph.";
    local_model_resources_.push_back(std::move(local_model_resource));
    return local_model_resources_.back().get();
  }
  MP_ASSIGN_OR_RETURN(
      auto op_resolver_packet,
      model_resources_cache_service.GetObject().GetGraphOpResolverPacket());
  const std::string tag =
      absl::StrCat(CreateModelResourcesTag(sc->OriginalNode()), tag_suffix);
  MP_ASSIGN_OR_RETURN(auto model_resources,
                      ModelResources::Create(tag, std::move(external_file),
                                             op_resolver_packet));
  MP_RETURN_IF_ERROR(
      model_resources_cache_service.GetObject().AddModelResources(
          std::move(model_resources)));
  return model_resources_cache_service.GetObject().GetModelResources(tag);
}

absl::StatusOr<const ModelResources*> ModelTaskGraph::GetOrCreateModelResources(
    SubgraphContext* sc, std::unique_ptr<proto::ExternalFile> external_file,
    std::string tag_suffix) {
  auto model_resources_cache_service = sc->Service(kModelResourcesCacheService);
  if (model_resources_cache_service.IsAvailable()) {
    std::string tag =
        absl::StrCat(CreateModelResourcesTag(sc->OriginalNode()), tag_suffix);
    if (model_resources_cache_service.GetObject().Exists(tag)) {
      return model_resources_cache_service.GetObject().GetModelResources(tag);
    }
  }
  return ModelTaskGraph::CreateModelResources(sc, std::move(external_file),
                                              tag_suffix);
}

absl::StatusOr<const ModelAssetBundleResources*>
ModelTaskGraph::CreateModelAssetBundleResources(
    SubgraphContext* sc, std::unique_ptr<proto::ExternalFile> external_file,
    std::string tag_suffix) {
  auto model_resources_cache_service = sc->Service(kModelResourcesCacheService);
  bool has_file_pointer_meta = external_file->has_file_pointer_meta();
  // if external file is set by file pointer, no need to add the model asset
  // bundle resources into the model resources service since the memory is
  // not owned by this model asset bundle resources.
  if (!model_resources_cache_service.IsAvailable() || has_file_pointer_meta) {
    MP_ASSIGN_OR_RETURN(
        auto local_model_asset_bundle_resource,
        ModelAssetBundleResources::Create("", std::move(external_file)));
    if (!has_file_pointer_meta) {
      ABSL_LOG(WARNING)
          << "A local ModelResources object is created. Please consider using "
             "ModelResourcesCacheService to cache the created ModelResources "
             "object in the CalculatorGraph.";
    }
    local_model_asset_bundle_resources_.push_back(
        std::move(local_model_asset_bundle_resource));
    return local_model_asset_bundle_resources_.back().get();
  }
  const std::string tag = absl::StrCat(
      CreateModelAssetBundleResourcesTag(sc->OriginalNode()), tag_suffix);
  MP_ASSIGN_OR_RETURN(
      auto model_bundle_resources,
      ModelAssetBundleResources::Create(tag, std::move(external_file)));
  MP_RETURN_IF_ERROR(
      model_resources_cache_service.GetObject().AddModelAssetBundleResources(
          std::move(model_bundle_resources)));
  return model_resources_cache_service.GetObject().GetModelAssetBundleResources(
      tag);
}

GenericNode& ModelTaskGraph::AddInference(
    const ModelResources& model_resources,
    const proto::Acceleration& acceleration, Graph& graph) const {
  auto& inference_subgraph =
      graph.AddNode("mediapipe.tasks.core.InferenceSubgraph");
  auto& inference_subgraph_opts =
      inference_subgraph.GetOptions<InferenceSubgraphOptions>();
  inference_subgraph_opts.mutable_base_options()
      ->mutable_acceleration()
      ->CopyFrom(acceleration);
  // When the model resources tag is available, the ModelResourcesCalculator
  // will retrieve the cached model resources from the graph service by tag.
  // Otherwise, provides the external file and asks the
  // ModelResourcesCalculator to create a local model resources in its
  // Calculator::Open().
  if (!model_resources.GetTag().empty()) {
    inference_subgraph_opts.set_model_resources_tag(model_resources.GetTag());
  } else {
    inference_subgraph_opts.mutable_base_options()
        ->mutable_model_asset()
        ->CopyFrom(model_resources.GetModelFile());
  }
  return inference_subgraph;
}

}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
