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

#include "mediapipe/tasks/cc/core/model_resources_cache.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/core/model_asset_bundle_resources.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "tensorflow/lite/core/api/op_resolver.h"

namespace mediapipe {
namespace tasks {
namespace core {

ModelResourcesCache::ModelResourcesCache(
    std::unique_ptr<tflite::OpResolver> graph_op_resolver) {
  if (graph_op_resolver) {
    graph_op_resolver_packet_ =
        api2::PacketAdopting<tflite::OpResolver>(std::move(graph_op_resolver));
  }
}

bool ModelResourcesCache::Exists(const std::string& tag) const {
  return model_resources_collection_.contains(tag);
}

bool ModelResourcesCache::ModelAssetBundleExists(const std::string& tag) const {
  return model_asset_bundle_resources_collection_.contains(tag);
}

absl::Status ModelResourcesCache::AddModelResources(
    std::unique_ptr<ModelResources> model_resources) {
  if (model_resources == nullptr) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument, "ModelResources object is null.",
        MediaPipeTasksStatus::kRunnerModelResourcesCacheServiceError);
  }
  const std::string& tag = model_resources->GetTag();
  if (tag.empty()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "ModelResources must have a non-empty tag.",
        MediaPipeTasksStatus::kRunnerModelResourcesCacheServiceError);
  }
  if (Exists(tag)) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::Substitute("ModelResources with tag \"$0\" already exists.", tag),
        MediaPipeTasksStatus::kRunnerModelResourcesCacheServiceError);
  }
  model_resources_collection_.emplace(tag, std::move(model_resources));
  return absl::OkStatus();
}

absl::Status ModelResourcesCache::AddModelResourcesCollection(
    std::vector<std::unique_ptr<ModelResources>>& model_resources_collection) {
  for (auto& model_resources : model_resources_collection) {
    MP_RETURN_IF_ERROR(AddModelResources(std::move(model_resources)));
  }
  return absl::OkStatus();
}

absl::StatusOr<const ModelResources*> ModelResourcesCache::GetModelResources(
    const std::string& tag) const {
  if (tag.empty()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "ModelResources must be retrieved with a non-empty tag.",
        MediaPipeTasksStatus::kRunnerModelResourcesCacheServiceError);
  }
  if (!Exists(tag)) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::Substitute("ModelResources with tag \"$0\" does not exist.", tag),
        MediaPipeTasksStatus::kRunnerModelResourcesCacheServiceError);
  }
  return model_resources_collection_.at(tag).get();
}

absl::Status ModelResourcesCache::AddModelAssetBundleResources(
    std::unique_ptr<ModelAssetBundleResources> model_asset_bundle_resources) {
  if (model_asset_bundle_resources == nullptr) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "ModelAssetBundleResources object is null.",
        MediaPipeTasksStatus::kRunnerModelResourcesCacheServiceError);
  }
  const std::string& tag = model_asset_bundle_resources->GetTag();
  if (tag.empty()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "ModelAssetBundleResources must have a non-empty tag.",
        MediaPipeTasksStatus::kRunnerModelResourcesCacheServiceError);
  }
  if (ModelAssetBundleExists(tag)) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::Substitute(
            "ModelAssetBundleResources with tag \"$0\" already exists.", tag),
        MediaPipeTasksStatus::kRunnerModelResourcesCacheServiceError);
  }
  model_asset_bundle_resources_collection_.emplace(
      tag, std::move(model_asset_bundle_resources));
  return absl::OkStatus();
}

absl::Status ModelResourcesCache::AddModelAssetBundleResourcesCollection(
    std::vector<std::unique_ptr<ModelAssetBundleResources>>&
        model_asset_bundle_resources_collection) {
  for (auto& model_bundle_resources : model_asset_bundle_resources_collection) {
    MP_RETURN_IF_ERROR(
        AddModelAssetBundleResources(std::move(model_bundle_resources)));
  }
  return absl::OkStatus();
}

absl::StatusOr<const ModelAssetBundleResources*>
ModelResourcesCache::GetModelAssetBundleResources(
    const std::string& tag) const {
  if (tag.empty()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "ModelAssetBundleResources must be retrieved with a non-empty tag.",
        MediaPipeTasksStatus::kRunnerModelResourcesCacheServiceError);
  }
  if (!ModelAssetBundleExists(tag)) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::Substitute(
            "ModelAssetBundleResources with tag \"$0\" does not exist.", tag),
        MediaPipeTasksStatus::kRunnerModelResourcesCacheServiceError);
  }
  return model_asset_bundle_resources_collection_.at(tag).get();
}

absl::StatusOr<api2::Packet<tflite::OpResolver>>
ModelResourcesCache::GetGraphOpResolverPacket() const {
  if (graph_op_resolver_packet_.IsEmpty()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "The graph op resolver is not set in ModelResourcesCache.",
        MediaPipeTasksStatus::kRunnerModelResourcesCacheServiceError);
  }
  return graph_op_resolver_packet_;
}

}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
