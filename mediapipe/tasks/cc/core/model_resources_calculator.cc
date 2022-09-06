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

#include <memory>
#include <string>
#include <type_traits>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_resources_cache.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/proto/model_resources_calculator.pb.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "tensorflow/lite/core/api/op_resolver.h"

namespace mediapipe {
namespace tasks {
namespace core {

// A ModelResourceCalculator either takes an existing ModelResources from the
// ModelResourcesCacheService or creates a local ModelResources object from the
// external file proto provided in the calculator options. It then distributes
// the model-related resources (e.g., flatbuffer model, op resolver, and model
// metadata extractor), to other calculators (e.g., InferenceCalculator) in the
// mediapipe task graphs.
//
// Example config:
// node {
//   calculator: "ModelResourcesCalculator"
//   output_side_packet: "MODEL:model"
//   output_side_packet: "OP_RESOLVER:op_resolver"
//   output_side_packet: "METADATA_EXTRACTOR:metadata_extractor"
//   options {
//     [mediapipe.tasks.core.proto.ModelResourcesCalculatorOptions.ext] {
//       model_resources_tag: "unique_model_resources_tag"
//       model_file {file_name: "/path/to/model"}
//     }
//   }
// }
class ModelResourcesCalculator : public api2::Node {
 public:
  static constexpr api2::SideOutput<ModelResources::ModelPtr> kModel{"MODEL"};
  static constexpr api2::SideOutput<tflite::OpResolver>::Optional kOpResolver{
      "OP_RESOLVER"};
  static constexpr api2::SideOutput<metadata::ModelMetadataExtractor>::Optional
      kMetadataExtractor{"METADATA_EXTRACTOR"};

  MEDIAPIPE_NODE_INTERFACE(ModelResourcesCalculator, kModel, kOpResolver,
                           kMetadataExtractor);

  static absl::Status UpdateContract(mediapipe::CalculatorContract* cc) {
    const auto& options = cc->Options<proto::ModelResourcesCalculatorOptions>();
    RET_CHECK(options.has_model_resources_tag() || options.has_model_file())
        << "ModelResourcesCalculatorOptions must specify at least one of "
           "'model_resources_tag' or 'model_file'";
    if (options.has_model_resources_tag()) {
      RET_CHECK(!options.model_resources_tag().empty())
          << "'model_resources_tag' should not be empty.";
      cc->UseService(kModelResourcesCacheService);
    }
    if (options.has_model_file()) {
      RET_CHECK(options.model_file().has_file_content() ||
                options.model_file().has_file_descriptor_meta() ||
                options.model_file().has_file_name())
          << "'model_file' must specify at least one of "
             "'file_content', 'file_descriptor_meta', or 'file_name'";
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    const auto& options = cc->Options<proto::ModelResourcesCalculatorOptions>();
    const ModelResources* model_resources = nullptr;
    if (cc->Service(kModelResourcesCacheService).IsAvailable()) {
      const std::string& model_resources_tag = options.model_resources_tag();
      auto status_or_model_resources =
          cc->Service(kModelResourcesCacheService)
              .GetObject()
              .GetModelResources(model_resources_tag);
      if (status_or_model_resources.ok()) {
        model_resources = status_or_model_resources.value();
      }
    }
    // If the ModelResources isn't available through the
    // ModelResourcesCacheService, creates a local ModelResources from the
    // CalculatorOptions as a fallback.
    if (model_resources == nullptr) {
      if (!options.has_model_file()) {
        return absl::InvalidArgumentError(
            "ModelResources is not available through the MediaPipe "
            "ModelResourcesCacheService, and the CalculatorOptions has no "
            "'model_file' field to create a local ModelResources.");
      }
      ASSIGN_OR_RETURN(
          model_resources_,
          ModelResources::Create(
              "", std::make_unique<proto::ExternalFile>(options.model_file())));
      model_resources = model_resources_.get();
    }
    kModel(cc).Set(model_resources->GetModelPacket());
    kOpResolver(cc).Set(model_resources->GetOpResolverPacket());
    kMetadataExtractor(cc).Set(model_resources->GetMetadataExtractorPacket());
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }

 private:
  std::unique_ptr<ModelResources> model_resources_;
};

MEDIAPIPE_REGISTER_NODE(ModelResourcesCalculator);

}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
