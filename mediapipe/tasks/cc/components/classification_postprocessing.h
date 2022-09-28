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

#ifndef MEDIAPIPE_TASKS_CC_COMPONENTS_CLASSIFICATION_POSTPROCESSING_H_
#define MEDIAPIPE_TASKS_CC_COMPONENTS_CLASSIFICATION_POSTPROCESSING_H_

#include "absl/status/status.h"
#include "mediapipe/tasks/cc/components/classification_postprocessing_options.pb.h"
#include "mediapipe/tasks/cc/components/proto/classifier_options.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"

namespace mediapipe {
namespace tasks {
namespace components {

// Configures a ClassificationPostprocessing subgraph using the provided model
// resources and ClassifierOptions.
// - Accepts CPU input tensors.
//
// Example usage:
//
//   auto& postprocessing =
//       graph.AddNode("mediapipe.tasks.components.ClassificationPostprocessingSubgraph");
//   MP_RETURN_IF_ERROR(ConfigureClassificationPostprocessing(
//       model_resources,
//       classifier_options,
//       &preprocessing.GetOptions<ClassificationPostprocessingOptions>()));
//
// The resulting ClassificationPostprocessing subgraph has the following I/O:
// Inputs:
//   TENSORS - std::vector<Tensor>
//     The output tensors of an InferenceCalculator.
//   TIMESTAMPS - std::vector<Timestamp> @Optional
//     The collection of timestamps that a single ClassificationResult should
//     aggregate. This is mostly useful for classifiers working on time series,
//     e.g. audio or video classification.
// Outputs:
//   CLASSIFICATION_RESULT - ClassificationResult
//     The output aggregated classification results.
absl::Status ConfigureClassificationPostprocessing(
    const tasks::core::ModelResources& model_resources,
    const tasks::components::proto::ClassifierOptions& classifier_options,
    ClassificationPostprocessingOptions* options);

}  // namespace components
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_COMPONENTS_CLASSIFICATION_POSTPROCESSING_H_
