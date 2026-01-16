/* Copyright 2025 The MediaPipe Authors.

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

#ifndef MEDIAPIPE_TASKS_CC_VISION_INTERACTIVE_SEGMENTER_INTERACTIVE_SEGMENTER_GRAPH_H_
#define MEDIAPIPE_TASKS_CC_VISION_INTERACTIVE_SEGMENTER_INTERACTIVE_SEGMENTER_GRAPH_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace interactive_segmenter {

class InteractiveSegmenterGraph : public core::ModelTaskGraph {
 public:
  // Returns the graph config to use for one instantiation of the model task
  // graph. Must be overridden by subclasses in which the graph authors define
  // the concrete task graphs based on user settings and model metadata.
  absl::StatusOr<mediapipe::CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override;
};

}  // namespace interactive_segmenter
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_INTERACTIVE_SEGMENTER_INTERACTIVE_SEGMENTER_GRAPH_H_
