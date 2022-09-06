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

#include "mediapipe/tasks/cc/vision/image_classification/image_classifier.h"

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/containers/classifications.pb.h"
#include "mediapipe/tasks/cc/core/base_task_api.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/core/task_api_factory.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/vision/image_classification/image_classifier_options.pb.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/kernels/register.h"

namespace mediapipe {
namespace tasks {
namespace vision {

namespace {

constexpr char kImageStreamName[] = "image_in";
constexpr char kImageTag[] = "IMAGE";
constexpr char kClassificationResultStreamName[] = "classification_result_out";
constexpr char kClassificationResultTag[] = "CLASSIFICATION_RESULT";
constexpr char kSubgraphTypeName[] =
    "mediapipe.tasks.vision.ImageClassifierGraph";

// Creates a MediaPipe graph config that only contains a single subgraph node of
// "mediapipe.tasks.vision.ImageClassifierGraph".
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<ImageClassifierOptions> options) {
  api2::builder::Graph graph;
  auto& subgraph = graph.AddNode(kSubgraphTypeName);
  subgraph.GetOptions<ImageClassifierOptions>().Swap(options.get());
  graph.In(kImageTag).SetName(kImageStreamName) >> subgraph.In(kImageTag);
  subgraph.Out(kClassificationResultTag)
          .SetName(kClassificationResultStreamName) >>
      graph.Out(kClassificationResultTag);
  return graph.GetConfig();
}

}  // namespace

absl::StatusOr<std::unique_ptr<ImageClassifier>> ImageClassifier::Create(
    std::unique_ptr<ImageClassifierOptions> options,
    std::unique_ptr<tflite::OpResolver> resolver) {
  return core::TaskApiFactory::Create<ImageClassifier, ImageClassifierOptions>(
      CreateGraphConfig(std::move(options)), std::move(resolver));
}

absl::StatusOr<ClassificationResult> ImageClassifier::Classify(Image image) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "GPU input images are currently not supported.",
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  ASSIGN_OR_RETURN(auto output_packets,
                   runner_->Process({{kImageStreamName,
                                      MakePacket<Image>(std::move(image))}}));
  return output_packets[kClassificationResultStreamName]
      .Get<ClassificationResult>();
}

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
