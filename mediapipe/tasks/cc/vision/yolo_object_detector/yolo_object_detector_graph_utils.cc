/* Copyright 2026 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/vision/yolo_object_detector/yolo_object_detector_graph_utils.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe::tasks::vision::yolo_object_detector {

namespace {
constexpr int kEndToEndFeatureDim = 6;
}

absl::StatusOr<std::pair<int, int>> ExtractModelInputShape(
    const std::string& model_path) {
  std::string model_buffer;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(model_path, &model_buffer));

  const tflite::Model* model = tflite::GetModel(model_buffer.data());
  if (!model || !model->subgraphs() || model->subgraphs()->size() == 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot parse TFLite model at: %s", model_path));
  }
  const auto* subgraph = model->subgraphs()->Get(0);
  if (!subgraph->inputs() || subgraph->inputs()->size() == 0) {
    return absl::InvalidArgumentError("Model has no inputs.");
  }
  const int input_idx = subgraph->inputs()->Get(0);
  if (!subgraph->tensors()) {
    return absl::InvalidArgumentError("Model subgraph has no tensors table.");
  }
  const auto* tensor = subgraph->tensors()->Get(input_idx);
  if (!tensor || !tensor->shape() || tensor->shape()->size() < 4) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected 4-D NHWC input tensor, got rank %d",
        tensor->shape() ? tensor->shape()->size() : 0));
  }
  // NHWC: [N, H, W, C]
  const int h = tensor->shape()->Get(1);
  const int w = tensor->shape()->Get(2);
  return std::make_pair(w, h);
}

YoloDecodeMode InferDecodeMode(std::vector<int> dims) {
  // Squeeze leading singleton dims.
  while (dims.size() > 2 && dims.front() == 1) {
    dims.erase(dims.begin());
  }
  for (int d : dims) {
    if (d == kEndToEndFeatureDim) return YoloDecodeMode::kEndToEnd;
  }
  return YoloDecodeMode::kUltralytics;
}

}  // namespace mediapipe::tasks::vision::yolo_object_detector
