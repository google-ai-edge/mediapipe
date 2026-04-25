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

#ifndef MEDIAPIPE_TASKS_CC_VISION_YOLO_OBJECT_DETECTOR_GRAPH_UTILS_H_
#define MEDIAPIPE_TASKS_CC_VISION_YOLO_OBJECT_DETECTOR_GRAPH_UTILS_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"

namespace mediapipe::tasks::vision::yolo_object_detector {

enum class YoloDecodeMode { kAuto, kUltralytics, kEndToEnd };

// Reads TFLite model's first input tensor H×W from flatbuffer (no inference).
// Returns {width, height} for the NHWC input tensor (shape[2], shape[1]).
absl::StatusOr<std::pair<int, int>> ExtractModelInputShape(
    const std::string& model_path);

// Infer YOLO decode mode from output tensor shape dims.
// Squeezes leading singleton dims, then checks if any remaining dim == 6.
// dim==6 → kEndToEnd; otherwise → kUltralytics.
YoloDecodeMode InferDecodeMode(std::vector<int> dims);

}  // namespace mediapipe::tasks::vision::yolo_object_detector

#endif  // MEDIAPIPE_TASKS_CC_VISION_YOLO_OBJECT_DETECTOR_GRAPH_UTILS_H_
