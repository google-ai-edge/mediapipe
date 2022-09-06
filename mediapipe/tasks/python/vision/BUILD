# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Placeholder for internal Python strict library compatibility macro.

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

py_library(
    name = "object_detector",
    srcs = [
        "object_detector.py",
    ],
    deps = [
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/python:packet_creator",
        "//mediapipe/python:packet_getter",
        "//mediapipe/tasks/cc/vision/object_detector/proto:object_detector_options_py_pb2",
        "//mediapipe/tasks/python/components/containers:detections",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:task_info",
        "//mediapipe/tasks/python/vision/core:base_vision_task_api",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
    ],
)
