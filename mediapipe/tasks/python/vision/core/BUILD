# Copyright 2022 The MediaPipe Authors.
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

# Placeholder for internal Python strict library and test compatibility macro.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "vision_task_running_mode",
    srcs = ["vision_task_running_mode.py"],
)

py_library(
    name = "image_processing_options",
    srcs = ["image_processing_options.py"],
    deps = [
        "//mediapipe/tasks/python/components/containers:rect",
    ],
)

py_library(
    name = "base_vision_task_api",
    srcs = [
        "base_vision_task_api.py",
    ],
    deps = [
        ":image_processing_options",
        ":vision_task_running_mode",
        "//mediapipe/framework:calculator_py_pb2",
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/tasks/python/components/containers:rect",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
)
