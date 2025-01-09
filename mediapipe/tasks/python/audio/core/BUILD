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
    name = "audio_task_running_mode",
    srcs = ["audio_task_running_mode.py"],
)

py_library(
    name = "audio_record",
    srcs = ["audio_record.py"],
    deps = [
        "@mediapipe_pip_deps_numpy//:pkg",
    ],
)

py_library(
    name = "base_audio_task_api",
    srcs = [
        "base_audio_task_api.py",
    ],
    deps = [
        ":audio_record",
        ":audio_task_running_mode",
        "//mediapipe/framework:calculator_py_pb2",
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/python:packet_creator",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
)
