# Copyright 2022 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
    name = "optional_dependencies",
    srcs = [
        "optional_dependencies.py",
    ],
)

py_library(
    name = "base_options",
    srcs = ["base_options.py"],
    visibility = [
        "//mediapipe/model_maker:__subpackages__",
        "//mediapipe/tasks:users",
    ],
    deps = [
        ":optional_dependencies",
        "//mediapipe/calculators/tensor:inference_calculator_py_pb2",
        "//mediapipe/tasks/cc/core/proto:acceleration_py_pb2",
        "//mediapipe/tasks/cc/core/proto:base_options_py_pb2",
        "//mediapipe/tasks/cc/core/proto:external_file_py_pb2",
    ],
)

py_library(
    name = "task_info",
    srcs = ["task_info.py"],
    deps = [
        ":optional_dependencies",
        "//mediapipe/calculators/core:flow_limiter_calculator_py_pb2",
        "//mediapipe/framework:calculator_options_py_pb2",
        "//mediapipe/framework:calculator_py_pb2",
        "@com_google_protobuf//:protobuf_python",
    ],
)
