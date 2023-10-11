# Copyright 2023 The MediaPipe Authors.
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

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_proto_library(
    name = "env_generator_calculator_proto",
    srcs = ["env_generator_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/tasks/cc/vision/face_geometry/proto:environment_proto",
    ],
)

cc_library(
    name = "env_generator_calculator",
    srcs = ["env_generator_calculator.cc"],
    deps = [
        ":env_generator_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc/vision/face_geometry/libs:validation_utils",
        "//mediapipe/tasks/cc/vision/face_geometry/proto:environment_cc_proto",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "geometry_pipeline_calculator_proto",
    srcs = ["geometry_pipeline_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/tasks/cc/core/proto:external_file_proto",
    ],
)

cc_library(
    name = "geometry_pipeline_calculator",
    srcs = ["geometry_pipeline_calculator.cc"],
    deps = [
        ":geometry_pipeline_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/core:external_file_handler",
        "//mediapipe/tasks/cc/core/proto:external_file_cc_proto",
        "//mediapipe/tasks/cc/vision/face_geometry/libs:geometry_pipeline",
        "//mediapipe/tasks/cc/vision/face_geometry/libs:validation_utils",
        "//mediapipe/tasks/cc/vision/face_geometry/proto:environment_cc_proto",
        "//mediapipe/tasks/cc/vision/face_geometry/proto:face_geometry_cc_proto",
        "//mediapipe/tasks/cc/vision/face_geometry/proto:geometry_pipeline_metadata_cc_proto",
        "//mediapipe/util:resource_util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = 1,
)
