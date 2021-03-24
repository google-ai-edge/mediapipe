# Copyright 2020 The MediaPipe Authors.
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

package(default_visibility = ["//visibility:public"])

mediapipe_proto_library(
    name = "environment_proto",
    srcs = ["environment.proto"],
)

mediapipe_proto_library(
    name = "face_geometry_proto",
    srcs = ["face_geometry.proto"],
    deps = [
        ":mesh_3d_proto",
        "//mediapipe/framework/formats:matrix_data_proto",
    ],
)

mediapipe_proto_library(
    name = "geometry_pipeline_metadata_proto",
    srcs = ["geometry_pipeline_metadata.proto"],
    deps = [
        ":mesh_3d_proto",
    ],
)

mediapipe_proto_library(
    name = "mesh_3d_proto",
    srcs = ["mesh_3d.proto"],
)
