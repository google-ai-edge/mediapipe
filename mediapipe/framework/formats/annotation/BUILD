#
# Copyright 2018 The MediaPipe Authors.
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
#

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

mediapipe_proto_library(
    name = "locus_proto",
    srcs = ["locus.proto"],
    portable_deps = ["//mediapipe/framework/formats/annotation:rasterization_cc_proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework/formats/annotation:rasterization_proto"],
)

mediapipe_proto_library(
    name = "rasterization_proto",
    srcs = ["rasterization.proto"],
    visibility = ["//visibility:public"],
)

# Expose the proto source files for building mediapipe AAR.
filegroup(
    name = "protos_src",
    srcs = glob(["*.proto"]),
    visibility = ["//mediapipe:__subpackages__"],
)
