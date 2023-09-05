#
# Copyright 2019 The MediaPipe Authors.
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

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")

licenses(["notice"])

package(default_visibility = ["//mediapipe/calculators/video:__subpackages__"])

proto_library(
    name = "flow_quantizer_model_proto",
    srcs = ["flow_quantizer_model.proto"],
    visibility = ["//visibility:public"],
)

mediapipe_cc_proto_library(
    name = "flow_quantizer_model_cc_proto",
    srcs = ["flow_quantizer_model.proto"],
    visibility = ["//visibility:public"],
    deps = [":flow_quantizer_model_proto"],
)

cc_library(
    name = "flow_quantizer_model",
    srcs = ["flow_quantizer_model.cc"],
    hdrs = ["flow_quantizer_model.h"],
    deps = [
        "//mediapipe/calculators/video/tool:flow_quantizer_model_cc_proto",
        "//mediapipe/framework:type_map",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats/motion:optical_flow_field",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/tool:status_util",
        "@com_google_absl//absl/log:absl_check",
    ],
    alwayslink = 1,
)
