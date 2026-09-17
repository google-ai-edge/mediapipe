# Copyright 2026 The MediaPipe Authors.
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

licenses(["notice"])

UNIVERSAL_EMBEDDER_DEPS = [
    "@com_google_absl//absl/log:absl_check",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:status_macros",
    "@com_google_absl//absl/status:statusor",
    "@com_google_absl//absl/strings",
    "//mediapipe/framework/formats:image",
    "//mediapipe/framework/port:status",
    "//mediapipe/tasks/c/audio/core:common",
    "//mediapipe/tasks/c/components/containers:embedding_result",
    "//mediapipe/tasks/c/components/containers:embedding_result_converter",
    "//mediapipe/tasks/c/core:base_options",
    "//mediapipe/tasks/c/core:base_options_converter",
    "//mediapipe/tasks/c/core:common",
    "//mediapipe/tasks/c/core:mp_status",
    "//mediapipe/tasks/c/core:mp_status_converter",
    "//mediapipe/tasks/c/vision/core:image",
    "//mediapipe/tasks/c/vision/core:image_frame_util",
    "//mediapipe/tasks/cc/components/containers:embedding_result",
    "//mediapipe/tasks/cc/retrieval/universal_embedder",
]

cc_library(
    name = "universal_embedder_lib",
    srcs = ["universal_embedder.cc"],
    hdrs = [
        "universal_embedder.h",
        "universal_embedder_internal.h",
    ],
    visibility = ["//visibility:public"],
    deps = UNIVERSAL_EMBEDDER_DEPS,
    alwayslink = True,
)
