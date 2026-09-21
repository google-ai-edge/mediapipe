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

AUDIO_EMBEDDER_DEPS = [
    "@com_google_absl//absl/log:absl_log",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:statusor",
    "@eigen//:eigen3",
    "//mediapipe/framework/formats:matrix",
    "//mediapipe/framework/port:status",
    "@com_google_absl//absl/log:absl_check",
    "//mediapipe/tasks/c/audio/core:common",
    "//mediapipe/tasks/c/audio/core:running_mode_converter",
    "//mediapipe/tasks/c/components/containers:embedding_result",
    "//mediapipe/tasks/c/components/containers:embedding_result_converter",
    "//mediapipe/tasks/c/components/processors:embedder_options",
    "//mediapipe/tasks/c/components/processors:embedder_options_converter",
    "//mediapipe/tasks/c/core:base_options",
    "//mediapipe/tasks/c/core:base_options_converter",
    "//mediapipe/tasks/c/core:mp_status",
    "//mediapipe/tasks/c/core:mp_status_converter",
    "//mediapipe/tasks/cc/audio/audio_embedder",
    "//mediapipe/tasks/cc/components/containers:embedding_result",
    "//mediapipe/tasks/cc/components/utils:cosine_similarity",
]

cc_library(
    name = "audio_embedder_lib",
    srcs = ["audio_embedder.cc"],
    hdrs = ["audio_embedder.h"],
    visibility = ["//visibility:public"],
    deps = AUDIO_EMBEDDER_DEPS,
)

cc_library(
    name = "audio_embedder_c_lib",
    srcs = ["audio_embedder.cc"],
    hdrs = ["audio_embedder.h"],
    visibility = ["//mediapipe/tasks/c:__subpackages__"],
    deps = AUDIO_EMBEDDER_DEPS,
    alwayslink = 1,
)

# TODO: mediapipe/tasks/cc/audio/utils:test_utils does not compile in the OSS build
