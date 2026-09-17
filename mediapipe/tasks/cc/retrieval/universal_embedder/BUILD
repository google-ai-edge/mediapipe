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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "universal_embedder",
    srcs = ["universal_embedder.cc"],
    hdrs = ["universal_embedder.h"],
    deps = [
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc/components/containers:embedding_result",
        "//mediapipe/tasks/cc/components/utils:cosine_similarity",
        "//mediapipe/tasks/cc/core:base_options",
        "//mediapipe/tasks/cc/core:embedding_provider",
        "//mediapipe/tasks/cc/core:running_mode",
        "//mediapipe/tasks/cc/core:tasks_logging_factory",  # build_cleaner: keep
        "//mediapipe/tasks/cc/core/logging:tasks_logger",
        "//mediapipe/tasks/cc/core/logging/factory:logging_factory_hdr",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert_lm//runtime/core:embedding_engine_impl",
        "@litert_lm//runtime/engine:embedding_engine",
        "@litert_lm//runtime/engine:embedding_engine_settings",
        "@litert_lm//runtime/engine:io_types",
        "@litert_lm//runtime/executor:executor_settings_base",
        "@litert_lm//runtime/util:memory_mapped_file",
    ],
)
