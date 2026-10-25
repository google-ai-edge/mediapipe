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

package(default_visibility = [
    "//mediapipe/tasks:__subpackages__",
])

licenses(["notice"])

cc_library(
    name = "text_summarizer",
    srcs = ["text_summarizer.cc"],
    hdrs = ["text_summarizer.h"],
    deps = [
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc/core:base_options",
        "//mediapipe/tasks/cc/core:running_mode",
        "//mediapipe/tasks/cc/core:tasks_logging_factory",  # build_cleaner: keep
        "//mediapipe/tasks/cc/core/logging:tasks_logger",
        "//mediapipe/tasks/cc/core/logging/factory:logging_factory_hdr",
        "//mediapipe/tasks/cc/text/utils:genai_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@litert_lm//c:engine_cpu",
    ],
)
