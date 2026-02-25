# Copyright 2025 The MediaPipe Authors.
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
    "//mediapipe/tasks/cc/core:__subpackages__",
])

cc_library(
    name = "logging_client",
    hdrs = ["logging_client.h"],
    deps = [
        "//mediapipe/util/analytics:mediapipe_log_extension_cc_proto",
    ],
)

cc_library(
    name = "tasks_dummy_logger",
    hdrs = ["tasks_dummy_logger.h"],
    deps = [
        ":logging_client",
        ":tasks_logger",
        "//mediapipe/framework:timestamp",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "tasks_logger",
    hdrs = ["tasks_logger.h"],
    deps = ["//mediapipe/framework:timestamp"],
)
