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

package(default_visibility = ["//mediapipe/tasks:internal"])

cc_library(
    name = "logging_factory",
    srcs = ["logging_factory.cc"],
    hdrs = ["logging_factory.h"],
    deps = [
        "//mediapipe/tasks/cc/core/logging:tasks_dummy_logger",
        "//mediapipe/tasks/cc/core/logging:tasks_logger",
    ],
)

cc_library(
    name = "logging_factory_hdr",
    hdrs = ["logging_factory.h"],
    deps = [
        "//mediapipe/tasks/cc/core/logging:tasks_logger",
    ],
)
