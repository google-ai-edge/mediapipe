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
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = [
    "//mediapipe/tasks/cc:__subpackages__",
])

cc_library(
    name = "tasks_dummy_logger",
    hdrs = ["tasks_dummy_logger.h"],
    deps = [":tasks_logger"],
)

cc_library(
    name = "tasks_logger",
    hdrs = ["tasks_logger.h"],
)
