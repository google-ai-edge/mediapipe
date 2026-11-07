# Copyright 2022 The MediaPipe Authors.
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

load("@rules_cc//cc:objc_library.bzl", "objc_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

exports_files(glob([
    "sources/*.h",
    "sources/*.swift",
]))

objc_library(
    name = "MPPBaseOptions",
    srcs = ["sources/MPPBaseOptions.m"],
    hdrs = ["sources/MPPBaseOptions.h"],
    deps = ["//third_party/apple_frameworks:Foundation"],
)

objc_library(
    name = "MPPTaskOptions",
    srcs = ["sources/MPPTaskOptions.m"],
    hdrs = ["sources/MPPTaskOptions.h"],
    deps = [
        ":MPPBaseOptions",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPTaskResult",
    srcs = ["sources/MPPTaskResult.m"],
    hdrs = ["sources/MPPTaskResult.h"],
    deps = ["//third_party/apple_frameworks:Foundation"],
)

objc_library(
    name = "MPPTaskPart",
    srcs = ["sources/MPPTaskPart.m"],
    hdrs = ["sources/MPPTaskPart.h"],
    deps = ["//third_party/apple_frameworks:Foundation"],
)

objc_library(
    name = "MPPCoreRunningMode",
    hdrs = ["sources/MPPCoreRunningMode.h"],
    deps = [
        "//third_party/apple_frameworks:Foundation",
    ],
)
