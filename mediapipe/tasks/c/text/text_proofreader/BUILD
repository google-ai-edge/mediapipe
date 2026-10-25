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
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

licenses(["notice"])

TEXT_PROOFREADER_DEPS = [
    "@com_google_absl//absl/log:absl_check",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:status_macros",
    "@com_google_absl//absl/status:statusor",
    "//mediapipe/framework/port:status",
    "//mediapipe/tasks/c/core:base_options",
    "//mediapipe/tasks/c/core:base_options_converter",
    "//mediapipe/tasks/c/core:mp_status",
    "//mediapipe/tasks/c/core:mp_status_converter",
    "//mediapipe/tasks/cc/text/text_proofreader",
]

cc_library(
    name = "text_proofreader_lib",
    srcs = ["text_proofreader.cc"],
    hdrs = ["text_proofreader.h"],
    visibility = ["//visibility:public"],
    deps = TEXT_PROOFREADER_DEPS,
)

# Support for C bindings
cc_library(
    name = "text_proofreader_c_lib",
    srcs = ["text_proofreader.cc"],
    hdrs = ["text_proofreader.h"],
    visibility = ["//mediapipe/tasks/c:__subpackages__"],
    deps = TEXT_PROOFREADER_DEPS,
    alwayslink = 1,
)

# bazel build -c opt --linkopt -s --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c/text/text_proofreader:libtext_proofreader.so
cc_binary(
    name = "libtext_proofreader.so",
    linkopts = [
        "-Wl,-soname=libtext_proofreader.so",
        "-fvisibility=hidden",
        "-Wl,-z,muldefs",
    ],
    linkshared = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [":text_proofreader_lib"],
)

# bazel build --config darwin_arm64 -c opt --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c/text/text_proofreader:libtext_proofreader.dylib
cc_binary(
    name = "libtext_proofreader.dylib",
    linkopts = [
        "-Wl,-install_name,libtext_proofreader.dylib",
        "-fvisibility=hidden",
        "-Wl,-z,muldefs",
    ],
    linkshared = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [":text_proofreader_lib"],
)
