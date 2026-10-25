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

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

TEXT_SUMMARIZER_DEPS = [
    "@com_google_absl//absl/log:absl_check",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:status_macros",
    "@com_google_absl//absl/status:statusor",
    "//mediapipe/framework/port:status",
    "//mediapipe/tasks/c/core:base_options",
    "//mediapipe/tasks/c/core:base_options_converter",
    "//mediapipe/tasks/c/core:mp_status",
    "//mediapipe/tasks/c/core:mp_status_converter",
    "//mediapipe/tasks/cc/text/text_summarizer:text_summarizer",
]

cc_library(
    name = "text_summarizer_lib",
    srcs = ["text_summarizer.cc"],
    hdrs = ["text_summarizer.h"],
    deps = TEXT_SUMMARIZER_DEPS,
)

# Support for C bindings
cc_library(
    name = "text_summarizer_c_lib",
    srcs = ["text_summarizer.cc"],
    hdrs = ["text_summarizer.h"],
    visibility = ["//mediapipe/tasks/c:__subpackages__"],
    deps = TEXT_SUMMARIZER_DEPS,
    alwayslink = 1,
)

# bazel build -c opt --linkopt -s --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c/text/text_summarizer:libtext_summarizer.so
cc_binary(
    name = "libtext_summarizer.so",
    linkopts = [
        "-Wl,-soname=libtext_summarizer.so",
        "-fvisibility=hidden",
    ],
    linkshared = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [":text_summarizer_lib"],
)

# bazel build --config darwin_arm64 -c opt --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c/text/text_summarizer:libtext_summarizer.dylib
cc_binary(
    name = "libtext_summarizer.dylib",
    linkopts = [
        "-Wl,-install_name,libtext_summarizer.dylib",
        "-fvisibility=hidden",
    ],
    linkshared = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [":text_summarizer_lib"],
)
