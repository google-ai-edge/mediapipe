# Copyright 2020 The MediaPipe Authors.
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

licenses(["notice"])

package(default_visibility = [
    "//mediapipe:__subpackages__",
])

cc_library(
    name = "low_pass_filter",
    srcs = ["low_pass_filter.cc"],
    hdrs = ["low_pass_filter.h"],
    deps = [
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "low_pass_filter_test",
    srcs = ["low_pass_filter_test.cc"],
    deps = [
        ":low_pass_filter",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "one_euro_filter",
    srcs = ["one_euro_filter.cc"],
    hdrs = ["one_euro_filter.h"],
    deps = [
        ":low_pass_filter",
        "//mediapipe/framework/port:integral_types",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "relative_velocity_filter",
    srcs = ["relative_velocity_filter.cc"],
    hdrs = ["relative_velocity_filter.h"],
    deps = [
        ":low_pass_filter",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "relative_velocity_filter_test",
    srcs = ["relative_velocity_filter_test.cc"],
    deps = [
        ":relative_velocity_filter",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:logging",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
    ],
)
