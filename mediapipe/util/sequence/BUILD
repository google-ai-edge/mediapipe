# Copyright 2019 The MediaPipe Authors.
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
#

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:private"])

cc_library(
    name = "media_sequence_util",
    hdrs = ["media_sequence_util.h"],
    visibility = [
        "//mediapipe:__subpackages__",
    ],
    deps = [
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "media_sequence",
    srcs = ["media_sequence.cc"],
    hdrs = ["media_sequence.h"],
    visibility = [
        "//mediapipe:__subpackages__",
    ],
    deps = [
        ":media_sequence_util",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "media_sequence_util_test",
    srcs = ["media_sequence_util_test.cc"],
    deps = [
        ":media_sequence_util",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "media_sequence_test",
    srcs = ["media_sequence_test.cc"],
    deps = [
        ":media_sequence",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:status",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)
