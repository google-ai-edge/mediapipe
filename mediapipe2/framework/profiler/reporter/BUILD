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
#

licenses(["notice"])

cc_library(
    name = "reporter_lib",
    srcs = [
        "reporter.cc",
        "statistic.cc",
    ],
    hdrs = [
        "reporter.h",
        "statistic.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_profile_cc_proto",
        "//mediapipe/framework/port:advanced_proto",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:re2",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_binary(
    name = "print_profile",
    srcs = ["print_profile.cc"],
    deps = [
        ":reporter_lib",
        "//mediapipe/framework/port:advanced_proto",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
    ],
)
