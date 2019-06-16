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

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "base",
    srcs = [
        "file.cc",
        "filesystem.cc",
        "helpers.cc",
    ],
    hdrs = [
        "file.h",
        "filesystem.h",
        "helpers.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
    ],
)
