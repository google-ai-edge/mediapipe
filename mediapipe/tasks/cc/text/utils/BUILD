# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

cc_library(
    name = "vocab_utils",
    srcs = [
        "vocab_utils.cc",
    ],
    hdrs = [
        "vocab_utils.h",
    ],
    deps = [
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "vocab_utils_test",
    srcs = ["vocab_utils_test.cc"],
    data = [
        "//mediapipe/tasks/testdata/text:vocab_files",
    ],
    deps = [
        ":vocab_utils",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/tasks/cc/core:utils",
        "@com_google_absl//absl/container:node_hash_map",
    ],
)
