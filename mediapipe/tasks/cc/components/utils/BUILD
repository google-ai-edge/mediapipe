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

package(default_visibility = ["//mediapipe/tasks:internal"])

cc_library(
    name = "cosine_similarity",
    srcs = ["cosine_similarity.cc"],
    hdrs = ["cosine_similarity.h"],
    deps = [
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/containers:embedding_result",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "cosine_similarity_test",
    srcs = ["cosine_similarity_test.cc"],
    deps = [
        ":cosine_similarity",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/tasks/cc/components/containers:embedding_result",
    ],
)

cc_library(
    name = "gate",
    hdrs = ["gate.h"],
    deps = [
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:gate_calculator_cc_proto",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
    ],
)

# TODO: Enable this test
