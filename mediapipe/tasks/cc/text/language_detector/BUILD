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

licenses(["notice"])

cc_library(
    name = "language_detector",
    srcs = ["language_detector.cc"],
    hdrs = ["language_detector.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/api2:builder",
        "//mediapipe/tasks/cc/components/containers:category",
        "//mediapipe/tasks/cc/components/containers:classification_result",
        "//mediapipe/tasks/cc/components/processors:classifier_options",
        "//mediapipe/tasks/cc/core:base_options",
        "//mediapipe/tasks/cc/core:base_task_api",
        "//mediapipe/tasks/cc/core:task_api_factory",
        "//mediapipe/tasks/cc/text/text_classifier:text_classifier_graph",
        "//mediapipe/tasks/cc/text/text_classifier/proto:text_classifier_graph_options_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)
