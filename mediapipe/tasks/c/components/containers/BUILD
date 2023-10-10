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
    name = "category",
    hdrs = ["category.h"],
)

cc_library(
    name = "category_converter",
    srcs = ["category_converter.cc"],
    hdrs = ["category_converter.h"],
    deps = [
        ":category",
        "//mediapipe/tasks/cc/components/containers:category",
    ],
)

cc_test(
    name = "category_converter_test",
    srcs = ["category_converter_test.cc"],
    deps = [
        ":category",
        ":category_converter",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/cc/components/containers:category",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "classification_result",
    hdrs = ["classification_result.h"],
)

cc_library(
    name = "classification_result_converter",
    srcs = ["classification_result_converter.cc"],
    hdrs = ["classification_result_converter.h"],
    deps = [
        ":category",
        ":category_converter",
        ":classification_result",
        "//mediapipe/tasks/cc/components/containers:classification_result",
    ],
)

cc_test(
    name = "classification_result_converter_test",
    srcs = ["classification_result_converter_test.cc"],
    deps = [
        ":classification_result",
        ":classification_result_converter",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/cc/components/containers:classification_result",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "embedding_result",
    hdrs = ["embedding_result.h"],
)

cc_library(
    name = "embedding_result_converter",
    srcs = ["embedding_result_converter.cc"],
    hdrs = ["embedding_result_converter.h"],
    deps = [
        ":embedding_result",
        "//mediapipe/tasks/cc/components/containers:embedding_result",
    ],
)

cc_test(
    name = "embedding_result_converter_test",
    srcs = ["embedding_result_converter_test.cc"],
    deps = [
        ":embedding_result",
        ":embedding_result_converter",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/cc/components/containers:embedding_result",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "language_detection_result_converter",
    srcs = ["language_detection_result_converter.cc"],
    hdrs = ["language_detection_result_converter.h"],
    deps = [
        "//mediapipe/tasks/c/text/language_detector",
        "//mediapipe/tasks/cc/text/language_detector",
    ],
)

cc_test(
    name = "language_detection_result_converter_test",
    srcs = ["language_detection_result_converter_test.cc"],
    linkstatic = 1,
    deps = [
        ":language_detection_result_converter",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/c/text/language_detector",
        "//mediapipe/tasks/cc/text/language_detector",
        "@com_google_googletest//:gtest_main",
    ],
)
