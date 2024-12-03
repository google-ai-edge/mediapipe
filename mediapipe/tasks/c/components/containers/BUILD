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
    name = "matrix",
    hdrs = ["matrix.h"],
)

cc_library(
    name = "matrix_converter",
    srcs = ["matrix_converter.cc"],
    hdrs = ["matrix_converter.h"],
    deps = [
        ":matrix",
        "@eigen_archive//:eigen3",
    ],
)

cc_test(
    name = "matrix_converter_test",
    srcs = ["matrix_converter_test.cc"],
    deps = [
        ":matrix",
        ":matrix_converter",
        "//mediapipe/framework/port:gtest",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "landmark",
    hdrs = ["landmark.h"],
)

cc_library(
    name = "landmark_converter",
    srcs = ["landmark_converter.cc"],
    hdrs = ["landmark_converter.h"],
    deps = [
        ":landmark",
        "//mediapipe/tasks/cc/components/containers:landmark",
    ],
)

cc_test(
    name = "landmark_converter_test",
    srcs = ["landmark_converter_test.cc"],
    deps = [
        ":landmark",
        ":landmark_converter",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/cc/components/containers:landmark",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "rect",
    hdrs = ["rect.h"],
)

cc_library(
    name = "rect_converter",
    srcs = ["rect_converter.cc"],
    hdrs = ["rect_converter.h"],
    deps = [
        ":rect",
        "//mediapipe/tasks/cc/components/containers:rect",
    ],
)

cc_test(
    name = "rect_converter_test",
    srcs = ["rect_converter_test.cc"],
    deps = [
        ":rect",
        ":rect_converter",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/cc/components/containers:rect",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "keypoint",
    hdrs = ["keypoint.h"],
)

cc_library(
    name = "keypoint_converter",
    srcs = ["keypoint_converter.cc"],
    hdrs = ["keypoint_converter.h"],
    deps = [
        ":keypoint",
        "//mediapipe/tasks/cc/components/containers:keypoint",
    ],
)

cc_test(
    name = "keypoint_converter_test",
    srcs = ["keypoint_converter_test.cc"],
    deps = [
        ":keypoint",
        ":keypoint_converter",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/cc/components/containers:keypoint",
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
    name = "detection_result",
    hdrs = ["detection_result.h"],
    deps = [":rect"],
)

cc_library(
    name = "detection_result_converter",
    srcs = ["detection_result_converter.cc"],
    hdrs = ["detection_result_converter.h"],
    deps = [
        ":category",
        ":category_converter",
        ":detection_result",
        ":keypoint",
        ":keypoint_converter",
        ":rect_converter",
        "//mediapipe/tasks/cc/components/containers:detection_result",
    ],
)

cc_test(
    name = "detection_result_converter_test",
    srcs = ["detection_result_converter_test.cc"],
    deps = [
        ":detection_result",
        ":detection_result_converter",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/cc/components/containers:detection_result",
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
