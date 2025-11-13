# Copyright 2022 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Placeholder for internal Python strict library and test compatibility macro.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "audio_data",
    srcs = ["audio_data.py"],
    deps = [
        ":audio_data_c",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "@mediapipe_pip_deps_numpy//:pkg",
    ],
)

py_library(
    name = "audio_data_c",
    srcs = ["audio_data_c.py"],
)

py_library(
    name = "bounding_box",
    srcs = ["bounding_box.py"],
    deps = [
        ":rect_c",
        "//mediapipe/framework/formats:location_data_py_pb2",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
)

py_library(
    name = "rect",
    srcs = ["rect.py"],
    deps = [
        ":rect_c",
        "//mediapipe/framework/formats:rect_py_pb2",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
)

py_library(
    name = "rect_c",
    srcs = ["rect_c.py"],
)

py_library(
    name = "landmark",
    srcs = ["landmark.py"],
    deps = [
        ":landmark_c",
        "//mediapipe/framework/formats:landmark_py_pb2",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
)

py_library(
    name = "landmark_c",
    srcs = ["landmark_c.py"],
)

py_library(
    name = "landmark_detection_result",
    srcs = ["landmark_detection_result.py"],
    deps = [
        ":landmark",
        ":rect",
        "//mediapipe/framework/formats:classification_py_pb2",
        "//mediapipe/framework/formats:landmark_py_pb2",
        "//mediapipe/tasks/cc/components/containers/proto:landmarks_detection_result_py_pb2",
        "//mediapipe/tasks/python/components/containers:category",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
)

py_library(
    name = "category",
    srcs = ["category.py"],
    deps = [
        ":category_c",
        "//mediapipe/framework/formats:classification_py_pb2",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
)

py_library(
    name = "category_c",
    srcs = ["category_c.py"],
)

py_library(
    name = "keypoint",
    srcs = ["keypoint.py"],
    deps = [
        ":keypoint_c",
        "//mediapipe/framework/formats:location_data_py_pb2",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
)

py_library(
    name = "keypoint_c",
    srcs = ["keypoint_c.py"],
)

py_library(
    name = "detections",
    srcs = ["detections.py"],
    deps = [
        ":bounding_box",
        ":category",
        ":category_c",
        ":detections_c",
        ":keypoint",
        "//mediapipe/framework/formats:detection_py_pb2",
        "//mediapipe/framework/formats:location_data_py_pb2",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
)

py_library(
    name = "detections_c",
    srcs = ["detections_c.py"],
    deps = [
        ":category_c",
        ":keypoint_c",
        ":rect_c",
    ],
)

py_library(
    name = "classification_result",
    srcs = ["classification_result.py"],
    deps = [
        ":category",
        ":classification_result_c",
        "//mediapipe/framework/formats:classification_py_pb2",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_py_pb2",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
)

py_library(
    name = "classification_result_c",
    srcs = ["classification_result_c.py"],
    deps = [
        ":category_c",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
)

py_library(
    name = "embedding_result",
    srcs = ["embedding_result.py"],
    deps = [
        ":embedding_result_c",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_py_pb2",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "@mediapipe_pip_deps_numpy//:pkg",
    ],
)

py_library(
    name = "embedding_result_c",
    srcs = ["embedding_result_c.py"],
)

py_library(
    name = "matrix_c",
    srcs = ["matrix_c.py"],
    deps = [
        "//mediapipe/tasks/python/core:optional_dependencies",
        "@mediapipe_pip_deps_numpy//:pkg",
    ],
)
