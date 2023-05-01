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
)

py_library(
    name = "bounding_box",
    srcs = ["bounding_box.py"],
    deps = [
        "//mediapipe/framework/formats:location_data_py_pb2",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
)

py_library(
    name = "rect",
    srcs = ["rect.py"],
    deps = [
        "//mediapipe/framework/formats:rect_py_pb2",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
)

py_library(
    name = "landmark",
    srcs = ["landmark.py"],
    deps = [
        "//mediapipe/framework/formats:landmark_py_pb2",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
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
        "//mediapipe/framework/formats:classification_py_pb2",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
)

py_library(
    name = "keypoint",
    srcs = ["keypoint.py"],
    deps = [
        "//mediapipe/framework/formats:location_data_py_pb2",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
)

py_library(
    name = "detections",
    srcs = ["detections.py"],
    deps = [
        ":bounding_box",
        ":category",
        ":keypoint",
        "//mediapipe/framework/formats:detection_py_pb2",
        "//mediapipe/framework/formats:location_data_py_pb2",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
)

py_library(
    name = "classification_result",
    srcs = ["classification_result.py"],
    deps = [
        ":category",
        "//mediapipe/framework/formats:classification_py_pb2",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_py_pb2",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
)

py_library(
    name = "embedding_result",
    srcs = ["embedding_result.py"],
    deps = [
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_py_pb2",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
)
