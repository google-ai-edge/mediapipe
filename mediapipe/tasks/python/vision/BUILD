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

# Placeholder for internal Python strict library and test compatibility macro.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "object_detector",
    srcs = [
        "object_detector.py",
    ],
    deps = [
        "//mediapipe/tasks/python/components/containers:detections",
        "//mediapipe/tasks/python/components/containers:detections_c",
        "//mediapipe/tasks/python/core:async_result_dispatcher",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:base_options_c",
        "//mediapipe/tasks/python/core:mediapipe_c_bindings",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:serial_dispatcher",
        "//mediapipe/tasks/python/vision/core:base_vision_task_api",
        "//mediapipe/tasks/python/vision/core:image",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:image_processing_options_c",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
    ],
)

py_library(
    name = "image_classifier",
    srcs = [
        "image_classifier.py",
    ],
    deps = [
        "//mediapipe/tasks/python/components/containers:classification_result",
        "//mediapipe/tasks/python/components/containers:classification_result_c",
        "//mediapipe/tasks/python/components/processors:classifier_options",
        "//mediapipe/tasks/python/components/processors:classifier_options_c",
        "//mediapipe/tasks/python/core:async_result_dispatcher",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:base_options_c",
        "//mediapipe/tasks/python/core:mediapipe_c_bindings",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:serial_dispatcher",
        "//mediapipe/tasks/python/vision/core:base_vision_task_api",
        "//mediapipe/tasks/python/vision/core:image",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:image_processing_options_c",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
    ],
)

py_library(
    name = "image_segmenter",
    srcs = [
        "image_segmenter.py",
    ],
    deps = [
        "//mediapipe/tasks/python/core:async_result_dispatcher",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:base_options_c",
        "//mediapipe/tasks/python/core:mediapipe_c_bindings",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:serial_dispatcher",
        "//mediapipe/tasks/python/vision/core:base_vision_task_api",
        "//mediapipe/tasks/python/vision/core:image",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:image_processing_options_c",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
    ],
)

py_library(
    name = "image_embedder",
    srcs = [
        "image_embedder.py",
    ],
    deps = [
        "//mediapipe/tasks/python/components/containers:embedding_result",
        "//mediapipe/tasks/python/components/containers:embedding_result_c",
        "//mediapipe/tasks/python/components/utils:cosine_similarity",
        "//mediapipe/tasks/python/core:async_result_dispatcher",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:base_options_c",
        "//mediapipe/tasks/python/core:mediapipe_c_bindings",
        "//mediapipe/tasks/python/core:mediapipe_c_utils",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:serial_dispatcher",
        "//mediapipe/tasks/python/vision/core:base_vision_task_api",
        "//mediapipe/tasks/python/vision/core:image",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:image_processing_options_c",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
    ],
)

py_library(
    name = "interactive_segmenter",
    srcs = [
        "interactive_segmenter.py",
    ],
    deps = [
        ":image_segmenter",
        "//mediapipe/tasks/python/components/containers:keypoint",
        "//mediapipe/tasks/python/components/containers:keypoint_c",
        "//mediapipe/tasks/python/core:async_result_dispatcher",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:base_options_c",
        "//mediapipe/tasks/python/core:mediapipe_c_bindings",
        "//mediapipe/tasks/python/core:mediapipe_c_utils",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:serial_dispatcher",
        "//mediapipe/tasks/python/vision/core:image",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:image_processing_options_c",
    ],
)

py_library(
    name = "gesture_recognizer_result",
    srcs = ["gesture_recognizer_result.py"],
    deps = [
        ":gesture_recognizer_result_c",
        "//mediapipe/tasks/python/components/containers:category",
        "//mediapipe/tasks/python/components/containers:landmark",
        "//mediapipe/tasks/python/core:optional_dependencies",
    ],
)

py_library(
    name = "gesture_recognizer_result_c",
    srcs = ["gesture_recognizer_result_c.py"],
    deps = [
        "//mediapipe/tasks/python/components/containers:category_c",
        "//mediapipe/tasks/python/components/containers:landmark_c",
    ],
)

py_library(
    name = "gesture_recognizer",
    srcs = [
        "gesture_recognizer.py",
    ],
    deps = [
        ":gesture_recognizer_result",
        ":gesture_recognizer_result_c",
        "//mediapipe/tasks/python/components/processors:classifier_options",
        "//mediapipe/tasks/python/components/processors:classifier_options_c",
        "//mediapipe/tasks/python/core:async_result_dispatcher",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:base_options_c",
        "//mediapipe/tasks/python/core:mediapipe_c_bindings",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:serial_dispatcher",
        "//mediapipe/tasks/python/vision/core:base_vision_task_api",
        "//mediapipe/tasks/python/vision/core:image",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:image_processing_options_c",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
    ],
)

py_library(
    name = "hand_landmarker",
    srcs = [
        "hand_landmarker.py",
    ],
    deps = [
        "//mediapipe/tasks/python/components/containers:category",
        "//mediapipe/tasks/python/components/containers:category_c",
        "//mediapipe/tasks/python/components/containers:landmark",
        "//mediapipe/tasks/python/components/containers:landmark_c",
        "//mediapipe/tasks/python/core:async_result_dispatcher",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:base_options_c",
        "//mediapipe/tasks/python/core:mediapipe_c_bindings",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:serial_dispatcher",
        "//mediapipe/tasks/python/vision/core:base_vision_task_api",
        "//mediapipe/tasks/python/vision/core:image",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:image_processing_options_c",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
    ],
)

py_library(
    name = "pose_landmarker",
    srcs = [
        "pose_landmarker.py",
    ],
    deps = [
        "//mediapipe/tasks/python/components/containers:landmark",
        "//mediapipe/tasks/python/components/containers:landmark_c",
        "//mediapipe/tasks/python/core:async_result_dispatcher",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:base_options_c",
        "//mediapipe/tasks/python/core:mediapipe_c_bindings",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:serial_dispatcher",
        "//mediapipe/tasks/python/vision/core:base_vision_task_api",
        "//mediapipe/tasks/python/vision/core:image",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:image_processing_options_c",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
    ],
)

py_library(
    name = "face_detector",
    srcs = [
        "face_detector.py",
    ],
    deps = [
        "//mediapipe/tasks/python/components/containers:detections",
        "//mediapipe/tasks/python/components/containers:detections_c",
        "//mediapipe/tasks/python/core:async_result_dispatcher",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:base_options_c",
        "//mediapipe/tasks/python/core:mediapipe_c_bindings",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:serial_dispatcher",
        "//mediapipe/tasks/python/vision/core:base_vision_task_api",
        "//mediapipe/tasks/python/vision/core:image",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:image_processing_options_c",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
    ],
)

py_library(
    name = "face_landmarker",
    srcs = [
        "face_landmarker.py",
    ],
    deps = [
        "//mediapipe/tasks/python/components/containers:category",
        "//mediapipe/tasks/python/components/containers:category_c",
        "//mediapipe/tasks/python/components/containers:landmark",
        "//mediapipe/tasks/python/components/containers:landmark_c",
        "//mediapipe/tasks/python/components/containers:matrix_c",
        "//mediapipe/tasks/python/core:async_result_dispatcher",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:base_options_c",
        "//mediapipe/tasks/python/core:mediapipe_c_bindings",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:serial_dispatcher",
        "//mediapipe/tasks/python/vision/core:base_vision_task_api",
        "//mediapipe/tasks/python/vision/core:image",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:image_processing_options_c",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
        "@mediapipe_pip_deps_numpy//:pkg",
    ],
)

py_library(
    name = "holistic_landmarker",
    srcs = [
        "holistic_landmarker.py",
    ],
    deps = [
        "//mediapipe/framework/formats:classification_py_pb2",
        "//mediapipe/framework/formats:landmark_py_pb2",
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/python:packet_creator",
        "//mediapipe/python:packet_getter",
        "//mediapipe/tasks/cc/vision/holistic_landmarker/proto:holistic_landmarker_graph_options_py_pb2",
        "//mediapipe/tasks/cc/vision/holistic_landmarker/proto:holistic_result_py_pb2",
        "//mediapipe/tasks/python/components/containers:category",
        "//mediapipe/tasks/python/components/containers:landmark",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:task_info",
        "//mediapipe/tasks/python/vision/core:base_vision_task_api",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
    ],
)
