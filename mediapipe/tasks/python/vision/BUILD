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

# Placeholder for internal Python strict library and test compatibility macro.

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

py_library(
    name = "object_detector",
    srcs = [
        "object_detector.py",
    ],
    deps = [
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/python:packet_creator",
        "//mediapipe/python:packet_getter",
        "//mediapipe/tasks/cc/vision/object_detector/proto:object_detector_options_py_pb2",
        "//mediapipe/tasks/python/components/containers:detections",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:task_info",
        "//mediapipe/tasks/python/vision/core:base_vision_task_api",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
    ],
)

py_library(
    name = "image_classifier",
    srcs = [
        "image_classifier.py",
    ],
    deps = [
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/python:packet_creator",
        "//mediapipe/python:packet_getter",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_py_pb2",
        "//mediapipe/tasks/cc/vision/image_classifier/proto:image_classifier_graph_options_py_pb2",
        "//mediapipe/tasks/python/components/containers:classification_result",
        "//mediapipe/tasks/python/components/containers:rect",
        "//mediapipe/tasks/python/components/processors:classifier_options",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:task_info",
        "//mediapipe/tasks/python/vision/core:base_vision_task_api",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
    ],
)

py_library(
    name = "image_segmenter",
    srcs = [
        "image_segmenter.py",
    ],
    deps = [
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/python:packet_creator",
        "//mediapipe/python:packet_getter",
        "//mediapipe/tasks/cc/components/proto:segmenter_options_py_pb2",
        "//mediapipe/tasks/cc/vision/image_segmenter/proto:image_segmenter_graph_options_py_pb2",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:task_info",
        "//mediapipe/tasks/python/vision/core:base_vision_task_api",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
    ],
)

py_library(
    name = "image_embedder",
    srcs = [
        "image_embedder.py",
    ],
    deps = [
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/python:packet_creator",
        "//mediapipe/python:packet_getter",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_py_pb2",
        "//mediapipe/tasks/cc/vision/image_embedder/proto:image_embedder_graph_options_py_pb2",
        "//mediapipe/tasks/python/components/containers:embedding_result",
        "//mediapipe/tasks/python/components/processors:embedder_options",
        "//mediapipe/tasks/python/components/utils:cosine_similarity",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:task_info",
        "//mediapipe/tasks/python/vision/core:base_vision_task_api",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
    ],
)

py_library(
    name = "gesture_recognizer",
    srcs = [
        "gesture_recognizer.py",
    ],
    deps = [
        "//mediapipe/framework/formats:classification_py_pb2",
        "//mediapipe/framework/formats:landmark_py_pb2",
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/python:packet_creator",
        "//mediapipe/python:packet_getter",
        "//mediapipe/tasks/cc/vision/gesture_recognizer/proto:gesture_recognizer_graph_options_py_pb2",
        "//mediapipe/tasks/python/components/containers:category",
        "//mediapipe/tasks/python/components/containers:landmark",
        "//mediapipe/tasks/python/components/processors:classifier_options",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:task_info",
        "//mediapipe/tasks/python/vision/core:base_vision_task_api",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
    ],
)

py_library(
    name = "hand_landmarker",
    srcs = [
        "hand_landmarker.py",
    ],
    deps = [
        "//mediapipe/framework/formats:classification_py_pb2",
        "//mediapipe/framework/formats:landmark_py_pb2",
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/python:packet_creator",
        "//mediapipe/python:packet_getter",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarker_graph_options_py_pb2",
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
