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
    name = "text_classifier",
    srcs = [
        "text_classifier.py",
    ],
    deps = [
        "//mediapipe/python:packet_creator",
        "//mediapipe/python:packet_getter",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_py_pb2",
        "//mediapipe/tasks/cc/components/processors/proto:classifier_options_py_pb2",
        "//mediapipe/tasks/cc/text/text_classifier/proto:text_classifier_graph_options_py_pb2",
        "//mediapipe/tasks/python/components/containers:classification_result",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:task_info",
        "//mediapipe/tasks/python/text/core:base_text_task_api",
    ],
)

py_library(
    name = "text_embedder",
    srcs = [
        "text_embedder.py",
    ],
    deps = [
        "//mediapipe/python:packet_creator",
        "//mediapipe/python:packet_getter",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_py_pb2",
        "//mediapipe/tasks/cc/components/processors/proto:embedder_options_py_pb2",
        "//mediapipe/tasks/cc/text/text_embedder/proto:text_embedder_graph_options_py_pb2",
        "//mediapipe/tasks/python/components/containers:embedding_result",
        "//mediapipe/tasks/python/components/utils:cosine_similarity",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:task_info",
        "//mediapipe/tasks/python/text/core:base_text_task_api",
    ],
)

py_library(
    name = "language_detector",
    srcs = [
        "language_detector.py",
    ],
    deps = [
        "//mediapipe/python:packet_creator",
        "//mediapipe/python:packet_getter",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_py_pb2",
        "//mediapipe/tasks/cc/components/processors/proto:classifier_options_py_pb2",
        "//mediapipe/tasks/cc/text/text_classifier/proto:text_classifier_graph_options_py_pb2",
        "//mediapipe/tasks/python/components/containers:classification_result",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:task_info",
        "//mediapipe/tasks/python/text/core:base_text_task_api",
    ],
)
