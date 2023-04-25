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
    name = "audio_classifier",
    srcs = [
        "audio_classifier.py",
    ],
    deps = [
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/python:packet_creator",
        "//mediapipe/python:packet_getter",
        "//mediapipe/tasks/cc/audio/audio_classifier/proto:audio_classifier_graph_options_py_pb2",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_py_pb2",
        "//mediapipe/tasks/cc/components/processors/proto:classifier_options_py_pb2",
        "//mediapipe/tasks/python/audio/core:audio_task_running_mode",
        "//mediapipe/tasks/python/audio/core:base_audio_task_api",
        "//mediapipe/tasks/python/components/containers:audio_data",
        "//mediapipe/tasks/python/components/containers:classification_result",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:task_info",
    ],
)

py_library(
    name = "audio_embedder",
    srcs = [
        "audio_embedder.py",
    ],
    deps = [
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/python:packet_creator",
        "//mediapipe/python:packet_getter",
        "//mediapipe/tasks/cc/audio/audio_embedder/proto:audio_embedder_graph_options_py_pb2",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_py_pb2",
        "//mediapipe/tasks/cc/components/processors/proto:embedder_options_py_pb2",
        "//mediapipe/tasks/python/audio/core:audio_task_running_mode",
        "//mediapipe/tasks/python/audio/core:base_audio_task_api",
        "//mediapipe/tasks/python/components/containers:audio_data",
        "//mediapipe/tasks/python/components/containers:embedding_result",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:task_info",
    ],
)
