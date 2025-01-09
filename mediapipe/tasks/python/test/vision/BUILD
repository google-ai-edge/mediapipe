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

# Placeholder for internal Python strict test compatibility macro.

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

py_test(
    name = "object_detector_test",
    srcs = ["object_detector_test.py"],
    data = [
        "//mediapipe/tasks/testdata/vision:test_images",
        "//mediapipe/tasks/testdata/vision:test_models",
    ],
    tags = ["not_run:arm"],
    deps = [
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/tasks/python/components/containers:bounding_box",
        "//mediapipe/tasks/python/components/containers:category",
        "//mediapipe/tasks/python/components/containers:detections",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/test:test_utils",
        "//mediapipe/tasks/python/vision:object_detector",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
        "@mediapipe_pip_deps_absl_py//:pkg",
        "@mediapipe_pip_deps_numpy//:pkg",
    ],
)

py_test(
    name = "image_classifier_test",
    srcs = ["image_classifier_test.py"],
    data = [
        "//mediapipe/tasks/testdata/vision:test_images",
        "//mediapipe/tasks/testdata/vision:test_models",
    ],
    tags = ["not_run:arm"],
    deps = [
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/tasks/python/components/containers:category",
        "//mediapipe/tasks/python/components/containers:classification_result",
        "//mediapipe/tasks/python/components/containers:rect",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/test:test_utils",
        "//mediapipe/tasks/python/vision:image_classifier",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
        "@mediapipe_pip_deps_absl_py//:pkg",
        "@mediapipe_pip_deps_numpy//:pkg",
    ],
)

py_test(
    name = "image_embedder_test",
    srcs = ["image_embedder_test.py"],
    data = [
        "//mediapipe/tasks/testdata/vision:test_images",
        "//mediapipe/tasks/testdata/vision:test_models",
    ],
    tags = ["not_run:arm"],
    deps = [
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/tasks/python/components/containers:embedding_result",
        "//mediapipe/tasks/python/components/containers:rect",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/test:test_utils",
        "//mediapipe/tasks/python/vision:image_embedder",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
        "@mediapipe_pip_deps_absl_py//:pkg",
        "@mediapipe_pip_deps_numpy//:pkg",
    ],
)

py_test(
    name = "image_segmenter_test",
    srcs = ["image_segmenter_test.py"],
    data = [
        "//mediapipe/tasks/testdata/vision:test_images",
        "//mediapipe/tasks/testdata/vision:test_models",
    ],
    tags = ["not_run:arm"],
    deps = [
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/test:test_utils",
        "//mediapipe/tasks/python/vision:image_segmenter",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
        "@mediapipe_pip_deps_absl_py//:pkg",
        "@mediapipe_pip_deps_numpy//:pkg",
    ],
)

py_test(
    name = "face_detector_test",
    srcs = ["face_detector_test.py"],
    data = [
        "//mediapipe/tasks/testdata/vision:test_images",
        "//mediapipe/tasks/testdata/vision:test_models",
        "//mediapipe/tasks/testdata/vision:test_protos",
    ],
    tags = ["not_run:arm"],
    deps = [
        "//mediapipe/framework/formats:detection_py_pb2",
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/tasks/python/components/containers:bounding_box",
        "//mediapipe/tasks/python/components/containers:category",
        "//mediapipe/tasks/python/components/containers:detections",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/test:test_utils",
        "//mediapipe/tasks/python/vision:face_detector",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
        "@com_google_protobuf//:protobuf_python",
        "@mediapipe_pip_deps_absl_py//:pkg",
    ],
)

py_test(
    name = "face_landmarker_test",
    srcs = ["face_landmarker_test.py"],
    data = [
        "//mediapipe/tasks/testdata/vision:test_images",
        "//mediapipe/tasks/testdata/vision:test_models",
        "//mediapipe/tasks/testdata/vision:test_protos",
    ],
    tags = ["not_run:arm"],
    deps = [
        "//mediapipe/framework/formats:classification_py_pb2",
        "//mediapipe/framework/formats:landmark_py_pb2",
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/tasks/python/components/containers:category",
        "//mediapipe/tasks/python/components/containers:landmark",
        "//mediapipe/tasks/python/components/containers:rect",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/test:test_utils",
        "//mediapipe/tasks/python/vision:face_landmarker",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
        "@com_google_protobuf//:protobuf_python",
        "@mediapipe_pip_deps_absl_py//:pkg",
        "@mediapipe_pip_deps_numpy//:pkg",
    ],
)

py_test(
    name = "hand_landmarker_test",
    srcs = ["hand_landmarker_test.py"],
    data = [
        "//mediapipe/tasks/testdata/vision:test_images",
        "//mediapipe/tasks/testdata/vision:test_models",
        "//mediapipe/tasks/testdata/vision:test_protos",
    ],
    tags = ["not_run:arm"],
    deps = [
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/tasks/cc/components/containers/proto:landmarks_detection_result_py_pb2",
        "//mediapipe/tasks/python/components/containers:landmark",
        "//mediapipe/tasks/python/components/containers:landmark_detection_result",
        "//mediapipe/tasks/python/components/containers:rect",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/test:test_utils",
        "//mediapipe/tasks/python/vision:hand_landmarker",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
        "@com_google_protobuf//:protobuf_python",
        "@mediapipe_pip_deps_absl_py//:pkg",
        "@mediapipe_pip_deps_numpy//:pkg",
    ],
)

py_test(
    name = "pose_landmarker_test",
    srcs = ["pose_landmarker_test.py"],
    data = [
        "//mediapipe/tasks/testdata/vision:test_images",
        "//mediapipe/tasks/testdata/vision:test_models",
        "//mediapipe/tasks/testdata/vision:test_protos",
    ],
    tags = ["not_run:arm"],
    deps = [
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/tasks/cc/components/containers/proto:landmarks_detection_result_py_pb2",
        "//mediapipe/tasks/python/components/containers:landmark",
        "//mediapipe/tasks/python/components/containers:landmark_detection_result",
        "//mediapipe/tasks/python/components/containers:rect",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/test:test_utils",
        "//mediapipe/tasks/python/vision:pose_landmarker",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
        "@com_google_protobuf//:protobuf_python",
        "@mediapipe_pip_deps_absl_py//:pkg",
        "@mediapipe_pip_deps_numpy//:pkg",
    ],
)

py_test(
    name = "holistic_landmarker_test",
    srcs = ["holistic_landmarker_test.py"],
    data = [
        "//mediapipe/tasks/testdata/vision:test_images",
        "//mediapipe/tasks/testdata/vision:test_models",
        "//mediapipe/tasks/testdata/vision:test_protos",
    ],
    tags = ["not_run:arm"],
    deps = [
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/tasks/cc/vision/holistic_landmarker/proto:holistic_result_py_pb2",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/test:test_utils",
        "//mediapipe/tasks/python/vision:holistic_landmarker",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "//mediapipe/tasks/python/vision/core:vision_task_running_mode",
        "@com_google_protobuf//:protobuf_python",
        "@mediapipe_pip_deps_absl_py//:pkg",
        "@mediapipe_pip_deps_numpy//:pkg",
    ],
)

py_test(
    name = "face_aligner_test",
    srcs = ["face_aligner_test.py"],
    data = [
        "//mediapipe/tasks/testdata/vision:test_images",
        "//mediapipe/tasks/testdata/vision:test_models",
    ],
    tags = ["not_run:arm"],
    deps = [
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/tasks/python/components/containers:rect",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/test:test_utils",
        "//mediapipe/tasks/python/vision:face_aligner",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "@mediapipe_pip_deps_absl_py//:pkg",
    ],
)

py_test(
    name = "face_stylizer_test",
    srcs = ["face_stylizer_test.py"],
    data = [
        "//mediapipe/tasks/testdata/vision:test_images",
        "//mediapipe/tasks/testdata/vision:test_models",
    ],
    deps = [
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/tasks/python/components/containers:rect",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/test:test_utils",
        "//mediapipe/tasks/python/vision:face_stylizer",
        "//mediapipe/tasks/python/vision/core:image_processing_options",
        "@mediapipe_pip_deps_absl_py//:pkg",
    ],
)
