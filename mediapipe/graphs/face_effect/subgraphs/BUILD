# Copyright 2020 The MediaPipe Authors.
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

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_simple_subgraph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_simple_subgraph(
    name = "face_landmarks_smoothing",
    graph = "face_landmarks_smoothing.pbtxt",
    register_as = "FaceLandmarksSmoothing",
    deps = [
        "//mediapipe/calculators/util:landmarks_smoothing_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "single_face_geometry_from_detection_gpu",
    graph = "single_face_geometry_from_detection_gpu.pbtxt",
    register_as = "SingleFaceGeometryFromDetectionGpu",
    deps = [
        ":face_landmarks_smoothing",
        "//mediapipe/calculators/core:concatenate_detection_vector_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/modules/face_detection:face_detection_short_range_gpu",
        "//mediapipe/modules/face_geometry:face_geometry_from_detection",
    ],
)

mediapipe_simple_subgraph(
    name = "single_face_geometry_from_landmarks_gpu",
    graph = "single_face_geometry_from_landmarks_gpu.pbtxt",
    register_as = "SingleFaceGeometryFromLandmarksGpu",
    deps = [
        ":face_landmarks_smoothing",
        "//mediapipe/calculators/core:concatenate_vector_calculator",
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/util:landmarks_smoothing_calculator",
        "//mediapipe/modules/face_geometry:face_geometry_from_landmarks",
        "//mediapipe/modules/face_landmark:face_landmark_front_gpu",
    ],
)
