# Copyright 2023 The MediaPipe Authors.
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

licenses(["notice"])

package(default_visibility = ["//mediapipe/tasks:internal"])

cc_library(
    name = "face_geometry_from_landmarks_graph",
    srcs = ["face_geometry_from_landmarks_graph.cc"],
    deps = [
        "//mediapipe/calculators/core:begin_loop_calculator",
        "//mediapipe/calculators/core:end_loop_calculator",
        "//mediapipe/calculators/core:split_proto_list_calculator",
        "//mediapipe/calculators/core:split_vector_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/tasks/cc/vision/face_geometry/calculators:env_generator_calculator",
        "//mediapipe/tasks/cc/vision/face_geometry/calculators:env_generator_calculator_cc_proto",
        "//mediapipe/tasks/cc/vision/face_geometry/calculators:geometry_pipeline_calculator",
        "//mediapipe/tasks/cc/vision/face_geometry/calculators:geometry_pipeline_calculator_cc_proto",
        "//mediapipe/tasks/cc/vision/face_geometry/proto:environment_cc_proto",
        "//mediapipe/tasks/cc/vision/face_geometry/proto:face_geometry_cc_proto",
        "//mediapipe/tasks/cc/vision/face_geometry/proto:face_geometry_graph_options_cc_proto",
        "//mediapipe/util:graph_builder_utils",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)
