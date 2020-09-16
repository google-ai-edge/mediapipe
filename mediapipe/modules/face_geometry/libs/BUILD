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

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "effect_renderer",
    srcs = ["effect_renderer.cc"],
    hdrs = ["effect_renderer.h"],
    deps = [
        ":mesh_3d_utils",
        ":validation_utils",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:matrix_data_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/gpu:gl_base",
        "//mediapipe/gpu:shader_util",
        "//mediapipe/modules/face_geometry/protos:environment_cc_proto",
        "//mediapipe/modules/face_geometry/protos:face_geometry_cc_proto",
        "//mediapipe/modules/face_geometry/protos:mesh_3d_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "geometry_pipeline",
    srcs = ["geometry_pipeline.cc"],
    hdrs = ["geometry_pipeline.h"],
    deps = [
        ":mesh_3d_utils",
        ":procrustes_solver",
        ":validation_utils",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/formats:matrix_data_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/modules/face_geometry/protos:environment_cc_proto",
        "//mediapipe/modules/face_geometry/protos:face_geometry_cc_proto",
        "//mediapipe/modules/face_geometry/protos:geometry_pipeline_metadata_cc_proto",
        "//mediapipe/modules/face_geometry/protos:mesh_3d_cc_proto",
        "@com_google_absl//absl/memory",
        "@eigen_archive//:eigen",
    ],
)

cc_library(
    name = "mesh_3d_utils",
    srcs = ["mesh_3d_utils.cc"],
    hdrs = ["mesh_3d_utils.h"],
    deps = [
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/modules/face_geometry/protos:mesh_3d_cc_proto",
    ],
)

cc_library(
    name = "procrustes_solver",
    srcs = ["procrustes_solver.cc"],
    hdrs = ["procrustes_solver.h"],
    deps = [
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "@com_google_absl//absl/memory",
        "@eigen_archive//:eigen",
    ],
)

cc_library(
    name = "validation_utils",
    srcs = ["validation_utils.cc"],
    hdrs = ["validation_utils.h"],
    deps = [
        ":mesh_3d_utils",
        "//mediapipe/framework/formats:matrix_data_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/modules/face_geometry/protos:environment_cc_proto",
        "//mediapipe/modules/face_geometry/protos:face_geometry_cc_proto",
        "//mediapipe/modules/face_geometry/protos:geometry_pipeline_metadata_cc_proto",
        "//mediapipe/modules/face_geometry/protos:mesh_3d_cc_proto",
    ],
)
