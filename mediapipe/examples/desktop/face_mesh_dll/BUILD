# Copyright 2019 The MediaPipe Authors.
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

load("windows_dll_library.bzl", "windows_dll_library")

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//examples:__pkg__"],
)

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

# Define the shared library
windows_dll_library(
    name = "face_mesh_lib",
    srcs = ["face_mesh_lib.cpp"],
    hdrs = ["face_mesh_lib.h"],
    # Define COMPILING_DLL to export symbols during compiling the DLL.
    copts = ["-DCOMPILING_DLL"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",

        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/tflite:tflite_model_calculator",
        "//mediapipe/calculators/util:local_file_contents_calculator",
        "//mediapipe/modules/face_landmark:face_landmark_front_side_model_cpu_with_face_counter",
    ]
)

# **Implicitly link to face_mesh_lib.dll**

## Link to face_mesh_lib.dll through its import library.
cc_binary(
    name = "face_mesh_cpu",
    srcs = ["face_mesh_cpu.cpp"],
    deps = [
        ":face_mesh_lib",
    ],
)