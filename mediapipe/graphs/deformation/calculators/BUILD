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
    name = "face_processor_calculator",
    srcs = ["face_processor_calculator.cc"],
    hdrs = ["Tensor.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
         "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
         "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:vector",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats/object_detection:anchor_cc_proto",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:parse_text_proto",
       "//mediapipe/util:resource_util",
        "@com_google_absl//absl/flags:flag",
    ],
    alwayslink = 1,
)

cc_library(
    name = "warp_affine_calculator",
    srcs = ["warp_affine_calculator.cc"],
    hdrs = ["Tensor.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_highgui",
    ],
    alwayslink = 1,
)




