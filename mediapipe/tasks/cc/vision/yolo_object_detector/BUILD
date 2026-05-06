# Copyright 2026 The MediaPipe Authors.
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "yolo_object_detector_graph_utils",
    srcs = ["yolo_object_detector_graph_utils.cc"],
    hdrs = ["yolo_object_detector_graph_utils.h"],
    deps = [
        "//mediapipe/framework/port:file_helpers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/lite:model_builder",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "yolo_object_detector_graph",
    srcs = ["yolo_object_detector_graph.cc"],
    deps = [
        ":yolo_object_detector_graph_utils",
        "//mediapipe/calculators/yolo:auto_nms_calculator",
        "//mediapipe/calculators/yolo:yolo_tensors_to_detections_calculator",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/processors:image_preprocessing_graph",
        "//mediapipe/tasks/cc/components/processors/proto:image_preprocessing_graph_options_cc_proto",
        "//mediapipe/tasks/cc/components/processors/proto:yolo_tensors_to_detections_calculator_cc_proto",
        "//mediapipe/calculators/util:detection_projection_calculator",
        "//mediapipe/calculators/util:detection_transformation_calculator",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/core:model_task_graph",
        "//mediapipe/tasks/cc/vision/yolo_object_detector/proto:yolo_object_detector_options_cc_proto",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_test(
    name = "yolo_object_detector_graph_test",
    srcs = ["yolo_object_detector_graph_test.cc"],
    linkstatic = 1,
    deps = [
        ":yolo_object_detector_graph",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "yolo_object_detector_graph_utils_test",
    srcs = ["yolo_object_detector_graph_utils_test.cc"],
    linkstatic = 1,
    deps = [
        ":yolo_object_detector_graph_utils",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:status_matchers",
    ],
)
