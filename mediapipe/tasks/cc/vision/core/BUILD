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

load("@org_tensorflow//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "running_mode",
    hdrs = ["running_mode.h"],
)

cc_library(
    name = "image_processing_options",
    hdrs = ["image_processing_options.h"],
    deps = [
        "//mediapipe/tasks/cc/components/containers:rect",
    ],
)

cc_library_with_tflite(
    name = "base_vision_task_api",
    hdrs = ["base_vision_task_api.h"],
    tflite_deps = [
        "//mediapipe/tasks/cc/core:base_task_api",
        "//mediapipe/tasks/cc/core:task_api_factory",
        "//mediapipe/tasks/cc/core:task_runner",
        "//mediapipe/tasks/cc/vision/utils:image_tensor_specs",
    ],
    deps = [
        ":image_processing_options",
        ":running_mode",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/tasks/cc/components/containers:rect",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library_with_tflite(
    name = "vision_task_api_factory",
    hdrs = ["vision_task_api_factory.h"],
    tflite_deps = [
        ":base_vision_task_api",
        "//mediapipe/tasks/cc/core:task_runner",
        "//mediapipe/tasks/cc/core:task_api_factory",
    ],
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite/core/api:op_resolver",
    ],
)
