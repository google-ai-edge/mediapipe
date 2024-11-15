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

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

cc_library(
    name = "interactive_segmenter_lib",
    srcs = ["interactive_segmenter.cc"],
    hdrs = ["interactive_segmenter.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/tasks/c/components/containers:keypoint",
        "//mediapipe/tasks/c/core:base_options",
        "//mediapipe/tasks/c/core:base_options_converter",
        "//mediapipe/tasks/c/vision/core:common",
        "//mediapipe/tasks/c/vision/image_segmenter:image_segmenter_result",
        "//mediapipe/tasks/c/vision/image_segmenter:image_segmenter_result_converter",
        "//mediapipe/tasks/cc/components/containers:keypoint",
        "//mediapipe/tasks/cc/vision/interactive_segmenter",
        "//mediapipe/tasks/cc/vision/utils:image_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_test(
    name = "interactive_segmenter_test",
    srcs = ["interactive_segmenter_test.cc"],
    data = [
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/tasks/testdata/vision:test_images",
        "//mediapipe/tasks/testdata/vision:test_models",
    ],
    linkstatic = 1,
    deps = [
        ":interactive_segmenter_lib",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/c/components/containers:keypoint",
        "//mediapipe/tasks/c/test:test_utils",
        "//mediapipe/tasks/c/vision/core:common",
        "//mediapipe/tasks/c/vision/image_segmenter:image_segmenter_result",
        "//mediapipe/tasks/cc/vision/utils:image_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

# bazel build -c opt --linkopt -s --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c/vision/interactive_segmenter:libinteractive_segmenter.so
cc_binary(
    name = "libinteractive_segmenter.so",
    linkopts = [
        "-Wl,-soname=libinteractive_segmenter.so",
        "-fvisibility=hidden",
    ],
    linkshared = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [":interactive_segmenter_lib"],
)

# bazel build --config darwin_arm64 -c opt --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c/vision/interactive_segmenter:libinteractive_segmenter.dylib
cc_binary(
    name = "libinteractive_segmenter.dylib",
    linkopts = [
        "-Wl,-install_name,libinteractive_segmenter.dylib",
        "-fvisibility=hidden",
    ],
    linkshared = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [":interactive_segmenter_lib"],
)
