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
    name = "hand_landmarker_result",
    hdrs = ["hand_landmarker_result.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/c/components/containers:category",
        "//mediapipe/tasks/c/components/containers:landmark",
    ],
)

cc_library(
    name = "hand_landmarker_result_converter",
    srcs = ["hand_landmarker_result_converter.cc"],
    hdrs = ["hand_landmarker_result_converter.h"],
    deps = [
        ":hand_landmarker_result",
        "//mediapipe/tasks/c/components/containers:category",
        "//mediapipe/tasks/c/components/containers:category_converter",
        "//mediapipe/tasks/c/components/containers:landmark",
        "//mediapipe/tasks/c/components/containers:landmark_converter",
        "//mediapipe/tasks/cc/components/containers:category",
        "//mediapipe/tasks/cc/components/containers:landmark",
        "//mediapipe/tasks/cc/vision/hand_landmarker:hand_landmarker_result",
    ],
)

cc_test(
    name = "hand_landmarker_result_converter_test",
    srcs = ["hand_landmarker_result_converter_test.cc"],
    linkstatic = 1,
    deps = [
        ":hand_landmarker_result",
        ":hand_landmarker_result_converter",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/c/components/containers:landmark",
        "//mediapipe/tasks/cc/components/containers:category",
        "//mediapipe/tasks/cc/components/containers:classification_result",
        "//mediapipe/tasks/cc/components/containers:landmark",
        "//mediapipe/tasks/cc/vision/hand_landmarker:hand_landmarker_result",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "hand_landmarker_lib",
    srcs = ["hand_landmarker.cc"],
    hdrs = ["hand_landmarker.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":hand_landmarker_result",
        ":hand_landmarker_result_converter",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/tasks/c/core:base_options",
        "//mediapipe/tasks/c/core:base_options_converter",
        "//mediapipe/tasks/c/vision/core:common",
        "//mediapipe/tasks/cc/vision/core:running_mode",
        "//mediapipe/tasks/cc/vision/hand_landmarker",
        "//mediapipe/tasks/cc/vision/hand_landmarker:hand_landmarker_result",
        "//mediapipe/tasks/cc/vision/utils:image_utils",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_test(
    name = "hand_landmarker_test",
    srcs = ["hand_landmarker_test.cc"],
    data = [
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/tasks/testdata/vision:test_images",
        "//mediapipe/tasks/testdata/vision:test_models",
    ],
    linkstatic = 1,
    deps = [
        ":hand_landmarker_lib",
        ":hand_landmarker_result",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/c/components/containers:landmark",
        "//mediapipe/tasks/c/vision/core:common",
        "//mediapipe/tasks/cc/vision/utils:image_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

# bazel build -c opt --linkopt -s --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c/vision/hand_landmarker:libhand_landmarker.so
cc_binary(
    name = "libhand_landmarker.so",
    linkopts = [
        "-Wl,-soname=libhand_landmarker.so",
        "-fvisibility=hidden",
    ],
    linkshared = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [":hand_landmarker_lib"],
)

# bazel build --config darwin_arm64 -c opt --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c/vision/hand_landmarker:libhand_landmarker.dylib
cc_binary(
    name = "libhand_landmarker.dylib",
    linkopts = [
        "-Wl,-install_name,libhand_landmarker.dylib",
        "-fvisibility=hidden",
    ],
    linkshared = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [":hand_landmarker_lib"],
)
