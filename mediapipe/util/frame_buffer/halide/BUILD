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

load("@halide//:halide.bzl", "halide_library")

package(default_visibility = ["//mediapipe/util/frame_buffer:__subpackages__"])

# Common Halide library:
cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    deps = ["@halide//:language"],
)

# Enable Halide's built-in profiler with:
#   bazel ... --define halide_target_features=profile
# and HTML output of its intermediate representation with:
#   bazel ... --define halide_extra_outputs=html

# RGB operations:
halide_library(
    name = "rgb_flip_halide",
    srcs = ["rgb_flip_generator.cc"],
    generator_name = "rgb_flip_generator",
)

halide_library(
    name = "rgb_resize_halide",
    srcs = ["rgb_resize_generator.cc"],
    generator_deps = [":common"],
    generator_name = "rgb_resize_generator",
)

halide_library(
    name = "rgb_rotate_halide",
    srcs = ["rgb_rotate_generator.cc"],
    generator_deps = [":common"],
    generator_name = "rgb_rotate_generator",
)

halide_library(
    name = "rgb_yuv_halide",
    srcs = ["rgb_yuv_generator.cc"],
    generator_name = "rgb_yuv_generator",
)

halide_library(
    name = "rgb_rgb_halide",
    srcs = ["rgb_rgb_generator.cc"],
    generator_name = "rgb_rgb_generator",
)

halide_library(
    name = "rgb_float_halide",
    srcs = ["rgb_float_generator.cc"],
    generator_name = "rgb_float_generator",
)

# YUV operations:
halide_library(
    name = "yuv_flip_halide",
    srcs = ["yuv_flip_generator.cc"],
    generator_name = "yuv_flip_generator",
)

halide_library(
    name = "yuv_rgb_halide",
    srcs = ["yuv_rgb_generator.cc"],
    generator_name = "yuv_rgb_generator",
)

halide_library(
    name = "yuv_resize_halide",
    srcs = ["yuv_resize_generator.cc"],
    generator_deps = [":common"],
    generator_name = "yuv_resize_generator",
)

halide_library(
    name = "yuv_rotate_halide",
    srcs = ["yuv_rotate_generator.cc"],
    generator_deps = [":common"],
    generator_name = "yuv_rotate_generator",
)

# Grayscale operations:

halide_library(
    name = "rgb_gray_halide",
    srcs = ["rgb_gray_generator.cc"],
    generator_name = "rgb_gray_generator",
)

halide_library(
    name = "gray_rotate_halide",
    srcs = ["gray_rotate_generator.cc"],
    generator_deps = [":common"],
    generator_name = "gray_rotate_generator",
)

halide_library(
    name = "gray_flip_halide",
    srcs = ["gray_flip_generator.cc"],
    generator_name = "gray_flip_generator",
)

halide_library(
    name = "gray_resize_halide",
    srcs = ["gray_resize_generator.cc"],
    generator_deps = [":common"],
    generator_name = "gray_resize_generator",
)
