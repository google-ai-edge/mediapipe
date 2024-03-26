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

load("@mediapipe//mediapipe:platforms.bzl", "config_setting_and_platform")

licenses(["notice"])  # Apache 2.0

# Generic Android
config_setting(
    name = "android",
    constraint_values = [
        "@platforms//os:android",
    ],
    visibility = ["//visibility:public"],
)

# Android x86 32-bit.
config_setting_and_platform(
    name = "android_x86",
    constraint_values = [
        "@platforms//os:android",
        "@platforms//cpu:x86_32",
    ],
    visibility = ["//visibility:public"],
)

# Android x86 64-bit.
config_setting_and_platform(
    name = "android_x86_64",
    constraint_values = [
        "@platforms//os:android",
        "@platforms//cpu:x86_64",
    ],
    visibility = ["//visibility:public"],
)

# Android ARMv7.
config_setting_and_platform(
    name = "android_arm",
    constraint_values = [
        "@platforms//os:android",
        "@platforms//cpu:armv7",
    ],
    visibility = ["//visibility:public"],
)

# Android ARM64.
config_setting_and_platform(
    name = "android_arm64",
    constraint_values = [
        "@platforms//os:android",
        "@platforms//cpu:arm64",
    ],
    visibility = ["//visibility:public"],
)

# Generic MacOS.
config_setting(
    name = "macos",
    constraint_values = [
        "@platforms//os:macos",
    ],
    visibility = ["//visibility:public"],
)

# MacOS x86 64-bit.
config_setting_and_platform(
    name = "macos_x86_64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
    visibility = ["//visibility:public"],
)

# MacOS ARM64.
config_setting_and_platform(
    name = "macos_arm64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
    visibility = ["//visibility:public"],
)

# Generic iOS.
config_setting(
    name = "ios",
    constraint_values = [
        "@platforms//os:ios",
    ],
    visibility = ["//visibility:public"],
)

# iOS device ARM32.
config_setting_and_platform(
    name = "ios_armv7",
    constraint_values = [
        "@platforms//os:ios",
        "@platforms//cpu:arm",
    ],
    visibility = ["//visibility:public"],
)

# iOS device ARM64.
config_setting_and_platform(
    name = "ios_arm64",
    constraint_values = [
        "@platforms//os:ios",
        "@platforms//cpu:arm64",
    ],
    visibility = ["//visibility:public"],
)

# iOS device ARM64E.
config_setting_and_platform(
    name = "ios_arm64e",
    constraint_values = [
        "@platforms//os:ios",
        "@platforms//cpu:arm64e",
    ],
    visibility = ["//visibility:public"],
)

# iOS simulator x86 32-bit.
config_setting_and_platform(
    name = "ios_i386",
    constraint_values = [
        "@platforms//os:ios",
        "@platforms//cpu:x86_32",
        "@build_bazel_apple_support//constraints:simulator",
    ],
    visibility = ["//visibility:public"],
)

# iOS simulator x86 64-bit.
config_setting_and_platform(
    name = "ios_x86_64",
    constraint_values = [
        "@platforms//os:ios",
        "@platforms//cpu:x86_64",
        "@build_bazel_apple_support//constraints:simulator",
    ],
    visibility = ["//visibility:public"],
)

# iOS simulator ARM64.
config_setting_and_platform(
    name = "ios_sim_arm64",
    constraint_values = [
        "@platforms//os:ios",
        "@platforms//cpu:arm64",
        "@build_bazel_apple_support//constraints:simulator",
    ],
    visibility = ["//visibility:public"],
)

alias(
    name = "apple",
    actual = select({
        ":macos": ":macos",
        ":ios": ":ios",
        "//conditions:default": ":ios",  # Arbitrarily chosen from above.
    }),
    visibility = ["//visibility:public"],
)

# Windows 64-bit.
config_setting_and_platform(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
        "@platforms//cpu:x86_64",
    ],
    visibility = ["//visibility:public"],
)

# Linux 64-bit.
config_setting_and_platform(
    name = "linux",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    visibility = ["//visibility:public"],
)

exports_files(
    ["provisioning_profile.mobileprovision"],
    visibility = ["//visibility:public"],
)
