# This contains the MediaPipe Tasks NPM package definitions.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")
load("@build_bazel_rules_nodejs//:index.bzl", "pkg_npm")
load("@npm//@bazel/rollup:index.bzl", "rollup_bundle")

package(default_visibility = ["//mediapipe/tasks:internal"])

# Audio

mediapipe_ts_library(
    name = "audio_lib",
    srcs = ["audio.ts"],
    deps = ["//mediapipe/tasks/web/audio:audio_lib"],
)

rollup_bundle(
    name = "audio_bundle",
    config_file = "rollup.config.mjs",
    entry_point = "audio.ts",
    output_dir = False,
    deps = [
        ":audio_lib",
        "@npm//@rollup/plugin-commonjs",
        "@npm//@rollup/plugin-node-resolve",
    ],
)

pkg_npm(
    name = "audio_pkg",
    package_name = "__PACKAGE_NAME__",
    srcs = ["package.json"],
    substitutions = {
        "__PACKAGE_NAME__": "@mediapipe/tasks-audio",
        "__DESCRIPTION__": "MediaPipe Audio Tasks",
        "__BUNDLE__": "audio_bundle.js",
    },
    tgz = "audio.tgz",
    deps = [":audio_bundle"],
)

# Text

mediapipe_ts_library(
    name = "text_lib",
    srcs = ["text.ts"],
    deps = ["//mediapipe/tasks/web/text:text_lib"],
)

rollup_bundle(
    name = "text_bundle",
    config_file = "rollup.config.mjs",
    entry_point = "text.ts",
    output_dir = False,
    deps = [
        ":text_lib",
        "@npm//@rollup/plugin-commonjs",
        "@npm//@rollup/plugin-node-resolve",
    ],
)

pkg_npm(
    name = "text_pkg",
    package_name = "__PACKAGE_NAME__",
    srcs = ["package.json"],
    substitutions = {
        "__PACKAGE_NAME__": "@mediapipe/tasks-text",
        "__DESCRIPTION__": "MediaPipe Text Tasks",
        "__BUNDLE__": "text_bundle.js",
    },
    tgz = "text.tgz",
    deps = [":text_bundle"],
)

# Vision

mediapipe_ts_library(
    name = "vision_lib",
    srcs = ["vision.ts"],
    deps = ["//mediapipe/tasks/web/vision:vision_lib"],
)

rollup_bundle(
    name = "vision_bundle",
    config_file = "rollup.config.mjs",
    entry_point = "vision.ts",
    output_dir = False,
    deps = [
        ":vision_lib",
        "@npm//@rollup/plugin-commonjs",
        "@npm//@rollup/plugin-node-resolve",
    ],
)

pkg_npm(
    name = "vision_pkg",
    package_name = "__PACKAGE_NAME__",
    srcs = ["package.json"],
    substitutions = {
        "__PACKAGE_NAME__": "@mediapipe/tasks-vision",
        "__DESCRIPTION__": "MediaPipe Vision Tasks",
        "__BUNDLE__": "vision_bundle.js",
    },
    tgz = "vision.tgz",
    deps = [":vision_bundle"],
)
