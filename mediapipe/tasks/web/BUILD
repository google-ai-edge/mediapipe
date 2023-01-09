# This contains the MediaPipe Tasks NPM package definitions.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")
load("@build_bazel_rules_nodejs//:index.bzl", "pkg_npm")
load("@npm//@bazel/rollup:index.bzl", "rollup_bundle")
load(
    "//mediapipe/framework/tool:mediapipe_files.bzl",
    "mediapipe_files",
)

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_files(srcs = [
    "wasm/audio_wasm_internal.js",
    "wasm/audio_wasm_internal.wasm",
    "wasm/audio_wasm_nosimd_internal.js",
    "wasm/audio_wasm_nosimd_internal.wasm",
    "wasm/text_wasm_internal.js",
    "wasm/text_wasm_internal.wasm",
    "wasm/text_wasm_nosimd_internal.js",
    "wasm/text_wasm_nosimd_internal.wasm",
    "wasm/vision_wasm_internal.js",
    "wasm/vision_wasm_internal.wasm",
    "wasm/vision_wasm_nosimd_internal.js",
    "wasm/vision_wasm_nosimd_internal.wasm",
])

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
    format = "esm",
    output_dir = False,
    sourcemap = "false",
    deps = [
        ":audio_lib",
        "@npm//@rollup/plugin-commonjs",
        "@npm//@rollup/plugin-node-resolve",
        "@npm//@rollup/plugin-terser",
        "@npm//google-protobuf",
    ],
)

pkg_npm(
    name = "audio_pkg",
    package_name = "@mediapipe/tasks-__NAME__",
    srcs = ["package.json"],
    substitutions = {
        "__NAME__": "audio",
        "__DESCRIPTION__": "MediaPipe Audio Tasks",
        "__TYPES__": "audio.d.ts",
    },
    tgz = "audio.tgz",
    deps = [
        "wasm/audio_wasm_internal.js",
        "wasm/audio_wasm_internal.wasm",
        "wasm/audio_wasm_nosimd_internal.js",
        "wasm/audio_wasm_nosimd_internal.wasm",
        ":audio_bundle",
        "//mediapipe/tasks/web/audio:README.md",
    ],
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
    format = "esm",
    output_dir = False,
    sourcemap = "false",
    deps = [
        ":text_lib",
        "@npm//@rollup/plugin-commonjs",
        "@npm//@rollup/plugin-node-resolve",
        "@npm//@rollup/plugin-terser",
        "@npm//google-protobuf",
    ],
)

pkg_npm(
    name = "text_pkg",
    package_name = "@mediapipe/tasks-__NAME__",
    srcs = ["package.json"],
    substitutions = {
        "__NAME__": "text",
        "__DESCRIPTION__": "MediaPipe Text Tasks",
        "__TYPES__": "text.d.ts",
    },
    tgz = "text.tgz",
    deps = [
        "wasm/text_wasm_internal.js",
        "wasm/text_wasm_internal.wasm",
        "wasm/text_wasm_nosimd_internal.js",
        "wasm/text_wasm_nosimd_internal.wasm",
        ":text_bundle",
        "//mediapipe/tasks/web/text:README.md",
    ],
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
    format = "esm",
    output_dir = False,
    sourcemap = "false",
    deps = [
        ":vision_lib",
        "@npm//@rollup/plugin-commonjs",
        "@npm//@rollup/plugin-node-resolve",
        "@npm//@rollup/plugin-terser",
        "@npm//google-protobuf",
    ],
)

pkg_npm(
    name = "vision_pkg",
    package_name = "@mediapipe/tasks-__NAME__",
    srcs = ["package.json"],
    substitutions = {
        "__NAME__": "vision",
        "__DESCRIPTION__": "MediaPipe Vision Tasks",
        "__TYPES__": "vision.d.ts",
    },
    tgz = "vision_pkg.tgz",
    deps = [
        "wasm/vision_wasm_internal.js",
        "wasm/vision_wasm_internal.wasm",
        "wasm/vision_wasm_nosimd_internal.js",
        "wasm/vision_wasm_nosimd_internal.wasm",
        ":vision_bundle",
        "//mediapipe/tasks/web/vision:README.md",
    ],
)
