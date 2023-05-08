# This contains the MediaPipe Vision Tasks.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")
load("@build_bazel_rules_nodejs//:index.bzl", "pkg_npm")
load("@npm//@bazel/rollup:index.bzl", "rollup_bundle")
load(
    "//mediapipe/framework/tool:mediapipe_files.bzl",
    "mediapipe_files",
)

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_files(srcs = [
    "wasm/vision_wasm_internal.js",
    "wasm/vision_wasm_internal.wasm",
    "wasm/vision_wasm_nosimd_internal.js",
    "wasm/vision_wasm_nosimd_internal.wasm",
])

VISION_LIBS = [
    "//mediapipe/tasks/web/core:fileset_resolver",
    "//mediapipe/tasks/web/vision/core:drawing_utils",
    "//mediapipe/tasks/web/vision/core:image",
    "//mediapipe/tasks/web/vision/core:mask",
    "//mediapipe/tasks/web/vision/face_detector",
    "//mediapipe/tasks/web/vision/face_landmarker",
    "//mediapipe/tasks/web/vision/face_stylizer",
    "//mediapipe/tasks/web/vision/gesture_recognizer",
    "//mediapipe/tasks/web/vision/hand_landmarker",
    "//mediapipe/tasks/web/vision/image_classifier",
    "//mediapipe/tasks/web/vision/image_embedder",
    "//mediapipe/tasks/web/vision/image_segmenter",
    "//mediapipe/tasks/web/vision/interactive_segmenter",
    "//mediapipe/tasks/web/vision/object_detector",
    "//mediapipe/tasks/web/vision/pose_landmarker",
]

mediapipe_ts_library(
    name = "vision_lib",
    srcs = ["index.ts"],
    visibility = ["//visibility:public"],
    deps = VISION_LIBS,
)

mediapipe_ts_library(
    name = "vision_types",
    srcs = ["types.ts"],
    visibility = ["//visibility:public"],
    deps = VISION_LIBS,
)

rollup_bundle(
    name = "vision_bundle",
    config_file = "//mediapipe/tasks/web:rollup.config.mjs",
    entry_point = "index.ts",
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

genrule(
    name = "package_json",
    srcs = ["//mediapipe/tasks/web:package.json"],
    outs = ["package.json"],
    cmd = "cp $< $@",
)

pkg_npm(
    name = "vision_pkg",
    package_name = "@mediapipe/tasks-__NAME__",
    srcs = ["README.md"],
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
        ":package_json",
        ":vision_bundle",
    ],
)
