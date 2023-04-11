# This contains the MediaPipe Text Tasks.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")
load("@build_bazel_rules_nodejs//:index.bzl", "pkg_npm")
load("@npm//@bazel/rollup:index.bzl", "rollup_bundle")
load(
    "//mediapipe/framework/tool:mediapipe_files.bzl",
    "mediapipe_files",
)

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_files(srcs = [
    "wasm/text_wasm_internal.js",
    "wasm/text_wasm_internal.wasm",
    "wasm/text_wasm_nosimd_internal.js",
    "wasm/text_wasm_nosimd_internal.wasm",
])

TEXT_LIBS = [
    "//mediapipe/tasks/web/core:fileset_resolver",
    "//mediapipe/tasks/web/text/language_detector",
    "//mediapipe/tasks/web/text/text_classifier",
    "//mediapipe/tasks/web/text/text_embedder",
]

mediapipe_ts_library(
    name = "text_lib",
    srcs = ["index.ts"],
    visibility = ["//visibility:public"],
    deps = TEXT_LIBS,
)

mediapipe_ts_library(
    name = "text_types",
    srcs = ["types.ts"],
    visibility = ["//visibility:public"],
    deps = TEXT_LIBS,
)

rollup_bundle(
    name = "text_bundle",
    config_file = "//mediapipe/tasks/web:rollup.config.mjs",
    entry_point = "index.ts",
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

genrule(
    name = "package_json",
    srcs = ["//mediapipe/tasks/web:package.json"],
    outs = ["package.json"],
    cmd = "cp $< $@",
)

pkg_npm(
    name = "text_pkg",
    package_name = "@mediapipe/tasks-__NAME__",
    srcs = ["README.md"],
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
        ":package_json",
        ":text_bundle",
    ],
)
