# This contains the MediaPipe Text Tasks.

load("@build_bazel_rules_nodejs//:index.bzl", "pkg_npm")
load("@npm//@bazel/rollup:index.bzl", "rollup_bundle")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")
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
    name = "text_bundle_mjs",
    config_file = "//mediapipe/tasks/web:rollup.config.mjs",
    entry_point = "index.ts",
    format = "esm",
    output_dir = False,
    sourcemap = "true",
    deps = [
        ":text_lib",
        "@npm//@rollup/plugin-commonjs",
        "@npm//@rollup/plugin-node-resolve",
        "@npm//@rollup/plugin-terser",
        "@npm//google-protobuf",
    ],
)

rollup_bundle(
    name = "text_bundle_cjs",
    config_file = "//mediapipe/tasks/web:rollup.config.mjs",
    entry_point = "index.ts",
    format = "cjs",
    output_dir = False,
    sourcemap = "true",
    deps = [
        ":text_lib",
        "@npm//@rollup/plugin-commonjs",
        "@npm//@rollup/plugin-node-resolve",
        "@npm//@rollup/plugin-terser",
        "@npm//google-protobuf",
    ],
)

genrule(
    name = "text_sources",
    srcs = [
        ":text_bundle_cjs",
        ":text_bundle_mjs",
    ],
    outs = [
        "text_bundle.cjs",
        "text_bundle.cjs.map",
        "text_bundle.mjs",
        "text_bundle.mjs.map",
    ],
    cmd = (
        "for FILE in $(SRCS); do " +
        "  OUT_FILE=$(GENDIR)/mediapipe/tasks/web/text/$$(" +
        "      basename $$FILE | sed -E 's/_([cm])js\\.js/.\\1js/'" +
        "  ); " +
        "  echo $$FILE ; echo $$OUT_FILE ; " +
        "  cp $$FILE $$OUT_FILE ; " +
        "done;"
    ),
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
        ":text_sources",
    ],
)
