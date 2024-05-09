# This contains the MediaPipe GenAI Tasks.

load("@build_bazel_rules_nodejs//:index.bzl", "pkg_npm")
load("@npm//@bazel/rollup:index.bzl", "rollup_bundle")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")
load(
    "//mediapipe/framework/tool:mediapipe_files.bzl",
    "mediapipe_files",
)

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_files(srcs = [
    "wasm/genai_experimental_wasm_internal.js",
    "wasm/genai_experimental_wasm_internal.wasm",
    "wasm/genai_experimental_wasm_nosimd_internal.js",
    "wasm/genai_experimental_wasm_nosimd_internal.wasm",
])

GENAI_EXPERIMENTAL_LIBS = [
    "//mediapipe/tasks/web/core:fileset_resolver",
    "//mediapipe/tasks/web/genai_experimental/rag_pipeline",
]

mediapipe_ts_library(
    name = "genai_experimental_lib",
    srcs = ["index.ts"],
    visibility = ["//visibility:public"],
    deps = GENAI_EXPERIMENTAL_LIBS,
)

mediapipe_ts_library(
    name = "genai_experimental_types",
    srcs = ["types.ts"],
    visibility = ["//visibility:public"],
    deps = GENAI_EXPERIMENTAL_LIBS,
)

rollup_bundle(
    name = "genai_experimental_bundle_mjs",
    config_file = "//mediapipe/tasks/web:rollup.config.mjs",
    entry_point = "index.ts",
    format = "esm",
    output_dir = False,
    sourcemap = "true",
    deps = [
        ":genai_experimental_lib",
        "@npm//@rollup/plugin-commonjs",
        "@npm//@rollup/plugin-node-resolve",
        "@npm//@rollup/plugin-terser",
        "@npm//google-protobuf",
    ],
)

rollup_bundle(
    name = "genai_experimental_bundle_cjs",
    config_file = "//mediapipe/tasks/web:rollup.config.mjs",
    entry_point = "index.ts",
    format = "cjs",
    output_dir = False,
    sourcemap = "true",
    deps = [
        ":genai_experimental_lib",
        "@npm//@rollup/plugin-commonjs",
        "@npm//@rollup/plugin-node-resolve",
        "@npm//@rollup/plugin-terser",
        "@npm//google-protobuf",
    ],
)

genrule(
    name = "genai_experimental_sources",
    srcs = [
        ":genai_experimental_bundle_cjs",
        ":genai_experimental_bundle_mjs",
    ],
    outs = [
        "genai_experimental_bundle.cjs",
        "genai_experimental_bundle.cjs.map",
        "genai_experimental_bundle.mjs",
        "genai_experimental_bundle.mjs.map",
    ],
    cmd = (
        "for FILE in $(SRCS); do " +
        "  OUT_FILE=$(GENDIR)/mediapipe/tasks/web/genai_experimental/$$(" +
        "      basename $$FILE | sed -E 's/_([cm])js\\.js/.\\1js/'" +
        "  ); " +
        "  echo $$FILE ; echo $$OUT_FILE ; " +
        "  cp $$FILE $$OUT_FILE ; " +
        "done;"
    ),
)

pkg_npm(
    name = "genai_experimental_pkg",
    package_name = "@mediapipe/tasks-genai-experimental",
    srcs = [
        "README.md",
        "package.json",
    ],
    substitutions = {
        "__NAME__": "genai_experimental",
        "__DESCRIPTION__": "MediaPipe GenAI Experimental Tasks",
        "__TYPES__": "genai_experimental.d.ts",
    },
    tgz = "genai_experimental.tgz",
    deps = [
        "package.json",
        "wasm/genai_experimental_wasm_internal.js",
        "wasm/genai_experimental_wasm_internal.wasm",
        "wasm/genai_experimental_wasm_nosimd_internal.js",
        "wasm/genai_experimental_wasm_nosimd_internal.wasm",
        ":genai_experimental_sources",
    ],
)
