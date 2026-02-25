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
    "wasm/genai_wasm_internal.js",
    "wasm/genai_wasm_internal.wasm",
    "wasm/genai_wasm_nosimd_internal.js",
    "wasm/genai_wasm_nosimd_internal.wasm",
])

GENAI_LIBS = [
    "//mediapipe/tasks/web/core:fileset_resolver",
    "//mediapipe/tasks/web/genai/llm_inference",
]

mediapipe_ts_library(
    name = "genai_lib",
    srcs = ["index.ts"],
    visibility = ["//visibility:public"],
    deps = GENAI_LIBS,
)

mediapipe_ts_library(
    name = "genai_types",
    srcs = ["types.ts"],
    visibility = ["//visibility:public"],
    deps = GENAI_LIBS,
)

rollup_bundle(
    name = "genai_bundle_mjs",
    config_file = "//mediapipe/tasks/web:rollup.config.mjs",
    entry_point = "index.ts",
    format = "esm",
    output_dir = False,
    sourcemap = "true",
    deps = [
        ":genai_lib",
        "@npm//@rollup/plugin-commonjs",
        "@npm//@rollup/plugin-node-resolve",
        "@npm//@rollup/plugin-terser",
        "@npm//google-protobuf",
    ],
)

rollup_bundle(
    name = "genai_bundle_cjs",
    config_file = "//mediapipe/tasks/web:rollup.config.mjs",
    entry_point = "index.ts",
    format = "cjs",
    output_dir = False,
    sourcemap = "true",
    deps = [
        ":genai_lib",
        "@npm//@rollup/plugin-commonjs",
        "@npm//@rollup/plugin-node-resolve",
        "@npm//@rollup/plugin-terser",
        "@npm//google-protobuf",
    ],
)

genrule(
    name = "genai_sources",
    srcs = [
        ":genai_bundle_cjs",
        ":genai_bundle_mjs",
    ],
    outs = [
        "genai_bundle.cjs",
        "genai_bundle.cjs.map",
        "genai_bundle.mjs",
        "genai_bundle.mjs.map",
    ],
    cmd = (
        "for FILE in $(SRCS); do " +
        "  OUT_FILE=$(GENDIR)/mediapipe/tasks/web/genai/$$(" +
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
    name = "genai_pkg",
    package_name = "@mediapipe/tasks-__NAME__",
    srcs = ["README.md"],
    substitutions = {
        "__NAME__": "genai",
        "__DESCRIPTION__": "MediaPipe GenAI Tasks",
        "__TYPES__": "genai.d.ts",
    },
    tgz = "genai.tgz",
    deps = [
        "wasm/genai_wasm_internal.js",
        "wasm/genai_wasm_internal.wasm",
        "wasm/genai_wasm_nosimd_internal.js",
        "wasm/genai_wasm_nosimd_internal.wasm",
        ":genai_sources",
        ":package_json",
    ],
)
