# This contains the MediaPipe Text Tasks.

load("//mediapipe/framework/port:build_config.bzl", "ts_library")
load(
    "//mediapipe/framework/tool:mediapipe_files.bzl",
    "mediapipe_files",
)
load("//mediapipe/tasks/web:build_defs.bzl", "pkg_npm", "rollup_bundle")

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_files(srcs = [
    "wasm/text_wasm_internal.js",
    "wasm/text_wasm_internal.wasm",
    "wasm/text_wasm_module_internal.js",
    "wasm/text_wasm_module_internal.wasm",
    "wasm/text_wasm_nosimd_internal.js",
    "wasm/text_wasm_nosimd_internal.wasm",
])

TEXT_LIBS = [
    "//mediapipe/tasks/web/core:fileset_resolver",
    "//mediapipe/tasks/web/text/language_detector",
    "//mediapipe/tasks/web/text/text_classifier",
    "//mediapipe/tasks/web/text/text_embedder",
]

ts_library(
    name = "text_lib",
    srcs = ["index.ts"],
    visibility = ["//visibility:public"],
    deps = TEXT_LIBS,
)

ts_library(
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
        "//:node_modules/@rollup/plugin-commonjs",
        "//:node_modules/@rollup/plugin-node-resolve",
        "//:node_modules/@rollup/plugin-terser",
        "//:node_modules/google-protobuf",
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
        "//:node_modules/@rollup/plugin-commonjs",
        "//:node_modules/@rollup/plugin-node-resolve",
        "//:node_modules/@rollup/plugin-terser",
        "//:node_modules/google-protobuf",
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

pkg_npm(
    name = "text_pkg",
    srcs = ["README.md"],
    package_json = "//mediapipe/tasks/web:package.json",
    substitutions = {
        "__NAME__": "text",
        "__DESCRIPTION__": "MediaPipe Text Tasks",
        "__TYPES__": "text.d.ts",
    },
    deps = [
        "wasm/text_wasm_internal.js",
        "wasm/text_wasm_internal.wasm",
        "wasm/text_wasm_module_internal.js",
        "wasm/text_wasm_module_internal.wasm",
        "wasm/text_wasm_nosimd_internal.js",
        "wasm/text_wasm_nosimd_internal.wasm",
        ":text_sources",
    ],
)
