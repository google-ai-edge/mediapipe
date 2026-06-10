# This contains the MediaPipe GenAI Tasks.

load("//mediapipe/framework/port:build_config.bzl", "ts_library")
load(
    "//mediapipe/framework/tool:mediapipe_files.bzl",
    "mediapipe_files",
)
load("//mediapipe/tasks/web:build_defs.bzl", "pkg_npm", "rollup_bundle")

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_files(srcs = [
    "wasm/genai_wasm_internal.js",
    "wasm/genai_wasm_internal.wasm",
    "wasm/genai_wasm_module_internal.js",
    "wasm/genai_wasm_module_internal.wasm",
    "wasm/genai_wasm_nosimd_internal.js",
    "wasm/genai_wasm_nosimd_internal.wasm",
])

GENAI_LIBS = [
    "//mediapipe/tasks/web/core:fileset_resolver",
    "//mediapipe/tasks/web/genai/llm_inference",
]

ts_library(
    name = "genai_lib",
    srcs = ["index.ts"],
    visibility = ["//visibility:public"],
    deps = GENAI_LIBS,
)

ts_library(
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
        "//:node_modules/@rollup/plugin-commonjs",
        "//:node_modules/@rollup/plugin-node-resolve",
        "//:node_modules/@rollup/plugin-terser",
        "//:node_modules/google-protobuf",
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
        "//:node_modules/@rollup/plugin-commonjs",
        "//:node_modules/@rollup/plugin-node-resolve",
        "//:node_modules/@rollup/plugin-terser",
        "//:node_modules/google-protobuf",
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

pkg_npm(
    name = "genai_pkg",
    srcs = ["README.md"],
    package_json = "//mediapipe/tasks/web:package.json",
    substitutions = {
        "__NAME__": "genai",
        "__DESCRIPTION__": "MediaPipe GenAI Tasks",
        "__TYPES__": "genai.d.ts",
    },
    deps = [
        "wasm/genai_wasm_internal.js",
        "wasm/genai_wasm_internal.wasm",
        "wasm/genai_wasm_module_internal.js",
        "wasm/genai_wasm_module_internal.wasm",
        "wasm/genai_wasm_nosimd_internal.js",
        "wasm/genai_wasm_nosimd_internal.wasm",
        ":genai_sources",
    ],
)
