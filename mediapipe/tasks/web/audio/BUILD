# This contains the MediaPipe Audio Tasks.

load("//mediapipe/framework/port:build_config.bzl", "ts_library")
load(
    "//mediapipe/framework/tool:mediapipe_files.bzl",
    "mediapipe_files",
)
load("//mediapipe/tasks/web:build_defs.bzl", "pkg_npm", "rollup_bundle")

package(default_visibility = ["//mediapipe/tasks:internal"])

AUDIO_LIBS = [
    "//mediapipe/tasks/web/audio/audio_classifier",
    "//mediapipe/tasks/web/core:fileset_resolver",
]

ts_library(
    name = "audio_lib",
    srcs = ["index.ts"],
    visibility = ["//visibility:public"],
    deps = AUDIO_LIBS,
)

ts_library(
    name = "audio_types",
    srcs = ["types.ts"],
    visibility = ["//visibility:public"],
    deps = AUDIO_LIBS,
)

mediapipe_files(srcs = [
    "wasm/audio_wasm_internal.js",
    "wasm/audio_wasm_internal.wasm",
    "wasm/audio_wasm_module_internal.js",
    "wasm/audio_wasm_module_internal.wasm",
    "wasm/audio_wasm_nosimd_internal.js",
    "wasm/audio_wasm_nosimd_internal.wasm",
])

rollup_bundle(
    name = "audio_bundle_mjs",
    config_file = "//mediapipe/tasks/web:rollup.config.mjs",
    entry_point = "index.ts",
    format = "esm",
    output_dir = False,
    sourcemap = "true",
    deps = [
        ":audio_lib",
        "//:node_modules/@rollup/plugin-commonjs",
        "//:node_modules/@rollup/plugin-node-resolve",
        "//:node_modules/@rollup/plugin-terser",
        "//:node_modules/google-protobuf",
    ],
)

rollup_bundle(
    name = "audio_bundle_cjs",
    config_file = "//mediapipe/tasks/web:rollup.config.mjs",
    entry_point = "index.ts",
    format = "cjs",
    output_dir = False,
    sourcemap = "true",
    deps = [
        ":audio_lib",
        "//:node_modules/@rollup/plugin-commonjs",
        "//:node_modules/@rollup/plugin-node-resolve",
        "//:node_modules/@rollup/plugin-terser",
        "//:node_modules/google-protobuf",
    ],
)

genrule(
    name = "audio_sources",
    srcs = [
        ":audio_bundle_cjs",
        ":audio_bundle_mjs",
    ],
    outs = [
        "audio_bundle.cjs",
        "audio_bundle.cjs.map",
        "audio_bundle.mjs",
        "audio_bundle.mjs.map",
    ],
    cmd = (
        "for FILE in $(SRCS); do " +
        "  OUT_FILE=$(GENDIR)/mediapipe/tasks/web/audio/$$(" +
        "      basename $$FILE | sed -E 's/_([cm])js\\.js/.\\1js/'" +
        "  ); " +
        "  echo $$FILE ; echo $$OUT_FILE ; " +
        "  cp $$FILE $$OUT_FILE ; " +
        "done;"
    ),
)

pkg_npm(
    name = "audio_pkg",
    srcs = ["README.md"],
    package_json = "//mediapipe/tasks/web:package.json",
    substitutions = {
        "__NAME__": "audio",
        "__DESCRIPTION__": "MediaPipe Audio Tasks",
        "__TYPES__": "audio.d.ts",
    },
    deps = [
        "wasm/audio_wasm_internal.js",
        "wasm/audio_wasm_internal.wasm",
        "wasm/audio_wasm_module_internal.js",
        "wasm/audio_wasm_module_internal.wasm",
        "wasm/audio_wasm_nosimd_internal.js",
        "wasm/audio_wasm_nosimd_internal.wasm",
        ":audio_sources",
    ],
)
