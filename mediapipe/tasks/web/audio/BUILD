# This contains the MediaPipe Audio Tasks.

load("@build_bazel_rules_nodejs//:index.bzl", "pkg_npm")
load("@npm//@bazel/rollup:index.bzl", "rollup_bundle")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")
load(
    "//mediapipe/framework/tool:mediapipe_files.bzl",
    "mediapipe_files",
)

package(default_visibility = ["//mediapipe/tasks:internal"])

AUDIO_LIBS = [
    "//mediapipe/tasks/web/audio/audio_classifier",
    "//mediapipe/tasks/web/audio/audio_embedder",
    "//mediapipe/tasks/web/core:fileset_resolver",
]

mediapipe_ts_library(
    name = "audio_lib",
    srcs = ["index.ts"],
    visibility = ["//visibility:public"],
    deps = AUDIO_LIBS,
)

mediapipe_ts_library(
    name = "audio_types",
    srcs = ["types.ts"],
    visibility = ["//visibility:public"],
    deps = AUDIO_LIBS,
)

mediapipe_files(srcs = [
    "wasm/audio_wasm_internal.js",
    "wasm/audio_wasm_internal.wasm",
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
        "@npm//@rollup/plugin-commonjs",
        "@npm//@rollup/plugin-node-resolve",
        "@npm//@rollup/plugin-terser",
        "@npm//google-protobuf",
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
        "@npm//@rollup/plugin-commonjs",
        "@npm//@rollup/plugin-node-resolve",
        "@npm//@rollup/plugin-terser",
        "@npm//google-protobuf",
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

genrule(
    name = "package_json",
    srcs = ["//mediapipe/tasks/web:package.json"],
    outs = ["package.json"],
    cmd = "cp $< $@",
)

pkg_npm(
    name = "audio_pkg",
    package_name = "@mediapipe/tasks-__NAME__",
    srcs = ["README.md"],
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
        ":audio_sources",
        ":package_json",
    ],
)
