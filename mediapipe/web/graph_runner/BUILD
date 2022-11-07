# The TypeScript graph runner used by all MediaPipe Web tasks.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")

package(default_visibility = [
    ":internal",
    "//mediapipe/tasks:internal",
])

package_group(
    name = "internal",
    packages = [
        "//mediapipe/app/pursuit/wasm/web_ml_cpu/typescript/...",
    ],
)

mediapipe_ts_library(
    name = "wasm_mediapipe_lib_ts",
    srcs = [
        ":wasm_mediapipe_lib.ts",
    ],
    allow_unoptimized_namespaces = True,
)

mediapipe_ts_library(
    name = "wasm_mediapipe_image_lib_ts",
    srcs = [
        ":wasm_mediapipe_image_lib.ts",
    ],
    allow_unoptimized_namespaces = True,
    deps = [":wasm_mediapipe_lib_ts"],
)

mediapipe_ts_library(
    name = "register_model_resources_graph_service_ts",
    srcs = [
        ":register_model_resources_graph_service.ts",
    ],
    allow_unoptimized_namespaces = True,
    deps = [":wasm_mediapipe_lib_ts"],
)
