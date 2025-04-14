load("@npm//@bazel/jasmine:index.bzl", "jasmine_node_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_ts_library(
    name = "cosine_similarity",
    srcs = ["cosine_similarity.ts"],
    deps = [
        "//mediapipe/tasks/web/components/containers:embedding_result",
    ],
)

mediapipe_ts_library(
    name = "cosine_similarity_test_lib",
    testonly = True,
    srcs = ["cosine_similarity_test.ts"],
    deps = [
        ":cosine_similarity",
        "//mediapipe/tasks/web/components/containers:embedding_result",
    ],
)

jasmine_node_test(
    name = "cosine_similarity_test",
    srcs = [":cosine_similarity_test_lib"],
)

mediapipe_ts_library(
    name = "numeric_conversion",
    srcs = ["numeric_conversion.ts"],
    visibility = ["//mediapipe:__subpackages__"],
)

mediapipe_ts_library(
    name = "numeric_conversion_test_lib",
    testonly = True,
    srcs = ["numeric_conversion_test.ts"],
    deps = [":numeric_conversion"],
)

jasmine_node_test(
    name = "numeric_conversion_test",
    srcs = [":numeric_conversion_test_lib"],
)
