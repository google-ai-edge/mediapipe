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
    srcs = ["cosine_similarity.test.ts"],
    deps = [
        ":cosine_similarity",
        "//mediapipe/tasks/web/components/containers:embedding_result",
    ],
)

jasmine_node_test(
    name = "cosine_similarity_test",
    deps = [":cosine_similarity_test_lib"],
)
