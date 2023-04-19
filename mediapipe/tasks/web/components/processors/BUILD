# This package contains options shared by all MediaPipe Tasks for Web.

load("@npm//@bazel/jasmine:index.bzl", "jasmine_node_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_ts_library(
    name = "classifier_options",
    srcs = ["classifier_options.ts"],
    deps = [
        "//mediapipe/tasks/cc/components/processors/proto:classifier_options_jspb_proto",
        "//mediapipe/tasks/web/core:classifier_options",
    ],
)

mediapipe_ts_library(
    name = "classifier_options_test_lib",
    testonly = True,
    srcs = ["classifier_options.test.ts"],
    deps = [
        ":classifier_options",
        "//mediapipe/tasks/cc/components/processors/proto:classifier_options_jspb_proto",
        "//mediapipe/tasks/web/core:classifier_options",
    ],
)

jasmine_node_test(
    name = "classifier_options_test",
    deps = [":classifier_options_test_lib"],
)

mediapipe_ts_library(
    name = "classifier_result",
    srcs = ["classifier_result.ts"],
    deps = [
        "//mediapipe/framework/formats:classification_jspb_proto",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_jspb_proto",
        "//mediapipe/tasks/web/components/containers:classification_result",
    ],
)

mediapipe_ts_library(
    name = "classifier_result_test_lib",
    testonly = True,
    srcs = ["classifier_result.test.ts"],
    deps = [
        ":classifier_result",
        "//mediapipe/framework/formats:classification_jspb_proto",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_jspb_proto",
    ],
)

jasmine_node_test(
    name = "classifier_result_test",
    deps = [":classifier_result_test_lib"],
)

mediapipe_ts_library(
    name = "detection_result",
    srcs = ["detection_result.ts"],
    deps = [
        "//mediapipe/framework/formats:detection_jspb_proto",
        "//mediapipe/tasks/web/components/containers:detection_result",
    ],
)

mediapipe_ts_library(
    name = "detection_result_test_lib",
    testonly = True,
    srcs = ["detection_result.test.ts"],
    deps = [
        ":detection_result",
        "//mediapipe/framework/formats:detection_jspb_proto",
        "//mediapipe/framework/formats:location_data_jspb_proto",
        "//mediapipe/tasks/web/components/containers:detection_result",
    ],
)

mediapipe_ts_library(
    name = "embedder_result",
    srcs = ["embedder_result.ts"],
    deps = [
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_jspb_proto",
        "//mediapipe/tasks/web/components/containers:embedding_result",
    ],
)

mediapipe_ts_library(
    name = "embedder_result_test_lib",
    testonly = True,
    srcs = ["embedder_result.test.ts"],
    deps = [
        ":embedder_result",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_jspb_proto",
    ],
)

jasmine_node_test(
    name = "embedder_result_test",
    deps = [":embedder_result_test_lib"],
)

mediapipe_ts_library(
    name = "embedder_options",
    srcs = ["embedder_options.ts"],
    deps = [
        "//mediapipe/tasks/cc/components/processors/proto:embedder_options_jspb_proto",
        "//mediapipe/tasks/web/core:embedder_options",
    ],
)

mediapipe_ts_library(
    name = "embedder_options_test_lib",
    testonly = True,
    srcs = ["embedder_options.test.ts"],
    deps = [
        ":embedder_options",
        "//mediapipe/tasks/cc/components/processors/proto:embedder_options_jspb_proto",
        "//mediapipe/tasks/web/core:embedder_options",
    ],
)

jasmine_node_test(
    name = "embedder_options_test",
    deps = [":embedder_options_test_lib"],
)

mediapipe_ts_library(
    name = "landmark_result",
    srcs = [
        "landmark_result.ts",
        "landmark_result_test_lib.ts",
    ],
    deps = [
        "//mediapipe/framework/formats:landmark_jspb_proto",
        "//mediapipe/tasks/web/components/containers:landmark",
    ],
)

mediapipe_ts_library(
    name = "landmark_result_test_lib",
    testonly = True,
    srcs = ["landmark_result.test.ts"],
    deps = [":landmark_result"],
)

jasmine_node_test(
    name = "landmark_result_test",
    deps = [":landmark_result_test_lib"],
)
