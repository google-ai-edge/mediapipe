load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

package(default_visibility = ["//mediapipe/framework/api3:__subpackages__"])

licenses(["notice"])

mediapipe_proto_library(
    name = "foo_proto",
    testonly = True,
    srcs = ["foo.proto"],
)

mediapipe_proto_library(
    name = "bar_proto",
    testonly = True,
    srcs = ["bar.proto"],
)

mediapipe_proto_library(
    name = "generator_proto",
    srcs = ["generator.proto"],
    visibility = ["//mediapipe:__subpackages__"],
    deps = [
        "//mediapipe/framework:packet_generator_proto",
    ],
)

mediapipe_proto_library(
    name = "multiplier_proto",
    testonly = True,
    srcs = ["multiplier.proto"],
)
