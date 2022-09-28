package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # FFTPACKv5 License.

exports_files(["LICENSE"])

cc_library(
    name = "pffft",
    srcs = ["pffft.c"],
    hdrs = ["pffft.h"],
    copts = select({
        "@bazel_tools//src/conditions:windows": [
            "/D_USE_MATH_DEFINES",
            "/W0",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)
