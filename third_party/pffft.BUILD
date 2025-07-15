package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # FFTPACKv5 License.

exports_files(["LICENSE"])

config_setting(
    name = "android-armv7",
    constraint_values = [
        "@platforms//os:android",
        "@platforms//cpu:armv7",
    ],
)

cc_library(
    name = "pffft",
    srcs = ["pffft.c"],
    hdrs = ["pffft.h"],
    copts = select({
        "@bazel_tools//src/conditions:windows": [
            "/D_USE_MATH_DEFINES",
            "/W0",
        ],
        # pffft uses Neon. With NDK 28 and probably onward, we need to
        # pass the flags below to enable Neon on ARM v7.
        ":android-armv7": [
            "-mfpu=neon",
            "-mfloat-abi=softfp",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)
