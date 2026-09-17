# Description:
#   Single-file C++ image decoding and encoding libraries

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # MIT license

COPTS = select({
    "@platforms//os:windows": [],
    "//conditions:default": [
        "-Wno-unused-function",
        "$(STACK_FRAME_UNLIMITED)",
    ],
})

cc_library(
    name = "stb_image",
    srcs = ["stb_image.c"],
    hdrs = ["stb_image.h"],
    copts = COPTS,
    includes = ["."],
)

cc_library(
    name = "stb_image_write",
    srcs = ["stb_image_write.c"],
    hdrs = ["stb_image_write.h"],
    copts = COPTS,
    includes = ["."],
)

cc_library(
    name = "stb_image_resize",
    hdrs = ["stb_image_resize2.h"],
    includes = ["."],
)

cc_library(
    name = "stblib",
    visibility = ["//visibility:public"],
    deps = [
        ":stb_image",
        ":stb_image_resize",
        ":stb_image_write",
    ],
)
